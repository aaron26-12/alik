
from __future__ import annotations

import json
import re
from typing import Dict, List

EMOTION_TO_ID = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprise": 4,
    "sleepy": 5,
    "coqueto": 6,
    "afraid": 7,
    "confused": 8,
    "proud": 9,
    "laugh": 10,
    # aliases
    "feliz": 1,
    "triste": 2,
    "enojado": 3,
    "sorprendido": 4,
    "cansado": 5,
    "miedo": 7,
    "confundido": 8,
    "orgulloso": 9,
    "risa": 10,
}

PRIMARY_EMOTIONS = list({
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4,
    "sleepy": 5, "coqueto": 6, "afraid": 7, "confused": 8, "proud": 9, "laugh": 10
}.keys())


def build_system_prompt(name: str, desc: str) -> str:
    return f"""
Eres {name}, {desc}.
Tu voz debe sentirse humana, cálida y expresiva.

Responde SIEMPRE con JSON estricto y sin texto alrededor.
Debes hablar en segmentos cortos para que la emoción cambie de forma suave y gradual.

Esquema:
{{
  "segments": [
    {{
      "text": "Texto corto",
      "emotion": "neutral|happy|sad|angry|surprise|sleepy|coqueto|afraid|confused|proud|laugh",
      "blend": "emoción secundaria opcional",
      "blend_amount": 0-100,
      "gaze": "direct|avert"
    }}
  ],
  "total_mood": "emoción general"
}}

Reglas:
- Máximo 2 o 3 frases por segmento.
- Usa cambios sutiles entre segmentos.
- Si el usuario está serio, no exageres de golpe.
- Si conviene, mezcla emociones con blend_amount.
- "direct" cuando sostienes contacto visual, "avert" cuando pensás o recordás algo.
- Evita markdown, listas y explicaciones fuera del JSON.
""".strip()


def normalize_model_json(raw: str) -> Dict:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: extract JSON object if the model wrapped text around it
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise

    segs = data.get("segments", [])
    if not isinstance(segs, list):
        segs = []

    normalized = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        emo = str(seg.get("emotion", "neutral")).lower().strip()
        if emo not in EMOTION_TO_ID:
            emo = "neutral"
        normalized.append({
            "text": str(seg.get("text", "")).strip(),
            "emotion": emo,
            "emotion_id": EMOTION_TO_ID.get(emo, 0),
            "blend": str(seg.get("blend", "")).lower().strip(),
            "blend_amount": int(seg.get("blend_amount", 0) or 0),
            "gaze": "avert" if str(seg.get("gaze", "direct")).lower().strip() == "avert" else "direct",
        })

    data["segments"] = normalized
    data["total_mood"] = str(data.get("total_mood", "neutral")).lower().strip()
    if data["total_mood"] not in EMOTION_TO_ID:
        data["total_mood"] = "neutral"
    return data


def segment_response(text: str) -> List[str]:
    chunks = []
    for part in re.split(r"([,;:—–])", text):
        if not part:
            continue
        part = part.strip()
        if not part:
            continue
        if part in ",;:—–":
            if chunks:
                chunks[-1] += part
            continue
        sub = re.split(r"(?<=[.!?])\s+", part)
        for s in sub:
            s = s.strip()
            if s:
                chunks.append(s)
    return chunks
