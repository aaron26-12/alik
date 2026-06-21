
from __future__ import annotations

import asyncio
import json
import os
import queue
import random
import re
import threading
import time
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import customtkinter as ctk
import pygame
import serial
import serial.tools.list_ports
from groq import Groq
import speech_recognition as sr
import edge_tts
from PIL import Image, ImageTk

from vision import WebcamTracker, FaceMetrics, emotion_label_from_id
from serial_link import SerialBridge
from voice import segment_response, build_system_prompt, normalize_model_json
from ui import AppUI

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
DEFAULTS_PATH = APP_DIR / "config.example.json"


def load_config() -> Dict[str, Any]:
    source = CONFIG_PATH if CONFIG_PATH.exists() else DEFAULTS_PATH
    with open(source, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


class AIEngine:
    def __init__(self, serial_bridge: SerialBridge, config: Dict[str, Any]):
        self.serial = serial_bridge
        self.config = config
        self.client: Optional[Groq] = None
        self.history: List[Dict[str, str]] = []
        self.speaking = False
        self.stop_flag = False
        self.system_prompt = build_system_prompt(config["persona_name"], config["persona_desc"])
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.system_prompt = build_system_prompt(config["persona_name"], config["persona_desc"])

    def init_client(self) -> None:
        if not self.config.get("api_key"):
            self.client = None
            return
        self.client = Groq(api_key=self.config["api_key"])

    def ask(self, user_text: str, on_segment=None, on_done=None, on_error=None) -> None:
        def worker() -> None:
            self.stop_flag = False
            try:
                self.history.append({"role": "user", "content": user_text})
                messages = [{"role": "system", "content": self.system_prompt}] + self.history[-10:]

                if self.client is None:
                    raw = json.dumps({
                        "segments": [{
                            "text": f"Modo sin API activo. Dijiste: {user_text}",
                            "emotion": "neutral",
                            "gaze": "direct"
                        }],
                        "total_mood": "neutral"
                    }, ensure_ascii=False)
                else:
                    response = self.client.chat.completions.create(
                        model=self.config["model"],
                        messages=messages,
                        temperature=0.84,
                        max_tokens=260,
                    )
                    raw = response.choices[0].message.content.strip()

                data = normalize_model_json(raw)
                segments = data.get("segments", [])
                total_mood = data.get("total_mood", "neutral")

                full_text = " ".join(seg.get("text", "") for seg in segments).strip()
                if full_text:
                    self.history.append({"role": "assistant", "content": full_text})

                if on_segment:
                    on_segment({"is_header": True, "emotion": total_mood, "text": ""})

                self.serial.send("STATE:speak")
                for seg in segments:
                    if self.stop_flag:
                        break
                    text = seg.get("text", "").strip()
                    if not text:
                        continue

                    emotion = seg.get("emotion", "neutral")
                    blend = seg.get("blend", "")
                    blend_amount = int(seg.get("blend_amount", 0) or 0)
                    gaze = seg.get("gaze", "direct")

                    self.serial.send(f"EMO:{seg.get('emotion_id', 0)}")
                    self.serial.send(f"GAZE:{self.config['gaze_speaking'] if gaze == 'direct' else 18}")

                    if blend and blend_amount > 0:
                        from voice import EMOTION_TO_ID
                        blend_id = EMOTION_TO_ID.get(str(blend).lower(), 0)
                        self.serial.send(f"BLEND:{blend_id},{blend_amount}")

                    if on_segment:
                        on_segment({
                            "text": text,
                            "emotion": emotion,
                            "blend": blend,
                            "blend_amount": blend_amount,
                            "gaze": gaze,
                        })

                    self._speak_microsegmented(text, emotion)
                    if self.stop_flag:
                        break
                    time.sleep(0.10)

                self.serial.send("STATE:listen")
                self.serial.send(f"GAZE:{self._current_listen_gaze()}")
                self.serial.send("EMO:0")
                self.serial.send("BLEND:0,0")
                if on_done:
                    on_done(full_text or user_text)
            except Exception as exc:
                if on_error:
                    on_error(str(exc))
            finally:
                self.speaking = False
                self.serial.send("STATE:idle")

        threading.Thread(target=worker, daemon=True).start()

    def interrupt(self) -> None:
        self.stop_flag = True
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def _speak_microsegmented(self, text: str, emotion: str) -> None:
        chunks = segment_response(text)
        if not chunks:
            chunks = [text]

        for idx, chunk in enumerate(chunks):
            if self.stop_flag:
                break
            self.speaking = True
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp = f.name
                communicate = edge_tts.Communicate(chunk, self.config["voice"])
                asyncio.run(communicate.save(tmp))
                pygame.mixer.music.load(tmp)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy() and not self.stop_flag:
                    time.sleep(0.03)
            finally:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                except Exception:
                    pass
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
                self.speaking = False


class VoiceEngine:
    def __init__(self, listen_lang: str):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 280
        self.listen_lang = listen_lang
        self.listening = False

    def listen_once(self, on_result, on_error) -> None:
        def worker() -> None:
            self.listening = True
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.45)
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                text = self.recognizer.recognize_google(audio, language=self.listen_lang)
                on_result(text)
            except sr.WaitTimeoutError:
                on_error("No se detectó voz.")
            except sr.UnknownValueError:
                on_error("No entendí lo que dijiste.")
            except Exception as exc:
                on_error(str(exc))
            finally:
                self.listening = False

        threading.Thread(target=worker, daemon=True).start()


class Controller:
    def __init__(self):
        self.config = load_config()
        self.serial = SerialBridge(self.config["serial_port"], self.config["baud_rate"])
        self.voice = VoiceEngine(self.config["listen_lang"])
        self.ai = AIEngine(self.serial, self.config)
        self.tracker = WebcamTracker(camera_index=self.config.get("camera_index", 0))
        self._live_gaze_listening = int(self.config.get("gaze_listening", 64))
        self._live_gaze_speaking = int(self.config.get("gaze_speaking", 42))

        self.serial.add_callback(self._on_serial_message)

        self.ui = AppUI(
            app_title=f"Animatrónico IA PRO — {self.config['persona_name']}",
            persona_name=self.config["persona_name"],
            on_send_text=self.process_text,
            on_listen=self.toggle_listen,
            on_interrupt=self.interrupt,
            on_open_settings=self.open_settings,
            on_clear_chat=self.clear_chat,
            on_refresh_camera=self.toggle_camera,
            on_emotion_manual=self.manual_emotion,
            on_mic_toggle=self.toggle_listen,
        )

        self._last_face_sent = 0.0
        self._last_status_sent = 0.0
        self._running = True
        self._open_camera()
        self._try_connect_serial()
        self.ai.init_client()
        self.ui.log_system("Sistema listo. Usá el micrófono, el chat o la webcam.")
        self.ui.after(60, self._tick_ui)

    def _open_camera(self) -> None:
        self.tracker.start()
        self.ui.set_camera_state(self.tracker.is_active)

    def toggle_camera(self) -> None:
        if self.tracker.is_active:
            self.tracker.stop()
        else:
            self.tracker.start()
        self.ui.set_camera_state(self.tracker.is_active)

    def _try_connect_serial(self) -> None:
        ports = self.serial.list_ports()
        desired = self.config["serial_port"]
        if desired not in ports and ports:
            desired = ports[0]
        ok = self.serial.connect(desired, self.config["baud_rate"])
        if ok:
            self.config["serial_port"] = desired
            self.ui.set_serial_state(True, desired)
            self.ui.log_system(f"Arduino conectado en {desired}.")
        else:
            self.ui.set_serial_state(False, "sin Arduino")
            self.ui.log_system("Arduino no encontrado. Sigue funcionando sin hardware.")

    def _on_serial_message(self, line: str) -> None:
        if line.startswith("STATUS:"):
            try:
                parts = dict(piece.split("=") for piece in line[7:].split(",") if "=" in piece)
                emo = int(parts.get("emo", 0))
                lr = float(parts.get("lr", 90))
                ud = float(parts.get("ud", 90))
                manual = int(parts.get("manual", 0))
                blinking = int(parts.get("blink", 0))
                self.ui.after(0, lambda: self.ui.update_servo_view(lr, ud, emo, blinking))
                self.ui.after(0, lambda: self.ui.set_override_state(manual == 1))
                self.ui.after(0, lambda: self.ui.set_expression_label(emo))
            except Exception:
                pass
        elif line == "READY":
            self.ui.after(0, lambda: self.ui.log_system("Arduino listo."))
        elif line.startswith("MANUAL:"):
            self.ui.after(0, lambda: self.ui.set_override_state(line.endswith("1")))

    def _current_listen_gaze(self) -> int:
        try:
            return int(self.ui.get_listen_gaze())
        except Exception:
            return int(self._live_gaze_listening)

    def _current_speak_gaze(self) -> int:
        try:
            return int(self.ui.get_speak_gaze())
        except Exception:
            return int(self._live_gaze_speaking)

    def manual_emotion(self, emotion_name: str) -> None:
        from voice import EMOTION_TO_ID
        emo_id = EMOTION_TO_ID.get(emotion_name, 0)
        self.serial.send(f"EMO:{emo_id}")
        self.ui.set_expression_label(emo_id)

    def process_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.ui.log_user(text)
        self.serial.send(f"GAZE:{self._current_speak_gaze()}")
        self.ai.ask(
            text,
            on_segment=self._on_ai_segment,
            on_done=self._on_ai_done,
            on_error=self._on_ai_error,
        )

    def toggle_listen(self) -> None:
        if self.voice.listening:
            return
        if self.ai.speaking:
            self.interrupt()
            time.sleep(0.2)
        self.serial.send("STATE:listen")
        self.serial.send(f"GAZE:{self._current_listen_gaze()}")
        self.ui.set_mic_state(True)
        self.ui.log_system("Escuchando...")
        self.voice.listen_once(self._on_voice_result, self._on_voice_error)

    def _on_voice_result(self, text: str) -> None:
        self.ui.after(0, lambda: self.ui.set_mic_state(False))
        self.ui.after(0, lambda: self.process_text(text))

    def _on_voice_error(self, message: str) -> None:
        self.ui.after(0, lambda: self.ui.set_mic_state(False))
        self.ui.after(0, lambda: self.ui.log_system(f"⚠️ {message}"))

    def _on_ai_segment(self, seg: Dict[str, Any]) -> None:
        if seg.get("is_header"):
            return
        text = seg.get("text", "")
        emotion = seg.get("emotion", "neutral")
        gaze = seg.get("gaze", "direct")
        self.ui.after(0, lambda: self.ui.log_ai(text, emotion))
        self.ui.after(0, lambda: self.ui.set_expression_text(emotion))
        gaze_pct = self._current_speak_gaze() if gaze == "direct" else 18
        self.ui.after(0, lambda: self.ui.set_gaze_display(gaze_pct))

    def _on_ai_done(self, full_text: str) -> None:
        self.ui.after(0, lambda: self.ui.set_mic_state(False))
        self.ui.after(0, lambda: self.ui.log_system("Respuesta completada."))

    def _on_ai_error(self, error: str) -> None:
        self.ui.after(0, lambda: self.ui.set_mic_state(False))
        self.ui.after(0, lambda: self.ui.log_system(f"❌ {error}"))

    def interrupt(self) -> None:
        self.ai.interrupt()
        self.serial.send("STATE:idle")
        self.serial.send("EMO:0")
        self.ui.set_mic_state(False)
        self.ui.log_system("Interrumpido.")

    def clear_chat(self) -> None:
        self.ui.clear_chat()
        self.ai.history.clear()

    def open_settings(self) -> None:
        self.ui.open_settings(self.config, self._save_settings)

    def _save_settings(self, new_config: Dict[str, Any]) -> None:
        self.config.update(new_config)
        self.ai.configure(self.config)
        self.ai.init_client()
        self.serial.disconnect()
        self._try_connect_serial()
        self.ui.set_title(f"Animatrónico IA PRO — {self.config['persona_name']}")
        self.ui.log_system("Configuración aplicada.")

    def _estimate_face_commands(self, metrics: Optional[FaceMetrics]) -> None:
        if metrics is None or not metrics.face_present or not self.serial.connected:
            return
        now = time.time()
        if now - self._last_face_sent < 0.08:
            return
        self._last_face_sent = now
        gaze_pct = int(max(0, min(100, metrics.gaze_contact * 100)))
        self.serial.send(f"GAZE:{gaze_pct}")
        self.serial.send(f"FACE:{metrics.face_lr:.1f},{metrics.face_ud:.1f},{metrics.confidence:.2f}")
        self.serial.send(f"EXP:{metrics.emotion_id}")

    def _tick_ui(self) -> None:
        if not self._running:
            return
        frame, metrics = self.tracker.read_frame()
        if frame is not None:
            self.ui.update_camera_frame(frame, metrics)
            self._estimate_face_commands(metrics)
        self.ui.after(33, self._tick_ui)

    def _on_camera_status(self, is_active: bool) -> None:
        self.ui.set_camera_state(is_active)

    def on_close(self) -> None:
        self._running = False
        self.tracker.stop()
        self.serial.disconnect()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        self.ui.destroy()


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    controller = Controller()
    controller.ui.protocol("WM_DELETE_WINDOW", controller.on_close)
    controller.ui.mainloop()


if __name__ == "__main__":
    main()
