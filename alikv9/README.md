# Animatrónico IA PRO

Versión reconstruida para que la cara se sienta más humana y más viva.

## Qué hace

- Usa la webcam para detectar rostro
- Estima mirada, sonrisa, apertura de ojos y emoción facial
- Cambia la mirada del animatrónico según tu presencia
- Divide la respuesta de la IA en segmentos emocionales
- Reproduce voz natural con cambios suaves
- Controla Arduino por Serial
- Mantiene joystick/manual override y parpadeo natural

## Estructura

```text
animatronico_proyecto/
├─ main.py
├─ vision.py
├─ voice.py
├─ serial_link.py
├─ ui.py
├─ install.py
├─ requirements.txt
├─ config.example.json
├─ arduino/
│  └─ animatronico_arduino.ino
└─ docs/
   └─ arquitectura.md
```

## Cómo funciona

1. `vision.py` mira la webcam y calcula métricas faciales.
2. `voice.py` le dice a la IA cómo responder en JSON.
3. `main.py` coordina webcam, voz, IA y Serial.
4. `ui.py` dibuja la interfaz moderna.
5. `arduino/animatronico_arduino.ino` mueve servos, ojos, párpados, cejas y mandíbula.

## Flujo

- Si aparece tu cara, el sistema ajusta contacto visual y emoción.
- Si hablás, la respuesta se parte en segmentos para que la emoción cambie de forma gradual.
- Mientras habla, el Arduino abre la mandíbula con un pulso suave para que parezca vivo.
- Si movés el joystick, vuelve a control manual.

## Instalación

```bash
python install.py
```

Después copiá `config.example.json` a `config.json` y completá la API key.

## Ejecutar

```bash
python main.py
```

## Nota importante

La emoción facial de webcam está calculada con heurísticas sobre landmarks faciales. No es un modelo clínico ni perfecto, pero sí sirve para dar una reacción mucho más humana y continua.
