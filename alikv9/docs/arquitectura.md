# Arquitectura

Entrada del usuario
- Micrófono → SpeechRecognition
- Webcam → OpenCV + MediaPipe Face Mesh
- Texto → barra de chat

Cerebro
- Groq genera respuesta en JSON por segmentos
- Cada segmento tiene emoción, mezcla y tipo de mirada

Capa expresiva
- Voz natural con edge-tts
- Mirada y emoción se envían por Serial al Arduino
- La webcam ajusta contacto visual, microexpresiones y presencia

Salida física
- Arduino controla ojos, párpados, cejas, mandíbula y modo manual

Flujo resumido
1. Detectar rostro y métricas faciales.
2. Convertir métricas en emoción y mirada.
3. Consultar al modelo para responder.
4. Reproducir voz por segmentos.
5. Mandar emoción/mirada al Arduino con transiciones suaves.
