
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np


EMO_ICONS = ["😐","😊","😢","😠","😲","😴","😏","😱","🤔","😤","😂"]
EMO_NAMES = ["neutral","happy","sad","angry","surprise","sleepy","coqueto","afraid","confused","proud","laugh"]


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master, config: Dict[str, Any], on_save: Callable[[Dict[str, Any]], None]):
        super().__init__(master)
        self.title("Configuración avanzada")
        self.geometry("560x680")
        self.resizable(False, False)
        self.config_ref = dict(config)
        self.on_save = on_save
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Configuración", font=("Arial", 20, "bold")).pack(pady=14)
        frame = ctk.CTkScrollableFrame(self)
        frame.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        self.entries = {}

        fields = [
            ("API Key Groq", "api_key"),
            ("Puerto serial", "serial_port"),
            ("Baud rate", "baud_rate"),
            ("Modelo", "model"),
            ("Voz", "voice"),
            ("Idioma STT", "listen_lang"),
            ("Nombre", "persona_name"),
            ("Descripción", "persona_desc"),
            ("Cámara", "camera_index"),
            ("Mirada escuchando %", "gaze_listening"),
            ("Mirada hablando %", "gaze_speaking"),
            ("Smoothing rostro", "face_smoothing"),
            ("Smoothing emoción", "emotion_smoothing"),
        ]

        for label, key in fields:
            ctk.CTkLabel(frame, text=label, anchor="w").pack(anchor="w", pady=(8, 0))
            e = ctk.CTkEntry(frame, width=480)
            e.insert(0, str(self.config_ref.get(key, "")))
            e.pack()
            self.entries[key] = e

        ctk.CTkButton(self, text="Guardar", height=40, command=self._save).pack(pady=12)

    def _save(self):
        new_cfg = dict(self.config_ref)
        for key, entry in self.entries.items():
            value = entry.get().strip()
            if key in {"baud_rate", "camera_index", "gaze_listening", "gaze_speaking"}:
                try:
                    new_cfg[key] = int(value)
                except Exception:
                    pass
            elif key in {"face_smoothing", "emotion_smoothing"}:
                try:
                    new_cfg[key] = float(value)
                except Exception:
                    pass
            else:
                new_cfg[key] = value
        self.on_save(new_cfg)
        self.destroy()


class AppUI(ctk.CTk):
    def __init__(
        self,
        app_title: str,
        persona_name: str,
        on_send_text: Callable[[str], None],
        on_listen: Callable[[], None],
        on_interrupt: Callable[[], None],
        on_open_settings: Callable[[], None],
        on_clear_chat: Callable[[], None],
        on_refresh_camera: Callable[[], None],
        on_emotion_manual: Callable[[str], None],
        on_mic_toggle: Callable[[], None],
    ):
        super().__init__()
        self._title = app_title
        self.title(app_title)
        self.geometry("1240x820")
        self.minsize(1100, 720)

        self.on_send_text = on_send_text
        self.on_listen = on_listen
        self.on_interrupt = on_interrupt
        self.on_open_settings = on_open_settings
        self.on_clear_chat = on_clear_chat
        self.on_refresh_camera = on_refresh_camera
        self.on_emotion_manual = on_emotion_manual
        self.on_mic_toggle = on_mic_toggle

        self.camera_img = None
        self._build_ui(persona_name)

    def set_title(self, title: str) -> None:
        self.title(title)

    def _build_ui(self, persona_name: str):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, corner_radius=22)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(self, corner_radius=22)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(left, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header, text=f"🤖 {persona_name}", font=("Arial", 24, "bold")).grid(row=0, column=0, sticky="w")
        self.serial_state = ctk.CTkLabel(header, text="● Desconectado", text_color="#ff6666", font=("Arial", 12, "bold"))
        self.serial_state.grid(row=0, column=1, sticky="w", padx=14)
        self.override_state = ctk.CTkLabel(header, text="", text_color="#ffaa00", font=("Arial", 12))
        self.override_state.grid(row=0, column=2, sticky="w", padx=10)

        ctk.CTkButton(header, text="⚙", width=40, command=self.on_open_settings).grid(row=0, column=3, padx=(8, 0))
        ctk.CTkButton(header, text="📷", width=40, command=self.on_refresh_camera).grid(row=0, column=4, padx=(8, 0))

        # Chat
        self.chat = ctk.CTkTextbox(left, wrap="word", font=("Consolas", 13), fg_color="#0f1117", text_color="#e8e8ea")
        self.chat.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        self.chat.tag_config("user", foreground="#6ad7ff")
        self.chat.tag_config("ai", foreground="#8cff9c")
        self.chat.tag_config("sys", foreground="#8a8a8a")
        self.chat.tag_config("emo", foreground="#ffbd59")
        self.chat.configure(state="disabled")

        # Input
        input_box = ctk.CTkFrame(left, fg_color="transparent")
        input_box.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        input_box.grid_columnconfigure(0, weight=1)
        self.input = ctk.CTkEntry(input_box, placeholder_text="Escribí o hablá con la cámara activa…", height=42)
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.input.bind("<Return>", self._send_text)

        ctk.CTkButton(input_box, text="Enviar", width=110, height=42, command=self._send_text).grid(row=0, column=1)

        # Buttons row
        buttons = ctk.CTkFrame(left, fg_color="transparent")
        buttons.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        buttons.grid_columnconfigure((0, 1, 2), weight=1)

        self.mic_btn = ctk.CTkButton(buttons, text="🎤 Hablar", height=46, command=self.on_mic_toggle)
        self.mic_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(buttons, text="⏹ Interrumpir", height=46, command=self.on_interrupt).grid(row=0, column=1, sticky="ew", padx=8)
        ctk.CTkButton(buttons, text="🗑 Limpiar", height=46, command=self.on_clear_chat).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        self.meter = ctk.CTkProgressBar(left, width=200)
        self.meter.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 12))
        self.meter.set(0.0)

        # Right side camera + stats
        ctk.CTkLabel(right, text="Visión en vivo", font=("Arial", 18, "bold")).grid(row=0, column=0, pady=(14, 6))
        self.camera_label = ctk.CTkLabel(right, text="", width=520, height=320)
        self.camera_label.grid(row=1, column=0, padx=12, pady=(0, 8))

        stats = ctk.CTkFrame(right, corner_radius=18)
        stats.grid(row=2, column=0, sticky="ew", padx=12, pady=8)
        stats.grid_columnconfigure(1, weight=1)

        self.face_state = ctk.CTkLabel(stats, text="Cara: sin detectar", anchor="w")
        self.face_state.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 2))
        self.expr_state = ctk.CTkLabel(stats, text="Emoción: neutral", anchor="w")
        self.expr_state.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=2)
        self.gaze_state = ctk.CTkLabel(stats, text="Contacto visual: 0%", anchor="w")
        self.gaze_state.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 10))

        self.gaze_bar = ctk.CTkProgressBar(stats)
        self.gaze_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        self.gaze_bar.set(0.0)

        # manual emotion
        emo_box = ctk.CTkFrame(right, corner_radius=18)
        emo_box.grid(row=3, column=0, sticky="ew", padx=12, pady=8)
        emo_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(emo_box, text="Emoción manual", font=("Arial", 14, "bold")).grid(row=0, column=0, pady=(10, 4))
        self.emo_var = ctk.StringVar(value="neutral")
        self.emo_menu = ctk.CTkOptionMenu(emo_box, variable=self.emo_var, values=EMO_NAMES, command=self.on_emotion_manual)
        self.emo_menu.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="ew")

        # sliders
        sliders = ctk.CTkFrame(right, corner_radius=18)
        sliders.grid(row=4, column=0, sticky="ew", padx=12, pady=8)
        sliders.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sliders, text="Contacto visual al escuchar").grid(row=0, column=0, pady=(10, 2))
        self.listen_gaze = ctk.CTkSlider(sliders, from_=0, to=100)
        self.listen_gaze.set(64)
        self.listen_gaze.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))

        ctk.CTkLabel(sliders, text="Contacto visual al hablar").grid(row=2, column=0, pady=(4, 2))
        self.speak_gaze = ctk.CTkSlider(sliders, from_=0, to=100)
        self.speak_gaze.set(42)
        self.speak_gaze.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))

        self.listen_gaze.bind("<ButtonRelease-1>", lambda e: None)
        self.speak_gaze.bind("<ButtonRelease-1>", lambda e: None)

        self.status_line = ctk.CTkLabel(right, text="Listo", anchor="w")
        self.status_line.grid(row=5, column=0, sticky="ew", padx=12, pady=(8, 14))

        self._cam_photo = None

    def set_serial_state(self, connected: bool, port: str):
        self.serial_state.configure(
            text=f"● Conectado ({port})" if connected else "● Desconectado",
            text_color="#71ff99" if connected else "#ff6666"
        )

    def set_camera_state(self, active: bool):
        self.status_line.configure(text="Webcam activa" if active else "Webcam desactivada")

    def set_override_state(self, manual: bool):
        self.override_state.configure(text="🕹 Manual" if manual else "")

    def set_expression_label(self, emo_id: int):
        if 0 <= emo_id < len(EMO_NAMES):
            self.expr_state.configure(text=f"Emoción: {EMO_ICONS[emo_id]} {EMO_NAMES[emo_id]}")

    def set_expression_text(self, text: str):
        self.expr_state.configure(text=f"Emoción: {text}")

    def set_gaze_display(self, pct: int):
        self.gaze_bar.set(max(0, min(100, pct)) / 100.0)
        self.gaze_state.configure(text=f"Contacto visual: {int(pct)}%")

    def get_listen_gaze(self) -> int:
        try:
            return int(self.listen_gaze.get())
        except Exception:
            return 64

    def get_speak_gaze(self) -> int:
        try:
            return int(self.speak_gaze.get())
        except Exception:
            return 42

    def set_mic_state(self, active: bool):
        self.mic_btn.configure(text="🔴 Escuchando…" if active else "🎤 Hablar")

    def update_servo_view(self, lr: float, ud: float, emo_id: int, blinking: int):
        # placeholder for future custom indicators
        pass

    def log_system(self, text: str):
        self._log("Sistema", text, "sys")

    def log_user(self, text: str):
        self._log("Tú", text, "user")

    def log_ai(self, text: str, emotion: str):
        self._log("Aria", text, "ai", emotion)

    def _log(self, who: str, text: str, tag: str, emotion: str = ""):
        self.chat.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.chat.insert("end", f"[{ts}] {who}: ", tag)
        self.chat.insert("end", text + "\n")
        if emotion:
            self.chat.insert("end", f"        ↳ [{emotion}]\n", "emo")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def clear_chat(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")

    def open_settings(self, config: Dict[str, Any], on_save):
        SettingsWindow(self, config, on_save)

    def _send_text(self, event=None):
        text = self.input.get().strip()
        if not text:
            return
        self.input.delete(0, "end")
        self.on_send_text(text)

    def update_camera_frame(self, frame, metrics):
        if frame is None:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((520, 320))
        self._cam_photo = ImageTk.PhotoImage(img)
        self.camera_label.configure(image=self._cam_photo, text="")
        if metrics is not None:
            if metrics.face_present:
                self.face_state.configure(text=f"Cara: detectada | confianza {metrics.confidence:.2f}")
                self.set_gaze_display(int(metrics.gaze_contact * 100))
                self.set_expression_text(metrics.emotion)
            else:
                self.face_state.configure(text="Cara: sin detectar")
