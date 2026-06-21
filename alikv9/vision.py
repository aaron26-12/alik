
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


EMOTION_ID = {
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
}

EMOTION_ORDER = [
    "neutral", "happy", "sad", "angry", "surprise",
    "sleepy", "coqueto", "afraid", "confused", "proud", "laugh"
]


def emotion_label_from_id(idx: int) -> str:
    for k, v in EMOTION_ID.items():
        if v == idx:
            return k
    return "neutral"


@dataclass
class FaceMetrics:
    face_present: bool
    face_lr: float
    face_ud: float
    gaze_contact: float
    confidence: float
    emotion: str
    emotion_id: int
    emotion_scores: Dict[str, float]
    smile: float
    mouth_open: float
    eye_open: float
    brow_raise: float
    frame_w: int = 0
    frame_h: int = 0


class WebcamTracker:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.last_metrics: Optional[FaceMetrics] = None
        self._smooth: Optional[Dict[str, float]] = None
        self._last_frame = None

        if mp is not None:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mp_drawing = mp.solutions.drawing_utils
            self._mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._mp_face_mesh = None
            self._mesh = None

    def start(self) -> bool:
        if self.is_active:
            return True
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.is_active = bool(self.cap and self.cap.isOpened())
        return self.is_active

    def stop(self) -> None:
        self.is_active = False
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    @staticmethod
    def _lm_px(lms, idx: int, w: int, h: int):
        lm = lms[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    def _smooth_val(self, key: str, value: float, alpha: float = 0.2) -> float:
        if self._smooth is None:
            self._smooth = {}
        prev = self._smooth.get(key, value)
        out = prev * (1 - alpha) + value * alpha
        self._smooth[key] = out
        return out

    def read_frame(self):
        if not self.is_active or self.cap is None:
            return self._last_frame, self.last_metrics

        ok, frame = self.cap.read()
        if not ok:
            return self._last_frame, self.last_metrics

        frame = cv2.flip(frame, 1)
        self._last_frame = frame.copy()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._mesh is None:
            metrics = FaceMetrics(
                face_present=False,
                face_lr=90,
                face_ud=90,
                gaze_contact=0.0,
                confidence=0.0,
                emotion="neutral",
                emotion_id=0,
                emotion_scores={},
                smile=0.0,
                mouth_open=0.0,
                eye_open=0.0,
                brow_raise=0.0,
                frame_w=w,
                frame_h=h,
            )
            self.last_metrics = metrics
            return frame, metrics

        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            metrics = FaceMetrics(
                face_present=False,
                face_lr=90,
                face_ud=90,
                gaze_contact=0.0,
                confidence=0.0,
                emotion="neutral",
                emotion_id=0,
                emotion_scores={},
                smile=0.0,
                mouth_open=0.0,
                eye_open=0.0,
                brow_raise=0.0,
                frame_w=w,
                frame_h=h,
            )
            self.last_metrics = metrics
            return frame, metrics

        face = results.multi_face_landmarks[0].landmark

        # Key landmarks
        left_eye_outer = self._lm_px(face, 33, w, h)
        left_eye_inner = self._lm_px(face, 133, w, h)
        right_eye_inner = self._lm_px(face, 362, w, h)
        right_eye_outer = self._lm_px(face, 263, w, h)

        left_eye_top = self._lm_px(face, 159, w, h)
        left_eye_bottom = self._lm_px(face, 145, w, h)
        right_eye_top = self._lm_px(face, 386, w, h)
        right_eye_bottom = self._lm_px(face, 374, w, h)

        mouth_left = self._lm_px(face, 61, w, h)
        mouth_right = self._lm_px(face, 291, w, h)
        mouth_top = self._lm_px(face, 13, w, h)
        mouth_bottom = self._lm_px(face, 14, w, h)

        brow_left_inner = self._lm_px(face, 70, w, h)
        brow_right_inner = self._lm_px(face, 300, w, h)

        face_center_x = (left_eye_outer[0] + right_eye_outer[0]) * 0.5
        face_center_y = (left_eye_outer[1] + right_eye_outer[1]) * 0.5
        face_width = max(1.0, np.linalg.norm(right_eye_outer - left_eye_outer))
        face_height = max(1.0, np.linalg.norm(mouth_bottom - brow_left_inner))

        # Face position in frame
        face_lr = self._clamp01(face_center_x / w)
        face_ud = self._clamp01(face_center_y / h)

        # Eye openness / mouth / smile
        eye_open = (
            np.linalg.norm(left_eye_top - left_eye_bottom) / (np.linalg.norm(left_eye_outer - left_eye_inner) + 1e-5)
            + np.linalg.norm(right_eye_top - right_eye_bottom) / (np.linalg.norm(right_eye_outer - right_eye_inner) + 1e-5)
        ) * 0.5

        mouth_open = np.linalg.norm(mouth_top - mouth_bottom) / (np.linalg.norm(mouth_left - mouth_right) + 1e-5)
        smile = np.linalg.norm(mouth_left - mouth_right) / (np.linalg.norm(mouth_top - mouth_bottom) + 1e-5)

        brow_raise = (
            (brow_left_inner[1] - left_eye_top[1]) / (face_height + 1e-5) +
            (brow_right_inner[1] - right_eye_top[1]) / (face_height + 1e-5)
        ) * 0.5

        # Iris / gaze proxy if landmarks exist
        gaze_contact = 0.55
        iris_left = np.mean([self._lm_px(face, i, w, h) for i in range(468, 473)], axis=0) if len(face) > 472 else None
        iris_right = np.mean([self._lm_px(face, i, w, h) for i in range(473, 478)], axis=0) if len(face) > 477 else None

        if iris_left is not None and iris_right is not None:
            left_ratio = (iris_left[0] - left_eye_outer[0]) / (left_eye_inner[0] - left_eye_outer[0] + 1e-5)
            right_ratio = (iris_right[0] - right_eye_inner[0]) / (right_eye_outer[0] - right_eye_inner[0] + 1e-5)
            up_ratio_left = (iris_left[1] - left_eye_top[1]) / (left_eye_bottom[1] - left_eye_top[1] + 1e-5)
            up_ratio_right = (iris_right[1] - right_eye_top[1]) / (right_eye_bottom[1] - right_eye_top[1] + 1e-5)
            gaze_contact = 1.0 - (abs(left_ratio - 0.5) + abs(right_ratio - 0.5)) * 0.9
            gaze_contact = self._clamp01(gaze_contact)

        # Heuristic emotions
        scores = self._emotion_scores(smile, mouth_open, eye_open, brow_raise, gaze_contact)
        emotion = max(scores, key=scores.get)
        emotion_id = EMOTION_ID.get(emotion, 0)
        confidence = max(scores.values()) if scores else 0.0

        metrics = FaceMetrics(
            face_present=True,
            face_lr=self._smooth_val("face_lr", face_lr, 0.25),
            face_ud=self._smooth_val("face_ud", face_ud, 0.25),
            gaze_contact=self._smooth_val("gaze_contact", gaze_contact, 0.22),
            confidence=self._smooth_val("confidence", confidence, 0.18),
            emotion=emotion,
            emotion_id=emotion_id,
            emotion_scores=scores,
            smile=self._smooth_val("smile", smile, 0.2),
            mouth_open=self._smooth_val("mouth_open", mouth_open, 0.2),
            eye_open=self._smooth_val("eye_open", eye_open, 0.2),
            brow_raise=self._smooth_val("brow_raise", brow_raise, 0.2),
            frame_w=w,
            frame_h=h,
        )
        self.last_metrics = metrics

        # Overlay
        color = (0, 255, 160) if metrics.face_present else (100, 100, 100)
        x = int(metrics.face_lr * w)
        y = int(metrics.face_ud * h)
        cv2.circle(frame, (x, y), 8, color, 2)
        cv2.putText(frame, f"{metrics.emotion} ({metrics.confidence:.2f})", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        return frame, metrics

    def _emotion_scores(self, smile, mouth_open, eye_open, brow_raise, gaze_contact):
        scores = {
            "neutral": 0.55,
            "happy": 0.12 + smile * 1.8 + eye_open * 0.1,
            "sad": 0.08 + max(0.0, 0.45 - smile) * 1.0 + max(0.0, 0.30 - eye_open) * 1.0,
            "angry": 0.06 + max(0.0, 0.18 - brow_raise) * 2.2 + max(0.0, 0.32 - eye_open) * 0.8,
            "surprise": 0.10 + mouth_open * 1.7 + eye_open * 1.5 + max(0.0, brow_raise) * 0.7,
            "sleepy": 0.06 + max(0.0, 0.36 - eye_open) * 2.0 + max(0.0, 0.18 - mouth_open) * 0.5,
            "coqueto": 0.05 + smile * 1.0 + gaze_contact * 0.4,
            "afraid": 0.04 + eye_open * 1.5 + mouth_open * 0.9 + max(0.0, brow_raise) * 0.8,
            "confused": 0.05 + abs(gaze_contact - 0.5) * 0.8 + max(0.0, 0.24 - eye_open) * 0.4,
            "proud": 0.04 + max(0.0, 0.22 - brow_raise) * 0.6 + smile * 0.4,
            "laugh": 0.05 + smile * 2.0 + max(0.0, mouth_open - 0.08) * 0.5,
        }
        # Normalize a little
        peak = max(scores.values()) if scores else 1.0
        if peak > 0:
            scores = {k: v / peak for k, v in scores.items()}
        return scores
