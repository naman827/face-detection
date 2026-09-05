"""
Real-time face mesh + Gemini-powered face attribute analysis.

Pipeline:
  1. OpenCV grabs webcam frames.
  2. MediaPipe's FaceLandmarker task (the current Tasks API — the old
     `mp.solutions.face_mesh` API this was originally built on was
     superseded in 2023 and no longer receives updates) finds a 468-point
     face mesh per frame.
  3. Periodically, a cropped face image is sent to Gemini in a background
     thread to estimate age / gender / expression, using structured
     (Pydantic-schema) output so we never hand-parse JSON.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in GEMINI_API_KEY

Run:
    python face_analysis_app.py
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from google import genai
from google.genai import types as genai_types

# ==========================================
# Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ==========================================
# Structured output schema for Gemini
# ==========================================
class FaceAttributes(BaseModel):
    """Schema Gemini is asked to fill in. Using response_schema means the
    SDK validates and parses the JSON for us — no manual json.loads()."""

    age: str = Field(description="Estimated age, e.g. a number or a range like '18-23'.")
    gender: str = Field(description="Perceived gender presentation, e.g. 'Man' or 'Woman'.")
    expression: str = Field(description="Facial expression, e.g. 'Neutral', 'Happy', 'Sad', 'Angry'.")


# ==========================================
# Configuration & Constants
# ==========================================
@dataclass
class Config:
    API_COOLDOWN_SECONDS: float = 4.0
    CAMERA_INDEX: int = 0
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)  # BGR White
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.7
    FONT_THICKNESS: int = 2
    FACE_CROP_PADDING: int = 30
    MIN_FACE_CONFIDENCE: float = 0.5

    GEMINI_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))

    # NOTE: gemini-1.5-flash / gemini-2.0-flash have been retired by Google,
    # and gemini-2.5-flash is scheduled for shutdown in Oct 2026 — override
    # with the GEMINI_MODEL env var if this name is retired by the time you
    # read this (check https://ai.google.dev/gemini-api/docs/models).

    FACE_LANDMARKER_MODEL_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    FACE_LANDMARKER_MODEL_PATH: Path = Path(__file__).parent / "models" / "face_landmarker.task"


# ==========================================
# Main Application Class
# ==========================================
class FaceAnalysisApp:
    def __init__(self) -> None:
        """Initializes the application, loads config, and sets up models."""
        self.config = Config()
        self._setup_api()
        self._setup_face_landmarker()

        # Thread-safe state management
        self.state_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini-analysis")
        self._pending_future: Optional[Future] = None
        self.last_api_call_time = 0.0
        self.current_analysis: Dict[str, str] = {
            "age": "Analyzing...",
            "gender": "Analyzing...",
            "expression": "Analyzing...",
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_api(self) -> None:
        """Validates the environment and configures the Google GenAI Client."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables or .env file.")
            raise ValueError("Missing API Key")

        self.client = genai.Client(api_key=api_key)
        logger.info("Gemini API client initialized (model=%s).", self.config.GEMINI_MODEL)

    def _ensure_face_landmarker_model(self) -> Path:
        """Downloads MediaPipe's face_landmarker.task bundle once and caches it locally."""
        model_path = self.config.FACE_LANDMARKER_MODEL_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not model_path.exists() or model_path.stat().st_size == 0:
            logger.info("Downloading Face Landmarker model bundle (one-time, ~4 MB)...")
            try:
                urllib.request.urlretrieve(self.config.FACE_LANDMARKER_MODEL_URL, model_path)
            except Exception as exc:
                raise RuntimeError(
                    "Could not download the MediaPipe Face Landmarker model. "
                    "Check your internet connection, or download it manually from "
                    f"{self.config.FACE_LANDMARKER_MODEL_URL} to {model_path}."
                ) from exc
            logger.info("Model saved to %s", model_path)

        return model_path

    def _setup_face_landmarker(self) -> None:
        """Initializes MediaPipe's current FaceLandmarker Tasks API.

        `mp.solutions.face_mesh` (the legacy API this project used to rely
        on) was deprecated in 2023 in favor of `mediapipe.tasks.vision`.
        """
        model_path = self._ensure_face_landmarker_model()

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.MIN_FACE_CONFIDENCE,
            min_face_presence_confidence=self.config.MIN_FACE_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_FACE_CONFIDENCE,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # As of mediapipe 0.10.x, the legacy `mp.solutions` drawing helpers
        # have been removed from the PyPI wheel entirely — the equivalents
        # now live under mediapipe.tasks.python.vision.
        self.mp_drawing = mp_vision.drawing_utils
        self.mp_drawing_styles = mp_vision.drawing_styles
        self.face_mesh_connections = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

        logger.info("MediaPipe Face Landmarker initialized.")

    # ------------------------------------------------------------------
    # Gemini analysis (runs on a background thread)
    # ------------------------------------------------------------------
    def _analyze_face(self, face_image: np.ndarray) -> Optional[Dict[str, str]]:
        """Sends a cropped face crop to Gemini and returns parsed attributes, or None on failure."""
        try:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_face)

            prompt = (
                "Analyze this cropped human face and estimate its age, "
                "perceived gender presentation, and facial expression."
            )

            response = self.client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                contents=[prompt, pil_image],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FaceAttributes,
                ),
            )

            attributes = response.parsed
            if attributes is None:
                logger.warning("Gemini response could not be parsed into FaceAttributes: %r", response.text)
                return None

            return {
                "age": attributes.age,
                "gender": attributes.gender,
                "expression": attributes.expression,
            }

        except Exception as exc:  # Network errors, API errors, decode errors, etc.
            logger.error("Face analysis request failed: %s", exc)
            return None

    def _on_analysis_done(self, future: Future) -> None:
        """Callback invoked (on a worker thread) once a Gemini request finishes."""
        try:
            result = future.result()
        except Exception as exc:  # Should already be caught in _analyze_face, but be defensive.
            logger.exception("Unexpected error in analysis future: %s", exc)
            return

        if result is not None:
            with self.state_lock:
                self.current_analysis = result

    def _maybe_submit_analysis(self, face_image: np.ndarray) -> None:
        """Submits a background analysis job if not already running and cooldown has elapsed."""
        now = time.time()
        with self.state_lock:
            busy = self._pending_future is not None and not self._pending_future.done()
            cooled_down = (now - self.last_api_call_time) > self.config.API_COOLDOWN_SECONDS
            if busy or not cooled_down:
                return
            self.last_api_call_time = now

        future = self.executor.submit(self._analyze_face, face_image)
        future.add_done_callback(self._on_analysis_done)
        with self.state_lock:
            self._pending_future = future

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Processes a single video frame, draws the mesh, and triggers analysis if needed."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return frame

        face_landmarks = result.face_landmarks[0]  # list[NormalizedLandmark] for the first face

        # 1. Draw the geometric mesh over the face shape. The current Tasks API's
        # draw_landmarks takes the landmark list directly — no protobuf wrapping needed.
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.face_mesh_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

        # 2. Calculate bounding box for text placement / cropping.
        x_coords = [int(lm.x * w) for lm in face_landmarks]
        y_coords = [int(lm.y * h) for lm in face_landmarks]
        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

        # 3. Trigger analysis in the background if cooldown has passed and no job is running.
        pad = self.config.FACE_CROP_PADDING
        y1, y2 = max(0, y_min - pad), min(h, y_max + pad)
        x1, x2 = max(0, x_min - pad), min(w, x_max + pad)
        cropped_face = frame[y1:y2, x1:x2].copy()

        if cropped_face.size > 0:
            self._maybe_submit_analysis(cropped_face)

        # 4. Render overlay text below the face mesh.
        with self.state_lock:
            display_data = self.current_analysis.copy()

        text_x = x_min
        text_y_start = y_max + 30
        lines = [
            f"Gender: {display_data['gender']}",
            f"Age: {display_data['age']}",
            f"Expression: {display_data['expression']}",
        ]

        for idx, line in enumerate(lines):
            y_pos = text_y_start + (idx * 30)
            if y_pos < h:
                cv2.putText(
                    frame,
                    line,
                    (text_x, y_pos),
                    self.config.FONT,
                    self.config.FONT_SCALE,
                    self.config.TEXT_COLOR,
                    self.config.FONT_THICKNESS,
                )

        return frame

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Starts the main video loop."""
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)

        if not cap.isOpened():
            logger.error("Could not open webcam (index=%s).", self.config.CAMERA_INDEX)
            return

        logger.info("Application running. Press 'q' to quit.")
        start_time = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Dropped frame or camera disconnected.")
                    time.sleep(0.5)
                    continue

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                processed_frame = self._process_frame(frame, timestamp_ms)
                cv2.imshow("Production Face AI", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Shutdown signal received.")
                    break
        except KeyboardInterrupt:
            logger.info("Force quit detected.")
        except Exception as exc:
            logger.exception("Unexpected fatal error: %s", exc)
        finally:
            self.close(cap)

    def close(self, cap: Optional[cv2.VideoCapture] = None) -> None:
        """Releases all resources — safe to call multiple times."""
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        self.face_landmarker.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Resources released. Application closed.")


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    app: Optional[FaceAnalysisApp] = None
    try:
        app = FaceAnalysisApp()
        app.run()
    except Exception as initialization_error:
        logger.critical("Failed to start application: %s", initialization_error)