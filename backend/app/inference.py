from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    "AMD_Mild",
    "AMD_Moderate",
    "AMD_Severe",
    "DME_Normal",
    "DME_Severe",
    "ERM_Normal",
    "NO_Normal",
    "RVO_Other",
    "RVO_Severe",
]


class InferenceError(RuntimeError):
    """Raised when inference cannot be completed."""


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


def split_prediction_label(label: str) -> tuple[str, str]:
    disease_class, separator, severity = label.rpartition("_")
    if not separator or not disease_class or not severity:
        raise InferenceError(f"Invalid model label format: {label}.")
    return disease_class, severity


def preprocess_image_bytes(image_bytes: bytes):
    """Match check.ipynb input shape: RGB image, 224x224, float32 batch."""
    if not image_bytes:
        raise InvalidImageError("No image bytes were received.")

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise InferenceError(f"Missing image preprocessing dependency: {exc.name}") from exc

    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("The uploaded file could not be decoded as an image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)


@dataclass
class ModelService:
    model_dir: Path
    interpreter: Any = None
    load_error: str | None = None
    model_path: Path | None = None
    input_index: int | None = None
    output_index: int | None = None
    _predict_lock: Lock = field(default_factory=Lock)

    @property
    def loaded(self) -> bool:
        return self.interpreter is not None and self.load_error is None

    def load(self) -> None:
        try:
            self.model_path = self._select_model_path()
            self.interpreter = self._load_interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            self._configure_tensors()
            self.load_error = None
        except Exception as exc:  # surface import/model incompatibilities through health
            self.interpreter = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _load_interpreter(model_path: Path) -> Any:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise InferenceError("LiteRT is not installed.") from exc
            return tf.lite.Interpreter(model_path=str(model_path), num_threads=1)

        return Interpreter(model_path=str(model_path), num_threads=1)

    def _select_model_path(self) -> Path:
        return self._select_artifact_path("oct_resnet50.tflite")

    def _select_artifact_path(self, *patterns: str) -> Path:
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(self.model_dir.glob(pattern))
        if not candidates:
            expected = ", ".join(patterns)
            raise FileNotFoundError(f"No model artifact found for: {expected}")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _configure_tensors(self) -> None:
        import numpy as np

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            raise InferenceError("Expected one model input and one model output.")

        input_shape = tuple(int(value) for value in input_details[0]["shape"])
        output_shape = tuple(int(value) for value in output_details[0]["shape"])
        if input_shape != (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3):
            raise InferenceError(
                f"Model input shape must be (1, 224, 224, 3), got {input_shape}."
            )
        if input_details[0]["dtype"] != np.float32:
            raise InferenceError(
                f"Model input dtype must be float32, got {input_details[0]['dtype']}."
            )
        if output_shape != (1, len(CLASS_NAMES)):
            raise InferenceError(
                "Model output shape does not match check.ipynb classes: "
                f"expected {len(CLASS_NAMES)}, got {output_shape}."
            )
        if output_details[0]["dtype"] != np.float32:
            raise InferenceError(
                f"Model output dtype must be float32, got {output_details[0]['dtype']}."
            )

        self.input_index = int(input_details[0]["index"])
        self.output_index = int(output_details[0]["index"])

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.loaded else "unavailable",
            "model_loaded": self.loaded,
            "model_dir": str(self.model_dir),
            "model_path": str(self.model_path) if self.model_path else None,
            "classes": CLASS_NAMES,
            "load_error": self.load_error,
        }

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.loaded:
            raise InferenceError(self.load_error or "Model is not loaded.")

        try:
            import numpy as np
        except ImportError as exc:
            raise InferenceError(f"Missing inference dependency: {exc.name}") from exc

        image = preprocess_image_bytes(image_bytes)

        with self._predict_lock:
            self.interpreter.set_tensor(self.input_index, image)
            self.interpreter.invoke()
            raw_outputs = self.interpreter.get_tensor(self.output_index)

        probs = np.asarray(raw_outputs, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[0] != 1:
            raise InferenceError(f"Unexpected model output shape: {probs.shape}.")

        class_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0][class_idx])
        combined_label = CLASS_NAMES[class_idx]
        disease_class, severity = split_prediction_label(combined_label)

        return {
            "disease_class": disease_class,
            "severity": severity,
            "combined_label": combined_label,
            "disease_confidence": confidence,
            "class_index": class_idx,
            "probabilities": {
                class_name: float(probs[0][index])
                for index, class_name in enumerate(CLASS_NAMES)
            },
            "warnings": [],
        }
