from __future__ import annotations

import numpy as np
import pytest

from app.inference import CLASS_NAMES, InferenceError, ModelService, preprocess_image_bytes


def test_preprocess_image_bytes_builds_rgb_resnet_batch():
    import cv2

    image = np.zeros((12, 10, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    ok, encoded = cv2.imencode(".png", image)

    assert ok

    batch = preprocess_image_bytes(encoded.tobytes())

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch[0, 0, 0].tolist() == [255.0, 0.0, 0.0]


def test_selects_oct_resnet_artifact(tmp_path):
    model_path = tmp_path / "oct_resnet50.tflite"
    model_path.write_text("model")

    service = ModelService(model_dir=tmp_path)

    assert service._select_model_path() == model_path


def test_configure_tensors_accepts_notebook_model_contract():
    interpreter = FakeInterpreter()
    service = ModelService(model_dir=None)  # type: ignore[arg-type]
    service.interpreter = interpreter

    service._configure_tensors()

    assert service.input_index == 4
    assert service.output_index == 8


def test_configure_tensors_rejects_wrong_output_shape():
    interpreter = FakeInterpreter()
    interpreter.output_details[0]["shape"] = np.array([1, 3])
    service = ModelService(model_dir=None, interpreter=interpreter)  # type: ignore[arg-type]

    with pytest.raises(InferenceError, match="output shape"):
        service._configure_tensors()


def test_predict_decodes_single_softmax_output():
    interpreter = FakeInterpreter()
    interpreter.output[0][2] = 0.91
    service = ModelService(model_dir=None)  # type: ignore[arg-type]
    service.interpreter = interpreter
    service.input_index = 4
    service.output_index = 8

    result = service.predict(_tiny_png_bytes())

    assert result["disease_class"] == "AMD_Severe"
    assert result["disease_confidence"] == pytest.approx(0.91)
    assert result["class_index"] == 2
    assert result["probabilities"]["AMD_Severe"] == pytest.approx(0.91)
    assert interpreter.invoked
    assert interpreter.input.shape == (1, 224, 224, 3)


class FakeInterpreter:
    def __init__(self):
        self.input_details = [
            {"index": 4, "shape": np.array([1, 224, 224, 3]), "dtype": np.float32}
        ]
        self.output_details = [
            {"index": 8, "shape": np.array([1, len(CLASS_NAMES)]), "dtype": np.float32}
        ]
        self.output = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
        self.input = None
        self.invoked = False

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details

    def set_tensor(self, index, value):
        assert index == 4
        self.input = value

    def invoke(self):
        self.invoked = True

    def get_tensor(self, index):
        assert index == 8
        return self.output


def _tiny_png_bytes() -> bytes:
    import cv2

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()
