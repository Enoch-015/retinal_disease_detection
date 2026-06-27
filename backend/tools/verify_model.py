from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Keras and TFLite OCT predictions.")
    parser.add_argument("keras_model", type=Path)
    parser.add_argument("tflite_model", type=Path)
    parser.add_argument("images", nargs="*", type=Path)
    return parser.parse_args()


def load_batches(paths: list[Path]) -> list[tuple[str, np.ndarray]]:
    import cv2

    batches: list[tuple[str, np.ndarray]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        batches.append((path.name, image[None, ...]))

    if batches:
        return batches

    rng = np.random.default_rng(2026)
    return [
        ("black", np.zeros((1, 224, 224, 3), dtype=np.float32)),
        ("midgray", np.full((1, 224, 224, 3), 127, dtype=np.float32)),
        (
            "deterministic-noise",
            rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8).astype(np.float32),
        ),
    ]


def main() -> None:
    args = parse_args()

    import tensorflow as tf
    import tf_keras
    from tf_keras.applications import imagenet_utils

    keras_model = tf_keras.models.load_model(
        args.keras_model,
        compile=False,
        custom_objects={"imagenet_utils": imagenet_utils},
    )
    interpreter = tf.lite.Interpreter(model_path=str(args.tflite_model))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    failed = False
    for name, batch in load_batches(args.images):
        keras_probs = np.asarray(keras_model.predict(batch, verbose=0))[0]
        interpreter.set_tensor(input_index, batch)
        interpreter.invoke()
        lite_probs = np.asarray(interpreter.get_tensor(output_index))[0]

        keras_class = int(np.argmax(keras_probs))
        lite_class = int(np.argmax(lite_probs))
        max_delta = float(np.max(np.abs(keras_probs - lite_probs)))
        matches = keras_class == lite_class
        failed = failed or not matches
        print(
            f"{name}: class={keras_class}/{lite_class} "
            f"match={matches} max_probability_delta={max_delta:.6f}"
        )

    if failed:
        raise SystemExit("Parity check failed: at least one top class changed.")


if __name__ == "__main__":
    main()
