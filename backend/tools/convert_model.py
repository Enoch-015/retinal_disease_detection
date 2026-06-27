from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the OCT Keras model to a dynamically quantized TFLite model."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import tensorflow as tf
    import tf_keras
    from tf_keras.applications import imagenet_utils

    model = tf_keras.models.load_model(
        args.source,
        compile=False,
        custom_objects={"imagenet_utils": imagenet_utils},
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converted = converter.convert()

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(converted)
    print(f"Wrote {len(converted) / (1024 * 1024):.1f} MiB to {args.destination}")


if __name__ == "__main__":
    main()
