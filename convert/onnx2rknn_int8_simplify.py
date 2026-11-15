# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Minimal ONNX to RKNN INT8 converter for RK3588."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN

HERE = Path(__file__).resolve().parent
DEFAULT_ONNX = HERE / "rf-detr-small-nogs.onnx"
MEAN = [[123.675, 116.28, 103.53]]
STD = [[58.395, 57.12, 57.375]]


def main() -> int:
    """Quantize a fixed-shape RF-DETR ONNX model and export an RKNN file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Patched 1x3x512x512 ONNX model")
    parser.add_argument("--calib-dir", type=Path, required=True, help="Representative calibration image directory")
    parser.add_argument("--output", type=Path, help="Output path; defaults to <model>-int8.rknn")
    parser.add_argument("--log", type=Path, default=HERE / "rknn-int8-simplify.log", help="Toolkit log path")
    args = parser.parse_args()

    onnx_path = args.onnx.expanduser().resolve()
    calibration_dir = args.calib_dir.expanduser().resolve()
    if not onnx_path.is_file():
        parser.error(f"ONNX model not found: {onnx_path}; run patch_gridsample.py first")
    if not calibration_dir.is_dir() or not any(calibration_dir.iterdir()):
        parser.error(f"calibration directory is missing or empty: {calibration_dir}")

    output_path = args.output.expanduser().resolve() if args.output else HERE / f"{onnx_path.stem}-int8.rknn"
    dataset_path = HERE / "int8-simplify-dataset.txt"
    images = sorted(
        path.resolve()
        for path in calibration_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not images:
        parser.error(f"no supported calibration images found in: {calibration_dir}")
    dataset_path.write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.stem}.building{output_path.suffix}")
    rknn = RKNN(verbose=True, verbose_file=str(args.log.expanduser().resolve()))
    try:
        rknn.config(
            target_platform="rk3588",
            mean_values=MEAN,
            std_values=STD,
            quantized_algorithm="normal",
            quantized_method="channel",
        )
        rknn.load_onnx(model=str(onnx_path), inputs=["input"], input_size_list=[[1, 3, 512, 512]])
        rknn.build(do_quantization=True, dataset=str(dataset_path))
        rknn.export_rknn(str(temporary_output))
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("RKNN Toolkit did not produce a non-empty model")
        temporary_output.replace(output_path)
    finally:
        rknn.release()
        temporary_output.unlink(missing_ok=True)

    print(f"Exported {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MiB)")
    print(f"Calibration images: {len(images)}; toolkit log: {args.log.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
