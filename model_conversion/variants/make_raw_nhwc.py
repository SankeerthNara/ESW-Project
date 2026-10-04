"""Calibration tensors for SNPE quantization in the DLC's NHWC input layout.

snpe-onnx-to-dlc turns the ONNX NCHW input into an NHWC DLC input, and the
quantizer reads raw files in the DLC layout. (The older make_raw_inputs.py
wrote NCHW, so models calibrated with it saw scrambled images.)
"""
import argparse, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True, help="image directories")
ap.add_argument("--dst", required=True)
ap.add_argument("--size", type=int, required=True)
ap.add_argument("--normalize", choices=["unit", "imagenet"], required=True)
ap.add_argument("--input_name", required=True)
ap.add_argument("--list_out", required=True)
args = ap.parse_args()

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

os.makedirs(args.dst, exist_ok=True)
lines = []
for d in args.src:
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # Same stretch-to-square the app does (no letterbox).
        img = Image.open(os.path.join(d, f)).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
        if args.normalize == "imagenet":
            arr = (arr - MEAN) / STD
        out = os.path.join(args.dst, f"{os.path.basename(d)}_{f}.raw")
        arr[None].astype(np.float32).tofile(out)  # NHWC
        lines.append(f"{args.input_name}:={out}")

with open(args.list_out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{len(lines)} calibration tensors -> {args.list_out}")
