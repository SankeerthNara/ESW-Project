"""
Converts a folder of calibration images into the raw float32 .raw tensors +
input_list.txt that qnn-onnx-converter / the quantizer tools expect.
"""
import argparse
import os
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--list_out", required=True)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    lines = []

    files = sorted(
        f for f in os.listdir(args.src)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not files:
        raise SystemExit(f"No images found in {args.src}")

    for f in files:
        img = Image.open(os.path.join(args.src, f)).convert("RGB")
        img = img.resize((args.width, args.height))
        arr = np.asarray(img).astype(np.float32) / 255.0        # HWC, 0-1
        arr = np.transpose(arr, (2, 0, 1))                       # CHW
        arr = np.expand_dims(arr, 0)                             # NCHW
        out_path = os.path.join(args.dst, f + ".raw")
        arr.astype(np.float32).tofile(out_path)
        lines.append(f"image:={out_path}")

    with open(args.list_out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} raw tensors + {args.list_out}")


if __name__ == "__main__":
    main()
