import argparse, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--height", type=int, required=True)
ap.add_argument("--list_out", required=True)
ap.add_argument("--input_name", default="image")
ap.add_argument("--normalize", choices=["unit", "imagenet"], default="unit")
args = ap.parse_args()

os.makedirs(args.dst, exist_ok=True)
lines = []

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

files = sorted(f for f in os.listdir(args.src) if f.lower().endswith((".jpg",".jpeg",".png")))
for f in files:
    img = Image.open(os.path.join(args.src, f)).convert("RGB").resize((args.width, args.height))
    arr = np.asarray(img).astype(np.float32) / 255.0
    if args.normalize == "imagenet":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2,0,1))[None, ...]
    out_path = os.path.join(args.dst, f + ".raw")
    arr.astype(np.float32).tofile(out_path)
    lines.append(f"{args.input_name}:={out_path}")

with open(args.list_out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"Wrote {len(lines)} raw tensors + {args.list_out}")
