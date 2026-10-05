"""Accuracy benchmarks for the FPS / Balance / Quality models.

Depth: Depth Anything V2 Small on the NYU Depth v2 test split (654 images).
Seg:   YOLOv8n-seg on a fixed COCO val2017 subset (first N image ids).

Each model is scored twice with identical pre/post-processing (the app's
stretch-to-square resize, no letterbox):
  float  - float32 ONNX on the host CPU
  device - the shipped quantized DLC run on the QIDK HTP with snpe-net-run
           (native uint8/uint16 tensors, same encodings the apps use)

Run inside the QAIRT container (python from /opt/qnn_venv):
  python eval_tools.py nyu-prep
  python eval_tools.py depth-inputs  --variant balance          # -> raw files + list for the board
  python eval_tools.py depth-score   --variant balance [--device_out DIR]
  python eval_tools.py coco-inputs   --variant balance --n 500
  python eval_tools.py seg-score     --variant balance --n 500 [--device_out DIR]
"""
import argparse, glob, io, json, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
VARIANTS = os.path.join(HERE, "..", "variants")
NYU_DIR = os.path.join(DATA, "nyu", "test")
COCO = os.path.join(DATA, "coco")

DEPTH_ONNX = {"fps": "build/depth_280.onnx", "balance": "build/depth_392.onnx", "quality": "build/depth_518.onnx"}
SEG_ONNX = {"fps": "/work/yolov8n_seg_384.onnx", "balance": "/work/yolov8n_seg_decoupled.onnx",
            "quality": "/work/yolov8n_seg_decoupled.onnx"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def encodings(variant, kind):
    return json.load(open(os.path.join(VARIANTS, "out", f"{variant}_{kind}.json")))


def quantize(x, enc):
    q = np.round(x / enc["scale"]) - enc["offset"]
    dt = np.uint8 if enc["bits"] == 8 else np.uint16
    return np.clip(q, 0, np.iinfo(dt).max).astype(dt)


def dequantize(path, enc):
    dt = np.uint8 if enc["bits"] == 8 else np.uint16
    q = np.fromfile(path, dt).astype(np.float32)
    return ((q + enc["offset"]) * enc["scale"]).reshape(enc["shape"])


# ----------------------------------------------------------------- depth ---
def nyu_prep(_):
    import pyarrow.parquet as pq
    os.makedirs(NYU_DIR, exist_ok=True)
    i = 0
    for f in sorted(glob.glob(os.path.join(DATA, "nyu", "val_*.parquet"))):
        for row in pq.read_table(f).to_pylist():
            Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(f"{NYU_DIR}/{i:04d}.png")
            np.save(f"{NYU_DIR}/{i:04d}.npy", np.array(Image.open(io.BytesIO(row["depth_map"]["bytes"])), np.float32))
            i += 1
    print(f"{i} NYU test samples -> {NYU_DIR}")


def nyu_ids():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{NYU_DIR}/*.png"))


def depth_input(img, size):
    arr = np.asarray(img.resize((size, size), Image.BILINEAR), np.float32) / 255.0
    return (arr - MEAN) / STD  # HWC


def depth_inputs(a):
    enc = encodings(a.variant, "depth")["inputs"]["image"]
    size = enc["shape"][1]
    d = os.path.join(OUT, f"{a.variant}_depth_in")
    os.makedirs(d, exist_ok=True)
    lines = []
    for i in nyu_ids():
        quantize(depth_input(Image.open(f"{NYU_DIR}/{i}.png"), size), enc).tofile(f"{d}/{i}.raw")
        lines.append(f"eval/{a.variant}_depth_in/{i}.raw")
    open(f"{d}/list.txt", "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} inputs -> {d}")


def depth_metrics(disp, gt):
    """Affine-invariant protocol used by MiDaS / Depth Anything for relative
    models: least-squares scale+shift in disparity space, Eigen crop, 1e-3..10 m."""
    import cv2
    disp = cv2.resize(disp, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros_like(gt, bool)
    mask[45:471, 41:601] = True
    mask &= (gt > 1e-3) & (gt < 10)
    p, t = disp[mask], 1.0 / gt[mask]
    s, sh = np.linalg.lstsq(np.stack([p, np.ones_like(p)], 1), t, rcond=None)[0]
    pred = 1.0 / np.clip(s * p + sh, 1 / 10, None)
    g = gt[mask]
    ratio = np.maximum(pred / g, g / pred)
    return np.mean(np.abs(pred - g) / g), np.sqrt(np.mean((pred - g) ** 2)), np.mean(ratio < 1.25)


def depth_score(a):
    enc = encodings(a.variant, "depth")
    size = enc["inputs"]["image"]["shape"][1]
    if not a.device_out:
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(VARIANTS, DEPTH_ONNX[a.variant]))
    m = []
    for k, i in enumerate(nyu_ids()):
        gt = np.load(f"{NYU_DIR}/{i}.npy")
        if a.device_out:
            disp = dequantize(f"{a.device_out}/Result_{k}/depth.raw", enc["outputs"]["depth"])[0]
        else:
            x = depth_input(Image.open(f"{NYU_DIR}/{i}.png"), size).transpose(2, 0, 1)[None]
            disp = sess.run(None, {"image": x})[0][0]
        m.append(depth_metrics(disp.astype(np.float32), gt))
    absrel, rmse, d1 = np.mean(m, 0)
    res = {"variant": a.variant, "size": size, "mode": "device" if a.device_out else "float",
           "n": len(m), "AbsRel": round(float(absrel), 4), "RMSE_m": round(float(rmse), 4), "delta1": round(float(d1), 4)}
    print(json.dumps(res))
    with open(os.path.join(OUT, "results.jsonl"), "a") as f:
        f.write(json.dumps({"task": "depth", **res}) + "\n")


# ------------------------------------------------------------------- seg ---
def coco_subset(n):
    from pycocotools.coco import COCO as C
    gt = C(os.path.join(COCO, "annotations", "instances_val2017.json"))
    return gt, sorted(gt.getImgIds())[:n]


def seg_input(img, size):
    return np.asarray(img.resize((size, size), Image.BILINEAR), np.float32) / 255.0  # HWC 0-1


def coco_inputs(a):
    enc = encodings(a.variant, "seg")["inputs"]["images"]
    size = enc["shape"][1]
    gt, ids = coco_subset(a.n)
    d = os.path.join(OUT, f"{a.variant}_seg_in")
    os.makedirs(d, exist_ok=True)
    lines = []
    for i in ids:
        img = Image.open(os.path.join(COCO, "coco", "images", "val2017", gt.imgs[i]["file_name"])).convert("RGB")
        quantize(seg_input(img, size), enc).tofile(f"{d}/{i}.raw")
        lines.append(f"eval/{a.variant}_seg_in/{i}.raw")
    open(f"{d}/list.txt", "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} inputs -> {d}")


def decode(boxes, scores, coeffs, proto, size, W, H, conf=0.001, iou=0.7, max_det=300):
    """ultralytics-val style: best class per anchor, class-aware NMS, masks from
    32 prototypes cropped to the box, mapped back through the stretch."""
    import torch, torchvision
    import torch.nn.functional as F
    cls = scores.argmax(0); sc = scores.max(0)
    keep = np.where(sc > conf)[0]
    if keep.size == 0:
        return []
    keep = keep[np.argsort(-sc[keep])][:30000]
    cx, cy, w, h = boxes[:, keep]
    xyxy = torch.tensor(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1))
    k = torchvision.ops.batched_nms(xyxy, torch.tensor(sc[keep]), torch.tensor(cls[keep]), iou)[:max_det].numpy()
    sel = keep[k]; bx = xyxy[k].numpy()
    P = proto.shape[0]  # proto is HWC here
    m = torch.sigmoid(torch.tensor(coeffs[:, sel].T) @ torch.tensor(proto.reshape(-1, 32)).T).reshape(-1, P, P)
    r = P / size
    ys = torch.arange(P)[None, :, None].float(); xs = torch.arange(P)[None, None, :].float()
    b = torch.tensor(bx * r)
    inside = (xs >= b[:, 0, None, None]) & (xs < b[:, 2, None, None]) & (ys >= b[:, 1, None, None]) & (ys < b[:, 3, None, None])
    m = F.interpolate((m * inside)[None], size=(H, W), mode="bilinear", align_corners=False)[0] > 0.5
    sx, sy = W / size, H / size
    out = []
    for j, a in enumerate(sel):
        x1, y1, x2, y2 = bx[j]
        out.append((int(cls[a]), float(sc[a]), [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy], m[j].numpy()))
    return out


def seg_score(a):
    from pycocotools import mask as mu
    from pycocotools.cocoeval import COCOeval
    enc = encodings(a.variant, "seg")
    size = enc["inputs"]["images"]["shape"][1]
    gt, ids = coco_subset(a.n)
    cat_ids = sorted(gt.getCatIds())
    if not a.device_out:
        import onnxruntime as ort
        sess = ort.InferenceSession(SEG_ONNX[a.variant])
    dets = []
    for k, i in enumerate(ids):
        info = gt.imgs[i]; W, H = info["width"], info["height"]
        if a.device_out:
            o = {n: dequantize(f"{a.device_out}/Result_{k}/{n}.raw", e)[0] for n, e in enc["outputs"].items()}
            proto = o["proto"]  # NHWC from the DLC
        else:
            img = Image.open(os.path.join(COCO, "coco", "images", "val2017", info["file_name"])).convert("RGB")
            b, s, c, p = sess.run(None, {"images": seg_input(img, size).transpose(2, 0, 1)[None]})
            o = {"boxes": b[0], "scores": s[0], "coeffs": c[0]}
            proto = p[0].transpose(1, 2, 0)
        for c, s, box, m in decode(o["boxes"], o["scores"], o["coeffs"], proto, size, W, H):
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8))); rle["counts"] = rle["counts"].decode()
            dets.append({"image_id": i, "category_id": cat_ids[c], "score": s, "bbox": [float(v) for v in box], "segmentation": rle})
    res = {"variant": a.variant, "size": size, "mode": "device" if a.device_out else "float", "n": len(ids)}
    for t in ("bbox", "segm"):
        E = COCOeval(gt, gt.loadRes(dets), t); E.params.imgIds = ids
        E.evaluate(); E.accumulate(); E.summarize()
        res[f"{t}_mAP50_95"] = round(float(E.stats[0]), 4); res[f"{t}_mAP50"] = round(float(E.stats[1]), 4)
    print(json.dumps(res))
    with open(os.path.join(OUT, "results.jsonl"), "a") as f:
        f.write(json.dumps({"task": "seg", **res}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nyu-prep", "depth-inputs", "depth-score", "coco-inputs", "seg-score"])
    ap.add_argument("--variant", choices=["fps", "balance", "quality"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--device_out")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"nyu-prep": nyu_prep, "depth-inputs": depth_inputs, "depth-score": depth_score,
     "coco-inputs": coco_inputs, "seg-score": seg_score}[a.cmd](a)
