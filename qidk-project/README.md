# On-Device 3-Model Pipeline (Depth + Segmentation + Pose) for QIDK SM8650

This package gives you everything to go from "three trained models" to a running
Android app on your **QIDK (Qualcomm Innovators Development Kit) SM8650 / Snapdragon 8 Gen 3**
board, Android 14, with all three models executing on the **NPU (Hexagon HTP)** via
**Qualcomm QNN / SNPE**, fed by the **live camera**.

Models used:
1. **Depth** — Depth Anything V2 (Small) — monocular depth
2. **Segmentation** — YOLOv8n-seg — instance segmentation
3. **Pose** — YOLOv8n-pose — keypoint/pose estimation

(I picked the YOLOv8 family for seg + pose since you're already using YOLOv8 for
segmentation — it shares one export toolchain (Ultralytics) with pose, which
keeps the conversion path identical for both and is well-proven on Hexagon NPUs.
If your "pinhole" pose model is actually something specific, swap step 2 in
`model_conversion/` for its export code — everything downstream is unaffected.)

## The 3 phases

### Phase 1 — Export each model to ONNX (`model_conversion/`)
Run on your normal training machine (PyTorch installed). Produces 3 `.onnx` files.

### Phase 2 — Convert ONNX → QNN `.dlc` + quantize (`model_conversion/convert_to_dlc.sh`)
Run on a **Linux x86_64 machine** (Ubuntu 20.04/22.04) with the **Qualcomm QNN SDK**
installed. This is a separate offline toolchain — it does NOT run inside Android
Studio. Produces 3 `.dlc` files, ready for the Hexagon NPU.

### Phase 3 — Android app (`android_app/`)
An Android Studio project (Kotlin) that:
- Captures live camera frames with CameraX
- Runs the 3 `.dlc` models on the NPU via the SNPE/QNN Java API
- Pipelines them (shared frame → depth, seg, pose all run per-frame)
- Composites depth-colormap + segmentation mask + pose skeleton into one overlay
  drawn live on top of the camera preview

Full setup instructions are in `model_conversion/README.md` (phases 1-2) and
`android_app/README.md` (phase 3, plus QNN SDK install instructions).

## Order of operations
1. Read `model_conversion/README.md`, install the QNN SDK, run the 3 export
   scripts, then `convert_to_dlc.sh`.
2. Copy the resulting `depth.dlc`, `seg.dlc`, `pose.dlc` into
   `android_app/app/src/main/assets/`.
3. Open `android_app/` in Android Studio, follow `android_app/README.md` to add
   the SNPE AAR dependency, build, and install directly onto the QIDK board over
   `adb` (USB debugging, Android 14 — standard `Run ▶` in Android Studio works
   once the board shows up under `adb devices`).
