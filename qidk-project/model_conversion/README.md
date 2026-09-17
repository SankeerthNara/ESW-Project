# Phase 1 & 2 — Export + Convert Models to .dlc

## 0. Install the Qualcomm QNN SDK (one-time, Linux x86_64 host)

You need a Linux machine (Ubuntu 20.04 or 22.04 — a VM is fine) for this. The
QNN toolchain does not run on Windows or macOS natively.

1. Create a free account at https://qpm.qualcomm.com (Qualcomm Package Manager) —
   this is the same login as the Qualcomm Developer Network you used for the QIDK.
2. Go to **Qualcomm AI Engine Direct SDK (QNN SDK)** in QPM and download the
   latest version (this is the successor/rebrand of SNPE; it still ships the
   SNPE Java API used in Phase 3).
3. Extract it, e.g. to `~/qnn/qairt/<version>/`. Set env vars (add to `~/.bashrc`):
   ```bash
   export QNN_SDK_ROOT=~/qnn/qairt/<version>
   export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
   ```
4. Install Python deps for the converters (SDK ships a setup script):
   ```bash
   cd $QNN_SDK_ROOT/bin
   ./check-python-dependency          # tells you what's missing
   pip install onnx==1.14.1 onnxruntime numpy pillow torch torchvision
   ```
5. Verify: `qnn-onnx-converter --version` should print a version, not "command not found".

## 1. Export each model to ONNX

Run these on your normal PyTorch box (GPU optional, CPU is fine for export).

```bash
pip install ultralytics onnx onnxsim torch torchvision

python 1_export_depth_anything_v2.py   # -> depth.onnx
python 2_export_yolov8_seg.py          # -> seg.onnx   (swap in your own weights if you already trained a custom YOLOv8-seg)
python 3_export_yolov8_pose.py         # -> pose.onnx  (swap for your real "pinhole" model's export code if it's not YOLOv8-pose)
```

Each script fixes the input to a **static shape** (e.g. 384x384 or 640x640).
QNN's converter strongly prefers static shapes — dynamic axes will fail or
silently fall back to CPU. Keep the resolutions noted in each script; they're
copied into the Android pre-processing code in Phase 3, so if you change them,
update `android_app/.../pipeline/*Stage.kt` to match.

## 2. Convert ONNX → DLC (quantized, HTP/NPU-ready)

```bash
chmod +x convert_to_dlc.sh
./convert_to_dlc.sh depth.onnx depth 384 384
./convert_to_dlc.sh seg.onnx   seg   640 640
./convert_to_dlc.sh pose.onnx  pose  640 640
```

This does two things per model:
1. `qnn-onnx-converter` → floating-point `.dlc`
2. `qnn-model-lib-generator` / `snpe-dlc-quantize` → **INT8-quantized `.dlc`**
   using a handful of representative camera-like images, so it can run on the
   Hexagon Tensor Processor (HTP/NPU) instead of falling back to CPU.

You need ~20-100 representative input images per model in
`model_conversion/calibration/<name>/` before running the script — the script
will tell you if the folder is empty. Just grab a handful of frames similar to
what your camera will actually see (people, indoor/outdoor scenes, etc.),
resized to match each model's input size.

Output: `depth.dlc`, `seg.dlc`, `pose.dlc` — copy these three files into
`android_app/app/src/main/assets/`.

## Sanity-check on the board before writing app code (optional but recommended)

The QNN SDK ships `qnn-net-run` — you can push a `.dlc` + a raw input tensor to
the board over adb and run it standalone to confirm it executes on HTP and get
a rough latency number, before you've written a line of Kotlin:

```bash
adb push depth.dlc /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. qnn-net-run --backend libQnnHtp.so --model depth.dlc --input_list input_list.txt"
```

See `$QNN_SDK_ROOT/docs` for the exact `qnn-net-run` invocation for your SDK
version — flags shift slightly between releases.
