# Real-Time On-Device Multi-Model Vision Pipeline for QIDK SM8650 (Snapdragon 8 Gen 3)

An end-to-end edge AI computer vision pipeline running monocular depth estimation, instance segmentation, and geometric 6-DoF pose estimation concurrently on the **Qualcomm Innovators Development Kit (QIDK) SM8650 (Snapdragon 8 Gen 3)** board under **Android 14 (API 34)**, accelerated on the **Qualcomm Hexagon NPU / HTP** via the **Qualcomm AI Runtime (QAIRT / QNN / SNPE)**.

---

## Architecture Overview

```
                        Live Camera Frame (CameraX)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
             DepthStage                       SegmentationStage
    (Depth Anything V2 Small)                   (YOLOv8n-seg)
        Hexagon NPU / HTP                     Hexagon NPU / HTP
          [1, 3, 392, 392]                      [1, 3, 640, 640]
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                              PoseEstimator
                     (Geometric 6-DoF Pinhole + PCA)
                                    │
                                    ▼
                               OverlayView
         ┌───────────────┬─────────────────┬───────────────┐
         │ Depth (Turbo) │  Segmentation   │ Pose (6-DoF)  │
         ├───────────────┼─────────────────┼───────────────┤
         │ Live Pipeline │ Quad Split (All)│  Live Camera  │
         └───────────────┴─────────────────┴───────────────┘
```

1. **Depth Estimation**: Depth Anything V2 Small (`vits`, 392×392 static input, ImageNet-normalized) running on Hexagon HTP.
2. **Instance Segmentation**: YOLOv8n-seg (640×640 static input, 80 COCO classes) running on Hexagon HTP.
3. **6-DoF Pose Estimation**: Analytical geometric back-projection from 2D mask pixels into 3D pseudo-metric space (0.3m–2.5m) using pinhole camera intrinsics and the depth map, solved via Principal Component Analysis (cyclic Jacobi symmetric 3×3 eigenvalue decomposition) to extract 3D centroid and orientation (roll, pitch, yaw) axes.
4. **Display Modes**:
   - `Depth`: Dense color-mapped depth visualization (Turbo colormap).
   - `Seg`: Live camera preview with overlaid semi-transparent instance masks, bounding boxes, and class labels.
   - `Pose`: Live camera preview with overlaid 3D Cartesian orientation axes (X=Red, Y=Green, Z=Orange) and metric coordinates.
   - `Pipeline`: Full combined composite (live camera + masks + 3D coordinate axes).
   - `All`: Synchronized 2×2 quad-split dashboard displaying all outputs concurrently.

---

## What We Did & Key Fixes

- **Re-exported Depth Model to Depth Anything V2 Small (`vits`)**:
  - Replaced the heavy ViT-Large variant (682 MB DLC with HTP cache, 335M params) with **Depth Anything V2 Small** (24.8M params).
  - Exported static ONNX at 392×392 (28×14 patch multiple), converted to float DLC, quantized to INT8 with ImageNet calibration, and compiled the SM8650 Hexagon HTP offline cache.
  - **Result**: DLC size decreased from **682 MB → 50 MB** (**14x reduction**), drastically cutting latency and memory bandwidth.
- **Switched to Lightweight YOLOv8n-seg**:
  - Replaced the heavy 143 MB YOLOv8x-seg model with **YOLOv8n-seg** (7.4 MB with SM8650 HTP cache), avoiding DSP out-of-memory errors during concurrent multi-thread inference.
- **Fixed Black Screen on Pose and Segmentation**:
  - In `OverlayView.kt`, the Seg and Pose modes were previously rendering on solid black backgrounds (`blackBg = true`), which completely hid the camera feed and produced pitch-black screens when no objects were detected.
  - Updated `OverlayView.kt` to draw the live camera preview as the base under Seg and Pose, adding clear on-screen status labels (`"Seg: 0 objects detected"`, `"Pose: 0 objects detected"`).
- **Fixed Internal Storage Asset Staling**:
  - In `SnpeModelRunner.kt`, cached models in `filesDir` were never re-extracted when the DLC in `assets/` changed. Added file length validation against the APK asset so new models are immediately extracted and used upon installation.
- **Optimized Detection Thresholds**:
  - Lowered `confThreshold` in `SegmentationStage.kt` to `0.25f` and `MIN_MASK_PIXELS` in `PoseEstimator.kt` to `15` to ensure consistent real-time detection on INT8 quantized tensors.

---

## Repository Structure

```
├── requirements.txt               # Python dependencies for model export & conversion
├── .gitignore                     # Git ignore rules for checkpoints, DLCs, and SDKs
├── README.md                      # This reproduction guide
│
├── model_conversion/              # Model export & offline conversion environment
│   ├── export_seg.py              # Export YOLOv8 segmentation model to ONNX
│   ├── export_pose.py             # Export YOLOv8 pose model to ONNX
│   ├── make_raw_inputs.py         # Calibration preprocessor for QNN quantizer
│   ├── calibration/               # Sample calibration images (depth, seg, pose)
│   └── Depth-Anything-V2/         # Depth Anything V2 source & checkpoints
│       ├── export_depth.py        # Export DA-V2 Small (392x392) to ONNX
│       └── checkpoints/           # PyTorch model weights (.pth)
│
├── qidk-project/                  # Standalone QIDK deployment bundle
│   ├── model_conversion/          # Standalone conversion scripts & convert_to_dlc.sh
│   └── android_app/               # Android Studio project (Kotlin)
│       ├── build.gradle.kts       # App configuration (minSdk 31, targetSdk 34, arm64-v8a)
│       ├── app/libs/              # snpe-release.aar (Qualcomm SNPE Java API)
│       ├── app/src/main/
│       │   ├── AndroidManifest.xml# Permissions & FastRPC native library linkages
│       │   ├── assets/            # Converted DLC models (depth.dlc, seg.dlc)
│       │   ├── jniLibs/arm64-v8a/ # Qualcomm native binaries (libSNPE.so, libQnnHtp.so)
│       │   └── java/.../multipipeline/
│       │       ├── MainActivity.kt        # UI, permissions & FPS benchmark
│       │       ├── camera/CameraManager.kt# CameraX frame capture
│       │       ├── snpe/SnpeModelRunner.kt# Qualcomm SNPE NPU/HTP hardware runner
│       │       ├── pipeline/
│       │       │   ├── PipelineOrchestrator.kt # Concurrent NPU execution coordinator
│       │       │   ├── DepthStage.kt           # Depth Anything V2 pre/post-processing
│       │       │   ├── SegmentationStage.kt    # YOLOv8-seg pre/post-processing & NMS
│       │       │   └── PoseEstimator.kt        # 6-DoF Pinhole back-projection & PCA
│       │       └── overlay/OverlayView.kt      # Rendering engine (5 display modes)
│
└── qnn/                           # Qualcomm AI Runtime (QAIRT / QNN) SDK (v2.50.0)
```

---

## Step-by-Step Reproduction Guide

### 1. Prerequisites

- **Host Machine**: Linux x86_64 (Ubuntu 20.04/22.04 LTS or Docker container with Python 3.10).
- **Qualcomm QNN / QAIRT SDK**: Download Qualcomm AI Engine Direct SDK from [Qualcomm Package Manager (QPM)](https://qpm.qualcomm.com).
- **Target Device**: QIDK SM8650 (Snapdragon 8 Gen 3) board connected via USB with Android 14.
- **Android Studio**: Hedgehog / Iguana / Koala with Android SDK 34 and NDK.

Install Python host dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Export Models to ONNX

#### A. Depth Anything V2 Small
```bash
cd model_conversion/Depth-Anything-V2
python3 export_depth.py
# Produces depth.onnx with static shape [1, 3, 392, 392]
```

#### B. YOLOv8n Segmentation
```bash
cd model_conversion
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
model.export(format='onnx', imgsz=640, dynamic=False, simplify=True, opset=17)
"
# Produces yolov8n-seg.onnx with static shape [1, 3, 640, 640]
```

---

### 3. Convert ONNX to Quantized DLC for Hexagon NPU

Set up the Qualcomm SDK environment:
```bash
export QNN_SDK_ROOT=/path/to/qnn/qairt/2.50.0.260828
source $QNN_SDK_ROOT/bin/envsetup.sh
```

#### A. Depth Model Conversion & HTP Cache
```bash
# 1. Generate ImageNet-normalized calibration raw tensors
python3 model_conversion/make_raw_inputs.py \
  --src model_conversion/calibration/depth \
  --dst model_conversion/build/depth/raw_392 \
  --width 392 --height 392 \
  --normalize imagenet --input_name image \
  --list_out model_conversion/build/depth/input_list_392.txt

# 2. Convert ONNX to float DLC
snpe-onnx-to-dlc \
  --input_network model_conversion/Depth-Anything-V2/depth.onnx \
  -d image 1,3,392,392 \
  -o model_conversion/build/depth/depth_small_float.dlc

# 3. Quantize DLC to INT8
snpe-dlc-quantize \
  --input_dlc model_conversion/build/depth/depth_small_float.dlc \
  --input_list model_conversion/build/depth/input_list_392.txt \
  --output_dlc model_conversion/depth.dlc

# 4. Compile offline HTP context binary cache for SM8650
snpe-dlc-graph-prepare \
  --input_dlc model_conversion/depth.dlc \
  --output_dlc model_conversion/depth_htp.dlc \
  --htp_socs sm8650
```

#### B. Segmentation Model Conversion & HTP Cache
```bash
# 1. Generate unit-normalized (0-1) calibration raw tensors
python3 model_conversion/make_raw_inputs.py \
  --src model_conversion/calibration/seg \
  --dst model_conversion/build/seg/raw_640 \
  --width 640 --height 640 \
  --normalize unit --input_name images \
  --list_out model_conversion/build/seg/input_list_640.txt

# 2. Convert ONNX to float DLC
snpe-onnx-to-dlc \
  --input_network model_conversion/yolov8n-seg.onnx \
  -d images 1,3,640,640 \
  -o model_conversion/build/seg/yolov8n_seg_float.dlc

# 3. Quantize DLC to INT8
snpe-dlc-quantize \
  --input_dlc model_conversion/build/seg/yolov8n_seg_float.dlc \
  --input_list model_conversion/build/seg/input_list_640.txt \
  --output_dlc model_conversion/seg.dlc

# 4. Compile offline HTP context binary cache for SM8650
snpe-dlc-graph-prepare \
  --input_dlc model_conversion/seg.dlc \
  --output_dlc model_conversion/seg_htp.dlc \
  --htp_socs sm8650
```

---

### 4. Deploying the Android Application

1. **Copy Models into Assets**:
   ```bash
   cp model_conversion/depth_htp.dlc qidk-project/android_app/app/src/main/assets/depth.dlc
   cp model_conversion/seg_htp.dlc   qidk-project/android_app/app/src/main/assets/seg.dlc
   ```

2. **Add SNPE AAR**:
   Copy `snpe-release.aar` from `$QNN_SDK_ROOT/lib/android/` into `qidk-project/android_app/app/libs/snpe-release.aar`.

3. **Build the APK**:
   ```bash
   cd qidk-project/android_app
   ./gradlew assembleDebug
   ```

4. **Install & Run on QIDK SM8650 Board**:
   - Ensure Developer Options and USB Debugging are enabled on the board.
   - Verify connection:
     ```bash
     adb devices
     ```
   - Install APK:
     ```bash
     adb install -r app/build/outputs/apk/debug/app-debug.apk
     ```
   - Or click **Run ▶** directly in Android Studio.

---

### 5. On-Device Verification

To verify that the models execute on the **Hexagon NPU / HTP** rather than falling back to CPU:
```bash
adb logcat | grep -iE "snpe|SnpeModelRunner|htp"
```
Look for log confirmation:
```
[SnpeModelRunner] Runtime Support -> DSP/HTP (unsigned PD check): true
[SnpeModelRunner] Built with runtime order: DSP (unsigned PD) -> CPU
```

Use the bottom navigation bar in the app to switch visualization modes:
- **Depth**: Monocular depth heatmap.
- **Seg**: Live camera feed with instance masks and class boxes.
- **Pose**: Live camera feed with 3D orientation axes and metric distance readouts.
- **Pipeline**: Live camera feed with combined segmentation and 3D pose overlays.
- **All**: Synchronized 4-quadrant display.
