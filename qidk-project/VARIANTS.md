# FPS / Balance / Quality apps

Three installable variants of the depth + segmentation + 6-DoF pose pipeline,
each with its own package, name and launcher logo, so they sit side by side on
the QIDK SM8650. `android_app/` is untouched; `test_multimodel/` is the
optimization testbed the three were derived from.

| App | Folder | Package | Logo | Depth (DA-V2 Small) | Seg (YOLOv8n-seg) | Camera |
|---|---|---|---|---|---|---|
| **QIDK FPS** | `app_fps/` | `com.example.qidk.fps` | ⚡ orange | 280², W8A8 | 384², W8A8 | 640×480 @ **60** |
| **QIDK Balance** | `app_balance/` | `com.example.qidk.balance` | ⚖ teal | 392², W8A8 | 640², W8A8 | 640×480 @ 30 |
| **QIDK Quality** | `app_quality/` | `com.example.qidk.quality` | 💎 purple | 518², W8A8 | 640², **W8A16** | 1280×720 @ 30 |

W8A8 = 8-bit weights and activations; W8A16 = 8-bit weights, 16-bit activations.
All models use per-channel weight quantization and 32-bit bias.

## Measured on the board (All view, 30 s each)

| App | Displayed fps (avg / min) | Pipeline latency |
|---|---|---|
| FPS | 60.0 / 59.6 | ~8–11 ms |
| Balance | 30.0 / 29.4 | ~18–21 ms |
| Quality | 30.0 / 29.7 | ~28–31 ms (little headroom; busy scenes may dip slightly below 30) |

Detection fidelity vs. the float32 ONNX model (same preprocessing), `bus.jpg`:

| | person ×3 | bus | small person |
|---|---|---|---|
| float32 @640 | .88 .88 .84 | .72 | .47 |
| Quality (W8A16) | .88 .88 .83 | .59 | .43 |
| Balance (W8A8) | .88 .88 .83 | .54 | .37 |
| FPS (W8A8 @384; float32 @384: .88 .85 .82 / .85 / .42) | .89 .83 .81 | .81 | .38 |

## Code layout

The Kotlin sources are identical across the three apps **except
`AppConfig.kt`** (variant name, camera fps/resolution, seg thresholds). Model
input sizes, tensor layouts and quantization encodings are read from the DLCs
and their `.json` sidecars in `assets/`, so swapping models needs no code
change. Keep shared changes in sync across all three folders.

## Rebuilding the models

```bash
docker start hungry_lamport
docker exec hungry_lamport bash /work/variants/build_variants.sh
# -> model_conversion/variants/out/{fps,balance,quality}_{depth,seg}.{dlc,json}
```
Copy `<variant>_depth.*` / `<variant>_seg.*` into
`app_<variant>/app/src/main/assets/` as `depth.dlc|json` and `seg.dlc|json`.

The script calibrates on 37 images in the DLC's NHWC layout. (The original
`make_raw_inputs.py` wrote NCHW, so the older `depth_small_htp.dlc` /
`seg_decoupled_htp.dlc` were calibrated on scrambled images.) Seg needs
`--set_output_tensors boxes,scores,coeffs,proto` at graph-prepare, or the HTP
cache only exposes `proto`.

## Build, install, verify

```bash
cd app_balance && ./gradlew assembleDebug && adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s ProfilePipeline        # 1/s: displayed fps, pipeline ms, NPU ms, objects

# Still-image QA through the exact pipeline (no camera):
adb push bus.jpg /sdcard/Android/data/com.example.qidk.balance/files/
adb shell am start -n com.example.qidk.balance/com.example.multipipeline.MainActivity --es test_image bus.jpg
adb logcat -s StillImageTest
```

`app/libs/snpe-release.aar` and `app/src/main/jniLibs/` are git-ignored, as in
the other projects; copy them from `test_multimodel/` if missing.

Notes: the camera has a stall watchdog (rebinds after 3 s without frames).
Android will not open the camera while the board is dozing or locked, so wake
and unlock it before launching.
