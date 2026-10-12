# test_multimodel — 30 fps depth + segmentation + 6-DoF pose on QIDK SM8650

Live camera → Depth Anything V2 Small + YOLOv8n-seg on the Hexagon NPU (HTP),
in parallel → geometric pose (pinhole back-projection + PCA) → overlay.
Runs at the camera's locked 30 fps with ~20 ms end-to-end pipeline latency.

## Models (`app/src/main/assets/`)

| Asset       | Source DLC                                       | Input            | NPU time* |
|-------------|--------------------------------------------------|------------------|-----------|
| `depth.dlc` | `model_conversion/depth_small_htp.dlc`           | 392×392 INT8     | ~6 ms     |
| `seg.dlc`   | `model_conversion/build/seg/seg_decoupled_htp.dlc` | 640×640 INT8   | ~1.6 ms   |

\* `snpe-throughput-net-run --use_dsp --perf_profile burst` on the board.

Both are INT8 with an offline SM8650 HTP cache (`snpe-dlc-graph-prepare --htp_socs sm8650`).
The seg model's detect head is exported decoupled (`boxes`, `scores`, `coeffs`,
`proto` as separate outputs) so each gets its own INT8 scale.

**If you re-export or re-quantize a model**, update the `QuantEncoding`
constants in `DepthStage.kt` / `SegmentationStage.kt` from
`snpe-dlc-info -i <model>.dlc` (input/output "Encoding Info"). Wrong values
still run (SNPE requantizes) but cost speed and, for outputs, accuracy.

## Where the time went, and what changed

The NPU was never the bottleneck. Measured on the board, before → after:

| Cause | Fix |
|---|---|
| SNPE `FloatTensor` API: every input float quantized and every output dequantized on a JNI thread — ~15 ms per model per frame | TF8 user buffers in the DLC's native uint8 encodings (`SnpeModelRunner`); preprocessing writes quantized bytes through per-channel LUTs (`ModelInputPacker`) |
| Depth and seg ran serially despite the "parallel" comment | One thread per model in `PipelineOrchestrator` |
| Camera AE free to drop to 10–15 fps indoors | `CONTROL_AE_TARGET_FPS_RANGE` locked to [30,30] |
| `ImageProxy.toBitmap()` + per-frame `Bitmap` rotation; hidden `Preview` use case | RGBA output with CameraX native rotation, copied into a reused bitmap; ImageAnalysis only |
| Masks drawn as one `drawRect` per mask pixel on the UI thread | All masks composited into one 160×160 bitmap off the UI thread, drawn scaled |
| New bitmaps / arrays every frame | Triple-buffered `RenderFrame` pool |

Quality fixes along the way: the previous seg setup fed CHW data to an NHWC
model and decoded 336 of 1344 anchors, so its detections were noise; seg now
runs at full 640 resolution with the stock 0.25 confidence threshold, and the
depth view uses a proper turbo colormap.

## Build & install

1. `app/libs/snpe-release.aar` from `$QNN_SDK_ROOT/lib/android/`.
2. `app/src/main/jniLibs/arm64-v8a/` holds the QAIRT HTP libs (see repo root README).
3. Copy the two DLCs above into `app/src/main/assets/` as `depth.dlc` / `seg.dlc`.
4. `./gradlew assembleDebug && adb install -r app/build/outputs/apk/debug/app-debug.apk`

Verify:
```bash
adb logcat -s ProfilePipeline   # once per second: displayed fps, pipeline ms, NPU ms, object count
adb logcat -s SnpeModelRunner   # runtime=DSP and tensor shapes per model
```
