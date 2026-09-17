# Phase 3 — Android App Setup & Deployment to QIDK SM8650

## 1. Open the project
Open the `android_app/` folder directly in Android Studio (File → Open).
Let it sync Gradle once (it'll complain about the missing SNPE AAR — that's
expected, fix it in step 2).

## 2. Add the SNPE/QNN Java bindings (AAR)
The Java/Kotlin API for running `.dlc` models on the Hexagon NPU is **not on
Maven** — it ships inside the QNN SDK you installed in Phase 2.

1. On the machine where you installed the QNN SDK, find the AAR. Depending on
   SDK version it's one of:
   - `$QNN_SDK_ROOT/lib/android/snpe-release.aar`
   - `$QNN_SDK_ROOT/java/android/snpe-release.aar`
   - Search for it: `find $QNN_SDK_ROOT -iname "*.aar"`
2. Copy that `.aar` file into `android_app/app/libs/` (create the `libs`
   folder if it doesn't exist).
3. In `app/build.gradle.kts`, uncomment this line:
   ```kotlin
   implementation(files("libs/snpe-release.aar"))
   ```
4. In `app/src/main/java/com/example/multipipeline/snpe/SnpeModelRunner.kt`,
   uncomment the real implementation block (marked clearly in the file) and
   delete/comment out the stub `run()`/`close()` below it.
5. The AAR bundles the native `.so` files it needs (`libSNPE.so`,
   `libQnnHtp.so`, etc.) for `arm64-v8a` — Gradle will package them
   automatically once the AAR dependency is active. You should NOT need to
   manually copy `.so` files into `jniLibs/` unless your specific SDK version's
   AAR doesn't include them (rare) — if you get `UnsatisfiedLinkError` at
   runtime, that's the sign to manually copy the missing `.so` from
   `$QNN_SDK_ROOT/lib/aarch64-android/` into `app/src/main/jniLibs/arm64-v8a/`.

## 3. Add the converted models
Copy `depth.dlc`, `seg.dlc`, `pose.dlc` (from Phase 2) into
`app/src/main/assets/`.

## 4. Connect the QIDK board and deploy

1. On the QIDK SM8650 board: **Settings → About → tap Build Number 7x** to
   enable Developer Options, then **Settings → Developer Options → USB
   debugging → ON**.
2. Connect the board to your dev machine via USB.
3. Confirm it's visible:
   ```bash
   adb devices
   # should list the board's serial, e.g.:  1234567890ABCDEF   device
   ```
   If it shows "unauthorized", accept the RSA key prompt on the board's screen.
4. In Android Studio, the board should now appear in the device dropdown next
   to the Run button. Select it and hit **Run ▶** — this builds, installs,
   and launches directly on the board.
5. Grant the camera permission when prompted on-device.

You should see the live camera preview with:
- a translucent depth heat-map overlay,
- colored segmentation masks + boxes,
- a cyan pose skeleton with green keypoint dots,
- an FPS/latency readout in the top-left corner (this is your true end-to-end
  pipeline latency per frame — depth + seg + pose combined, since they run
  concurrently).

## 5. If something doesn't run on the NPU
`SnpeModelRunner` requests runtime order `DSP → GPU → CPU`. If the `.dlc`
wasn't quantized correctly in Phase 2, SNPE will silently fall back to CPU —
your FPS counter will make this obvious (CPU fallback for all 3 models on a
live camera feed will typically run at low single-digit FPS or worse).
To confirm which backend actually executed:
```bash
adb logcat | grep -i snpe
```
SNPE logs which runtime it selected for each network at load time.

## 6. Known simplifications to be aware of
- `SegmentationStage` and `PoseStage` decode Ultralytics' YOLOv8 ONNX export
  layout. If Netron shows different output tensor names/shapes for your
  actual exported `seg.onnx`/`pose.onnx` (this happens across Ultralytics
  versions), update `OUTPUT0_NAME`/`OUTPUT1_NAME`/`numAnchors` accordingly.
- Mask rendering in `OverlayView.drawMask()` draws one rect per mask pixel —
  simple and correct, but not the fastest possible approach. If it's a
  bottleneck at your target resolution, replace it with a proper `Bitmap`
  composite the same way `buildDepthBitmap()` already does for depth.
- The depth colormap in `OverlayView.turbo()` is a rough blue→red approximation,
  not the real matplotlib "turbo" LUT — swap in a proper 256-entry LUT if you
  want closer-to-standard depth visualization.
