# RS-1 (research 1)

The research runtime for `QIDK_CoDesign_Proposal.pdf` (physical-error-budgeted
HW/SW co-design for aerial grasping). Package `com.example.qidk.rs1`; installs
next to the FPS / Balance / Quality apps without touching them.

Unlike those apps, the models and processors are not fixed: they are a runtime
**configuration** (`research/Configuration.kt`) chosen per network:

| Knob | Values |
|---|---|
| Model variant (size + resolution + precision) | S = fps (280²/384² W8A8), M = balance (392²/640² W8A8), L = quality (518²/640², seg W8A16) |
| Processor per network | NPU (HTP, quantized), GPU (fp16), CPU (fp32) |
| Placement | parallel (two threads) or sequential |
| Power mode | SNPE perf profile: burst / high / balanced / saver |

Every model used stays loaded (`research/ModelBank.kt`), so switching costs
nothing after the first use. Labels read `D:L@N S:M@G par burst` = depth
quality-variant on NPU, seg balance-variant on GPU, in parallel.

## Live

The **Cfg ▸** button cycles `PipelineConfig.PRESETS`. The HUD shows per-stage
latency, the first-use load time, and CPU/GPU/NPU temperatures.
`--ei config <i>` starts on preset *i*.

## Bench (the step-2 profile table)

```bash
adb shell mkdir -p /sdcard/Android/data/com.example.qidk.rs1/files/bench
adb push img1.jpg img2.jpg ... /sdcard/Android/data/com.example.qidk.rs1/files/bench/
adb shell am start -n com.example.qidk.rs1/com.example.rs1.MainActivity \
    --es bench bench --es sweep default --ei iters 40      # sweep: default | quick | power
adb logcat -s RS1Bench
adb pull /sdcard/Android/data/com.example.qidk.rs1/files/results/
```

Writes `<stamp>_summary.csv` (one row per configuration: p50/p90/p99 latency,
per-stage exec time, load time, depth RMSE/AbsRel, mask IoU, temperatures) and
`<stamp>_frames.csv` (every measured frame). Error is measured against the
reference configuration `D:L@G S:L@G` (largest models, fp16 GPU), the closest
to the float models available on the board, until ground-truth depth is recorded.

## Assets

`app/src/main/assets/{fps,balance,quality}_{depth,seg}.{dlc,json}` are copies of
the three variant apps' models (rebuild: see `../VARIANTS.md`).
`app/libs/snpe-release.aar` and `app/src/main/jniLibs/` are git-ignored; copy
from `../app_balance/` if missing.
