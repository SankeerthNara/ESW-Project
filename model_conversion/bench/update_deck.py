import copy, json
from pptx import Presentation

p = Presentation("/work/bench/deck_in.pptx")
S = lambda n: p.slides[n - 1]
res = {}
for l in open("/work/bench/out/results.jsonl"):
    r = json.loads(l); res[(r["task"], r["variant"])] = r

def set_cell(cell, text):
    tf = cell.text_frame; para = tf.paragraphs[0]
    runs = para.runs
    runs[0].text = text
    for r in runs[1:]: r._r.getparent().remove(r._r)
    for extra in tf.paragraphs[1:]: extra._p.getparent().remove(extra._p)

def fill_table(tbl, rows):
    for i, row in enumerate(rows):
        for j, v in enumerate(row): set_cell(tbl.cell(i, j), v)

def set_card(shape, heading, bullets):
    tf = shape.text_frame; ps = tf.paragraphs
    h, tmpl = ps[0], ps[1]
    h.runs[0].text = heading
    for r in h.runs[1:]: r._r.getparent().remove(r._r)
    tmpl_xml = copy.deepcopy(tmpl._p)
    for q in ps[1:]: q._p.getparent().remove(q._p)
    for label, body in bullets:
        new = copy.deepcopy(tmpl_xml); tf._txBody.append(new)
        para = tf.paragraphs[-1]; runs = para.runs
        if len(runs) >= 2:
            runs[0].text = f"• {label}: "; runs[1].text = body
            for r in runs[2:]: r._r.getparent().remove(r._r)
        else:
            runs[0].text = f"• {label}: {body}"

def clone_shapes(src, dst, kinds):
    for sh in src.shapes:
        if kinds(sh):
            dst.shapes._spTree.append(copy.deepcopy(sh._element))

def depth_acc(v):
    r = res.get(("depth", v))
    return (f"{r['AbsRel']:.3f}", f"{r['RMSE_m']:.2f} m") if r else ("—", "—")

# ---------------- slide 9: depth results (DA-V2 Small, measured on QIDK) ----
s9 = S(9)
tbl9 = [sh for sh in s9.shapes if sh.has_table][0].table
fill_table(tbl9, [
    ["Model (input, precision)", "Params", "QIDK NPU Latency", "NPU Throughput", "AbsRel*", "RMSE*", "Selection Status"],
    ["DA-V2 Small @280, INT8", "24.2M", "3.3 ms", "309 inf/s", *depth_acc("fps"), "FPS Mode (60 fps)"],
    ["DA-V2 Small @392, INT8", "24.2M", "6.0 ms", "168 inf/s", *depth_acc("balance"), "SELECTED (Balance)"],
    ["DA-V2 Small @518, INT8", "24.2M", "12.9 ms", "74 inf/s", *depth_acc("quality"), "Quality Mode"],
])
cards9 = [sh for sh in s9.shapes if sh.has_text_frame and sh.text_frame.text.startswith(("PRIMARY", "ACCURACY"))]
set_card(cards9[0], "PRIMARY SELECTION: Depth Anything V2 Small", [
    ("Why it replaced FastDepth/MiDaS", "ViT-S encoder gives sharper object boundaries and robust zero-shot depth; 24.2M params fits the Hexagon HTP after INT8 quantization."),
    ("Measured on QIDK SM8650", "6.0 ms pure NPU time at 392² (snpe-net-run, 100 runs, burst), 14x smaller DLC than the ViT-L model we started with (682 MB → 50 MB)."),
])
set_card(cards9[1], "ACCURACY / SPEED TRADE-OFF STRATEGY", [
    ("Three resolutions, one model", "280² / 392² / 518² trade 3.3 → 12.9 ms NPU time for finer depth detail; chosen per app variant without code changes."),
    ("*Accuracy protocol", "NYU Depth v2 test (654 imgs), float reference, scale-shift aligned in disparity (relative-depth model), Eigen crop. '—' = evaluation in progress."),
])

# ---------------- slide 12: segmentation results ---------------------------
s12 = S(12)
tbl12 = [sh for sh in s12.shapes if sh.has_table][0].table
fill_table(tbl12, [
    ["Model (input)", "Precision", "Params", "QIDK NPU Latency", "NPU Throughput", "Mask mAP*", "Selection Status"],
    ["YOLOv8n-Seg @384", "W8A8", "3.4M", "0.5 ms", "1599 inf/s", "—", "FPS Mode (60 fps)"],
    ["YOLOv8n-Seg @640", "W8A8", "3.4M", "1.1 ms", "643 inf/s", "30.5%", "SELECTED (Balance)"],
    ["YOLOv8n-Seg @640", "W8A16", "3.4M", "2.8 ms", "288 inf/s", "30.5%", "Quality Mode"],
])
cards12 = [sh for sh in s12.shapes if sh.has_text_frame and sh.text_frame.text.startswith(("PRIMARY", "PIPELINE"))]
set_card(cards12[0], "PRIMARY SELECTION: YOLOv8n-Seg (decoupled head)", [
    ("Measured on QIDK SM8650", "1.1 ms pure NPU time at 640² INT8; boxes, scores, mask coefficients and prototypes exported as separate outputs so each gets its own quantization scale."),
    ("Replaced YOLOv8x-Seg", "143 MB YOLOv8x-Seg caused DSP out-of-memory when running alongside depth; YOLOv8n-Seg DLC is 8 MB."),
])
set_card(cards12[1], "PIPELINE FUSION STRATEGY", [
    ("Mask-Gated Depth", "Instance masks select the target's depth pixels; background and floor are discarded before 3D back-projection."),
    ("*mAP", "COCO val2017 mask mAP50-95 of the float model at 640 (Ultralytics reference); INT8 on-device evaluation in progress."),
])

# ---------------- slides 15 / 16: pipeline integration ----------------------
for n in (15, 16):
    clone_shapes(s9, S(n), lambda sh: sh.has_table or (sh.has_text_frame and sh.text_frame.text.startswith(("PRIMARY", "ACCURACY"))))

s15 = S(15)
fill_table([sh for sh in s15.shapes if sh.has_table][0].table, [
    ["Stage", "Model", "Precision / I/O", "Input", "NPU Time (in app)", "Stage Total", "Observation"],
    ["Depth", "DA-V2 Small", "Float, FloatTensor", "392²", "28 ms", "31 ms", "Float math on HTP"],
    ["Segmentation", "YOLOv8n-Seg", "Float, FloatTensor", "640²", "110 ms", "117 ms", "Main bottleneck"],
    ["Pose + Pipeline", "Geometric PCA", "CPU", "masks + depth", "<1 ms", "147 ms", "6.6 FPS end-to-end"],
])
c15 = [sh for sh in s15.shapes if sh.has_text_frame and sh.text_frame.text.startswith(("PRIMARY", "ACCURACY"))]
set_card(c15[0], "BASELINE RESULT: 6.6 FPS (147 ms / frame)", [
    ("Measured on QIDK SM8650", "30 s live-camera run, 124 frames, pipeline 141–154 ms per frame; far below the 30 FPS control-loop target."),
    ("First working version", "Depth Anything V2 Small + YOLOv8n-Seg in float, depth and seg dispatched in parallel, PCA-based 6-DoF pose on CPU (<1 ms)."),
    ("Before that", "ViT-L depth (682 MB DLC) + YOLOv8x-Seg (143 MB) ran out of DSP memory when run together."),
])
set_card(c15[1], "BOTTLENECKS IDENTIFIED", [
    ("Float tensors over JNI", "FloatTensor API costs ~15 ms per model per frame in conversion alone."),
    ("Seg decoding bug", "CHW data fed to an NHWC model: only 336 of 1344 anchors decoded correctly."),
    ("UI-thread rendering", "Per-pixel mask drawing and bitmap building on the UI thread; camera stalls with no recovery."),
])

s16 = S(16)
fill_table([sh for sh in s16.shapes if sh.has_table][0].table, [
    ["App Variant", "Depth (DA-V2 S)", "Seg (YOLOv8n)", "Camera", "Displayed FPS", "Pipeline Latency", "NPU depth / seg"],
    ["QIDK FPS", "280², W8A8", "384², W8A8", "640×480 @60", "60.0 (min 59.8)", "7.8 ms", "3.3 / 0.5 ms"],
    ["QIDK Balance", "392², W8A8", "640², W8A8", "640×480 @30", "30.0 (min 29.3)", "20.1 ms", "6.0 / 1.1 ms"],
    ["QIDK Quality", "518², W8A8", "640², W8A16", "1280×720 @30", "30.0 (min 29.7)", "29.2 ms", "12.9 / 2.8 ms"],
])
c16 = [sh for sh in s16.shapes if sh.has_text_frame and sh.text_frame.text.startswith(("PRIMARY", "ACCURACY"))]
set_card(c16[0], "OPTIMISATIONS APPLIED", [
    ("Quantization", "INT8 per-channel weights, 32-bit bias, NHWC-correct calibration; W8A16 for Quality seg."),
    ("Zero-conversion I/O", "SNPE TF8/TF16 user buffers in each tensor's native encoding instead of FloatTensor."),
    ("Concurrency", "Depth and seg run in parallel on the HTP; overlays built off the UI thread with triple buffering; camera stall watchdog."),
])
set_card(c16[1], "RESULT vs BASELINE (measured on QIDK)", [
    ("Latency", "147 ms → 20.1 ms (Balance, 7.3x) and 7.8 ms (FPS, 19x)."),
    ("Frame rate", "6.6 → 60 FPS (FPS mode); Balance and Quality hold the 30 FPS camera rate with headroom."),
    ("Method", "30 s live runs per app; NPU times from snpe-net-run (100 inferences, burst profile)."),
])

# ---------------- slide 18: milestones --------------------------------------
def replace_bullet(shape, label_prefix, label, body):
    for para in shape.text_frame.paragraphs:
        if para.text.lstrip("• ").startswith(label_prefix):
            runs = para.runs
            if len(runs) >= 2:
                runs[0].text = f"• {label}: "; runs[1].text = body
                for r in runs[2:]: r._r.getparent().remove(r._r)
            else: runs[0].text = f"• {label}: {body}"
for sh in S(18).shapes:
    if not sh.has_text_frame: continue
    replace_bullet(sh, "Optimal Selection", "Deployed Models", "Depth Anything V2 Small (3.3–12.9 ms NPU), YOLOv8n-Seg (0.5–2.8 ms NPU), geometric PCA 6-DoF pose on CPU.")
    replace_bullet(sh, "Pipeline Feasibility", "Pipeline Measured", "60 FPS (FPS mode) and 30 FPS (Balance / Quality) on the QIDK, up from 6.6 FPS baseline.")
    replace_bullet(sh, "Hexagon NPU Compilation", "Hexagon NPU Compilation (done)", "ONNX → DLC, INT8/W8A16 quantization and offline HTP cache for SM8650.")
    for para in sh.text_frame.paragraphs:
        for r in para.runs:
            if r.text.strip() == "ACTIVE IMPLEMENTATION": r.text = "COMPLETED"

for sh in S(16).shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Pipeline Integration: Currently"):
        r = sh.text_frame.paragraphs[0].runs; r[0].text = "Pipeline Integration: Optimised Version"
        for x in r[1:]: x._r.getparent().remove(x._r)
p.save("/work/bench/deck_out.pptx"); print("saved")
