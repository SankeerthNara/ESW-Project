package com.example.rs1.research

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.util.Log
import com.example.rs1.pipeline.PipelineOrchestrator
import com.example.rs1.pipeline.RenderFrame
import com.example.rs1.pipeline.SegInstance
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Offline profiling sweep: the measured (error, latency) table that step 2
 * of the proposal chooses from, and the data for motivation observations
 * 3 (no processor is best for both networks) and 4 (sequential placement
 * wastes processors).
 *
 * For every configuration it runs still images through the exact live
 * pipeline and records per-frame latency per stage, chip temperatures, and
 * output error against a reference configuration. Until ground-truth depth
 * is recorded, the reference is the largest models in fp16 on the GPU
 * ([PipelineConfig.REFERENCE]), so "error" here means "deviation from the
 * best this board can compute", not error against the world:
 *  - depth: relative disparity, aligned to the reference by least-squares
 *    scale + shift (standard for relative depth), then RMSE and AbsRel;
 *  - segmentation: mask IoU of each reference instance with the
 *    same-class prediction whose box overlaps it most (0 if missed).
 *
 * Writes `<stamp>_summary.csv` (one row per configuration) and
 * `<stamp>_frames.csv` (one row per measured frame) to [outDir].
 */
class Bench(
    private val orchestrator: PipelineOrchestrator,
    private val chip: ChipMonitor,
    private val outDir: File,
) {
    companion object {
        const val TAG = "RS1Bench"
        private const val GRID = 128          // common grid for depth comparison
        private const val MASK_GRID = 160     // common grid for mask comparison
        private const val ABSREL_MIN = 0.1f   // ignore far (near-zero disparity) pixels in AbsRel
    }

    data class Options(
        val configs: List<PipelineConfig>,
        val iters: Int = 40,
        val warmup: Int = 5,
        val cooldownS: Int = 0,
    )

    private class RefOut(val depth: FloatArray?, val instances: List<SegInstance>)

    fun run(images: List<File>, opt: Options) {
        val bitmaps = images.mapNotNull { f ->
            BitmapFactory.decodeFile(f.path, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 })
                ?.let { f.name to it }
                ?: null.also { Log.e(TAG, "cannot decode ${f.path}") }
        }
        require(bitmaps.isNotEmpty()) { "no decodable images" }
        val frame = RenderFrame()
        fun load(i: Int) {
            val img = bitmaps[i % bitmaps.size].second
            Canvas(frame.cameraBitmap(img.width, img.height)).drawBitmap(img, 0f, 0f, null)
        }

        outDir.mkdirs()
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val summaryFile = File(outDir, "${stamp}_summary.csv")
        val framesFile = File(outDir, "${stamp}_frames.csv")
        Log.i(TAG, "bench: ${opt.configs.size} configs x ${opt.iters} frames over ${bitmaps.size} images -> ${summaryFile.path}")

        // Reference outputs, one per image.
        var refCfg = PipelineConfig.REFERENCE
        val refs = ArrayList<RefOut>()
        for (attempt in listOf(PipelineConfig.REFERENCE, PipelineConfig.uniform(Variant.QUALITY, Proc.NPU, Proc.NPU))) {
            try {
                orchestrator.config = attempt
                refs.clear()
                for (i in bitmaps.indices) {
                    load(i)
                    orchestrator.process(frame, render = false)
                    refs.add(RefOut(frame.depth?.let { resample(it.normalizedDepth, it.width, it.height, GRID) }, frame.instances))
                }
                refCfg = attempt
                break
            } catch (e: ModelBank.Unavailable) {
                Log.w(TAG, "reference ${attempt.label} unavailable, trying next: ${e.message}")
            }
        }
        require(refs.size == bitmaps.size) { "no reference configuration could run" }
        Log.i(TAG, "reference = ${refCfg.label}")

        val header = "config,depth_variant,depth_proc,seg_variant,seg_proc,parallel,power,status,frames," +
            "total_p50,total_p90,total_p99,total_mean,depth_exec_p50,seg_exec_p50,depth_stage_p50,seg_stage_p50,pose_p50," +
            "load_ms,depth_rmse,depth_absrel,seg_miou,seg_pred_per_img,seg_ref_per_img," +
            "cpu_c_start,gpu_c_start,npu_c_start,cpu_c_end,gpu_c_end,npu_c_end,thermal_status_end,headroom_end,current_ma_mean,voltage_v,reference"
        summaryFile.writeText(header + "\n")
        framesFile.writeText("config,iter,image,total_ms,depth_exec_ms,seg_exec_ms,depth_stage_ms,seg_stage_ms,pose_ms,objects\n")

        for ((ci, cfg) in opt.configs.withIndex()) {
            if (opt.cooldownS > 0 && ci > 0) Thread.sleep(opt.cooldownS * 1000L)
            orchestrator.config = cfg
            val c0 = chip.sample()
            val cols = "${q(cfg.label)},${cfg.depth.variant},${cfg.depth.proc},${cfg.seg.variant},${cfg.seg.proc},${cfg.parallel},${cfg.depth.power}"

            // First frame pays any model loads.
            val loadMs = try {
                load(0)
                orchestrator.process(frame, render = false)
                frame.timings.switchMs
            } catch (e: ModelBank.Unavailable) {
                Log.w(TAG, "[${ci + 1}/${opt.configs.size}] ${cfg.label}: ${e.message}")
                summaryFile.appendText("$cols,${q("unavailable: " + e.message)},0" + ",".repeat(header.count { it == ',' } - 8) + "\n")
                continue
            }
            Log.i(TAG, "${cfg.label} ran: ${frame.ranLabel}")
            repeat(opt.warmup) { load(it); orchestrator.process(frame, render = false) }

            // Quality: one pass over the images.
            var rmseSum = 0.0; var absRelSum = 0.0; var depthN = 0
            var iouSum = 0.0; var iouN = 0; var predN = 0; var refN = 0
            for (i in bitmaps.indices) {
                load(i)
                orchestrator.process(frame, render = false)
                val ref = refs[i]
                val d = frame.depth
                if (d != null && ref.depth != null) {
                    val (rmse, absRel) = depthError(resample(d.normalizedDepth, d.width, d.height, GRID), ref.depth)
                    rmseSum += rmse; absRelSum += absRel; depthN++
                }
                for (r in ref.instances) { iouSum += bestMaskIou(r, frame.instances); iouN++ }
                predN += frame.instances.size; refN += ref.instances.size
            }

            // Latency.
            val n = opt.iters
            val total = FloatArray(n); val dExec = FloatArray(n); val sExec = FloatArray(n)
            val dStage = FloatArray(n); val sStage = FloatArray(n); val pose = FloatArray(n)
            var currentSum = 0.0; var currentN = 0
            val rows = StringBuilder()
            for (k in 0 until n) {
                load(k)
                orchestrator.process(frame, render = false)
                val t = frame.timings
                total[k] = t.totalMs; dExec[k] = t.depthExecMs; sExec[k] = t.segExecMs
                dStage[k] = t.depthStageMs; sStage[k] = t.segStageMs; pose[k] = t.poseMs
                rows.append("${q(cfg.label)},$k,${bitmaps[k % bitmaps.size].first},${f(t.totalMs)},${f(t.depthExecMs)},${f(t.segExecMs)},${f(t.depthStageMs)},${f(t.segStageMs)},${f(t.poseMs)},${frame.instances.size}\n")
                if (k % 10 == 0) chip.sample().currentMa.takeIf { !it.isNaN() }?.let { currentSum += it; currentN++ }
            }
            framesFile.appendText(rows.toString())
            val c1 = chip.sample()

            val nImg = bitmaps.size.toFloat()
            val row = listOf(
                "ok", "$n",
                f(pct(total, 50)), f(pct(total, 90)), f(pct(total, 99)), f(total.average().toFloat()),
                f(pct(dExec, 50)), f(pct(sExec, 50)), f(pct(dStage, 50)), f(pct(sStage, 50)), f(pct(pose, 50)),
                f(loadMs),
                f4(if (depthN > 0) rmseSum / depthN else Double.NaN), f4(if (depthN > 0) absRelSum / depthN else Double.NaN),
                f4(if (iouN > 0) iouSum / iouN else Double.NaN), f(predN / nImg), f(refN / nImg),
                f(c0.cpuC), f(c0.gpuC), f(c0.npuC), f(c1.cpuC), f(c1.gpuC), f(c1.npuC),
                "${c1.thermalStatus}", f4(c1.headroom.toDouble()),
                f(if (currentN > 0) (currentSum / currentN).toFloat() else Float.NaN), f(c1.voltageV),
                q(refCfg.label),
            ).joinToString(",")
            summaryFile.appendText("$cols,$row\n")
            Log.i(TAG, "[%d/%d] %-28s p50 %6.1f p99 %6.1f ms | D exec %5.1f S exec %5.1f | depthRMSE %.4f segIoU %.3f | load %.0f ms | NPU %.0f°".format(
                ci + 1, opt.configs.size, cfg.label, pct(total, 50), pct(total, 99), pct(dExec, 50), pct(sExec, 50),
                if (depthN > 0) rmseSum / depthN else Double.NaN, if (iouN > 0) iouSum / iouN else Double.NaN, loadMs, c1.npuC))
        }
        Log.i(TAG, "bench done: ${summaryFile.path}")
    }

    // --- metrics -----------------------------------------------------------

    /** Bilinear resample of a w x h map to g x g. */
    private fun resample(src: FloatArray, w: Int, h: Int, g: Int): FloatArray {
        val out = FloatArray(g * g)
        for (y in 0 until g) {
            val sy = ((y + 0.5f) * h / g - 0.5f).coerceIn(0f, h - 1f)
            val y0 = sy.toInt(); val y1 = min(y0 + 1, h - 1); val fy = sy - y0
            for (x in 0 until g) {
                val sx = ((x + 0.5f) * w / g - 0.5f).coerceIn(0f, w - 1f)
                val x0 = sx.toInt(); val x1 = min(x0 + 1, w - 1); val fx = sx - x0
                val a = src[y0 * w + x0] * (1 - fx) + src[y0 * w + x1] * fx
                val b = src[y1 * w + x0] * (1 - fx) + src[y1 * w + x1] * fx
                out[y * g + x] = a * (1 - fy) + b * fy
            }
        }
        return out
    }

    /** RMSE and AbsRel of [pred] after least-squares scale+shift alignment to [ref]. */
    private fun depthError(pred: FloatArray, ref: FloatArray): Pair<Double, Double> {
        val n = pred.size
        var sp = 0.0; var sr = 0.0; var spp = 0.0; var spr = 0.0
        for (i in 0 until n) { val p = pred[i].toDouble(); val r = ref[i].toDouble(); sp += p; sr += r; spp += p * p; spr += p * r }
        val den = n * spp - sp * sp
        val a = if (abs(den) < 1e-12) 1.0 else (n * spr - sp * sr) / den
        val b = (sr - a * sp) / n
        var se = 0.0; var rel = 0.0; var relN = 0
        for (i in 0 until n) {
            val e = a * pred[i] + b - ref[i]
            se += e * e
            if (ref[i] > ABSREL_MIN) { rel += abs(e) / ref[i]; relN++ }
        }
        return sqrt(se / n) to (if (relN > 0) rel / relN else Double.NaN)
    }

    private fun bestMaskIou(ref: SegInstance, preds: List<SegInstance>): Double {
        val match = preds.filter { it.classId == ref.classId }.maxByOrNull { boxIou(it.box, ref.box) } ?: return 0.0
        if (boxIou(match.box, ref.box) <= 0f) return 0.0
        var inter = 0; var union = 0
        for (y in 0 until MASK_GRID) for (x in 0 until MASK_GRID) {
            val a = maskAt(ref, x, y); val b = maskAt(match, x, y)
            if (a && b) inter++
            if (a || b) union++
        }
        return if (union == 0) 0.0 else inter.toDouble() / union
    }

    private fun maskAt(s: SegInstance, gx: Int, gy: Int): Boolean {
        val mx = (gx * s.maskWidth) / MASK_GRID
        val my = (gy * s.maskHeight) / MASK_GRID
        return s.mask[my * s.maskWidth + mx] > 0.5f
    }

    private fun boxIou(a: FloatArray, b: FloatArray): Float {
        val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = iw * ih
        val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** Nearest-rank percentile. */
    private fun pct(v: FloatArray, p: Int): Float {
        if (v.isEmpty()) return Float.NaN
        val s = v.sortedArray()
        val rank = Math.ceil(p / 100.0 * s.size).toInt().coerceIn(1, s.size)
        return s[rank - 1]
    }

    private fun f(v: Float) = if (v.isNaN()) "" else "%.2f".format(Locale.US, v)
    private fun f4(v: Double) = if (v.isNaN()) "" else "%.4f".format(Locale.US, v)
    private fun q(s: String) = "\"" + s.replace("\"", "'") + "\""
}
