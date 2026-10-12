package com.example.rs1.pipeline

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import com.example.rs1.research.ModelBank
import com.example.rs1.research.PipelineConfig
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Runs depth + segmentation under the current [PipelineConfig], then
 * computes geometric pose from their combined output (pose has a real
 * dependency on both, unlike the two networks).
 *
 * Parallel placement gives each network its own thread, so they overlap on
 * different processors (or on the same one, if both are placed there) and
 * one model's CPU pre/post-processing overlaps the other's execution.
 * Sequential placement runs depth then segmentation, the naive baseline of
 * proposal Fig. 4a. Models come from a [ModelBank], so switching
 * configuration does not reload anything already used.
 */
class PipelineOrchestrator(private val bank: ModelBank, initial: PipelineConfig) {

    companion object {
        private const val TAG = "PipelineDebug"
        private const val MASK_ALPHA = 140
        private const val EDGE_LO = 0.35f
        private const val EDGE_HI = 0.65f
    }

    /** Read once per frame; set from any thread to switch configuration. */
    @Volatile var config: PipelineConfig = initial

    private val poseEstimator = PoseEstimator()

    private val depthExecutor = Executors.newSingleThreadExecutor { Thread(it, "depth-stage") }
    private val segExecutor = Executors.newSingleThreadExecutor { Thread(it, "seg-stage") }

    private var depthPixels = IntArray(0)
    private var maskPixels = IntArray(0)
    private val turbo = IntArray(256) { turboColor(it / 255f) }

    /**
     * [frame.camera] must already hold the current camera image. With
     * [render] off (bench), the overlay bitmaps are not drawn, so timings
     * cover only the pipeline itself.
     *
     * @throws ModelBank.Unavailable if the configuration cannot be loaded.
     */
    fun process(frame: RenderFrame, render: Boolean = true) {
        val cfg = config
        val tLoad = System.nanoTime()
        val depthStage = bank.depth(cfg.depth)
        val segStage = bank.seg(cfg.seg)
        val switchMs = (System.nanoTime() - tLoad) / 1e6f

        val t0 = System.nanoTime()
        val cam = frame.camera!!

        val runDepth = {
            runCatching { depthStage.run(cam).also { if (render) colorizeDepth(it, frame.depthBitmap(depthStage.size)) } }
                .onFailure { Log.e(TAG, "Depth stage threw: ${it.message}", it) }
                .getOrNull()
        }
        val runSeg = {
            runCatching { segStage.run(cam).also { if (render) drawMasks(it, frame.maskBitmap(segStage.protoSize)) } }
                .onFailure { Log.e(TAG, "Segmentation stage threw: ${it.message}", it) }
                .getOrNull()
        }
        val depth: DepthResult?
        val seg: SegResult?
        if (cfg.parallel) {
            val depthFuture = depthExecutor.submit<DepthResult?>(runDepth)
            val segFuture = segExecutor.submit<SegResult?>(runSeg)
            depth = depthFuture.get()
            seg = segFuture.get()
        } else {
            depth = runDepth()
            seg = runSeg()
        }

        val tPose = System.nanoTime()
        val pose = if (depth != null && seg != null) {
            runCatching { poseEstimator.run(seg, depth, cam.width, cam.height) }
                .onFailure { Log.e(TAG, "Pose estimation threw: ${it.message}", it) }
                .getOrNull()
        } else null
        val tEnd = System.nanoTime()

        frame.hasDepth = depth != null
        frame.depth = depth
        frame.seg = seg
        frame.instances = seg?.instances ?: emptyList()
        frame.poses = pose?.objects ?: emptyList()
        frame.configLabel = cfg.label
        frame.ranLabel = "${depthStage.describe()} | ${segStage.describe()}"
        frame.timings = StageTimings(
            totalMs = (tEnd - t0) / 1e6f,
            depthExecMs = depthStage.lastExecMs,
            segExecMs = segStage.lastExecMs,
            depthStageMs = depthStage.lastStageMs,
            segStageMs = segStage.lastStageMs,
            poseMs = (tEnd - tPose) / 1e6f,
            switchMs = switchMs,
        )
    }

    private fun colorizeDepth(depth: DepthResult, target: Bitmap) {
        val src = depth.normalizedBytes
        if (depthPixels.size != src.size) depthPixels = IntArray(src.size)
        for (i in src.indices) depthPixels[i] = turbo[src[i].toInt() and 0xFF]
        target.setPixels(depthPixels, 0, depth.width, 0, 0, depth.width, depth.height)
    }

    /**
     * All instance masks composited into one small bitmap, drawn scaled with
     * filtering. Alpha ramps with mask probability around the 0.5 decision
     * boundary, so upscaled edges come out smooth instead of blocky.
     */
    private fun drawMasks(seg: SegResult, target: Bitmap) {
        if (maskPixels.size != target.width * target.height) maskPixels = IntArray(target.width * target.height)
        java.util.Arrays.fill(maskPixels, 0)
        for (inst in seg.instances) {
            val rgb = classColor(inst.classId) and 0xFFFFFF
            val m = inst.mask
            for (i in m.indices) {
                val a = ((m[i] - EDGE_LO) * (1f / (EDGE_HI - EDGE_LO))).coerceIn(0f, 1f)
                if (a > 0f) maskPixels[i] = ((a * MASK_ALPHA).toInt() shl 24) or rgb
            }
        }
        target.setPixels(maskPixels, 0, target.width, 0, 0, target.width, target.height)
    }

    fun close() {
        depthExecutor.shutdown()
        segExecutor.shutdown()
    }
}

private val CLASS_PALETTE = intArrayOf(
    Color.rgb(0, 255, 100), Color.rgb(255, 100, 0), Color.rgb(0, 150, 255), Color.rgb(200, 0, 255),
    Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.rgb(128, 255, 0),
)

fun classColor(classId: Int): Int = CLASS_PALETTE[classId % CLASS_PALETTE.size]

/** Google's polynomial fit of the "turbo" colormap (t in 0..1). */
private fun turboColor(t: Float): Int {
    fun ch(c0: Float, c1: Float, c2: Float, c3: Float, c4: Float, c5: Float): Int {
        val v = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
        return (v.coerceIn(0f, 1f) * 255f).roundToInt()
    }
    val r = ch(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f, -152.94239396f, 59.28637943f)
    val g = ch(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f, 4.27729857f, 2.82956604f)
    val b = ch(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f, -89.90310912f, 27.34824973f)
    return Color.rgb(r, g, b)
}
