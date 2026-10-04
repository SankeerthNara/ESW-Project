package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Runs depth + segmentation on the NPU concurrently (both are independent,
 * both read the raw frame), then computes geometric pose from their combined
 * output (pose has a real dependency on both, unlike the NPU models).
 *
 * Each model gets its own thread, so one model's CPU pre/post-processing
 * overlaps the other's NPU execution. The render bitmaps (depth colormap,
 * mask overlay) are also built here rather than on the UI thread.
 */
class PipelineOrchestrator(context: Context) {

    companion object {
        private const val TAG = "PipelineDebug"
        private const val MASK_ALPHA = 140
        private const val EDGE_LO = 0.35f
        private const val EDGE_HI = 0.65f
    }

    private val depthStage = DepthStage(context)
    private val segStage = SegmentationStage(context)
    private val poseEstimator = PoseEstimator()

    private val depthExecutor = Executors.newSingleThreadExecutor { Thread(it, "depth-stage") }
    private val segExecutor = Executors.newSingleThreadExecutor { Thread(it, "seg-stage") }

    private val depthPixels = IntArray(depthStage.size * depthStage.size)
    private val maskPixels = IntArray(segStage.protoSize * segStage.protoSize)
    private val turbo = IntArray(256) { turboColor(it / 255f) }

    /** [frame.camera] must already hold the current camera image. */
    fun process(frame: RenderFrame) {
        val t0 = System.nanoTime()
        val cam = frame.camera!!

        val depthFuture = depthExecutor.submit<DepthResult?> {
            runCatching { depthStage.run(cam).also { colorizeDepth(it, frame.depthBitmap(depthStage.size)) } }
                .onFailure { Log.e(TAG, "Depth stage threw: ${it.message}", it) }
                .getOrNull()
        }
        val segFuture = segExecutor.submit<SegResult?> {
            runCatching { segStage.run(cam).also { drawMasks(it, frame.maskBitmap(segStage.protoSize)) } }
                .onFailure { Log.e(TAG, "Segmentation stage threw: ${it.message}", it) }
                .getOrNull()
        }
        val depth = depthFuture.get()
        val seg = segFuture.get()

        val pose = if (depth != null && seg != null) {
            runCatching { poseEstimator.run(seg, depth, cam.width, cam.height) }
                .onFailure { Log.e(TAG, "Pose estimation threw: ${it.message}", it) }
                .getOrNull()
        } else null

        frame.hasDepth = depth != null
        frame.instances = seg?.instances ?: emptyList()
        frame.poses = pose?.objects ?: emptyList()
        frame.timings = StageTimings(
            totalMs = (System.nanoTime() - t0) / 1_000_000,
            depthNpuMs = depthStage.lastNpuMs,
            segNpuMs = segStage.lastNpuMs,
        )
    }

    private fun colorizeDepth(depth: DepthResult, target: Bitmap) {
        val src = depth.normalizedBytes
        for (i in src.indices) depthPixels[i] = turbo[src[i].toInt() and 0xFF]
        target.setPixels(depthPixels, 0, depth.width, 0, 0, depth.width, depth.height)
    }

    /**
     * All instance masks composited into one small bitmap, drawn scaled with
     * filtering. Alpha ramps with mask probability around the 0.5 decision
     * boundary, so upscaled edges come out smooth instead of blocky.
     */
    private fun drawMasks(seg: SegResult, target: Bitmap) {
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
        depthStage.close()
        segStage.close()
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
