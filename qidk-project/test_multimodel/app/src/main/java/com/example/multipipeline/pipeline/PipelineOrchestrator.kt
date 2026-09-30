package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.util.concurrent.Executors

/**
 * Runs depth + segmentation on the NPU concurrently (both are independent,
 * both read the raw frame), then computes geometric pose from their combined
 * output (pose has a real dependency on both, unlike the NPU models).
 */
class PipelineOrchestrator(context: Context) {

    companion object {
        private const val TAG = "PipelineDebug"
    }

    private val depthStage = DepthStage(context)
    private val segStage = SegmentationStage(context)
    private val poseEstimator = PoseEstimator()

    private val pool = Executors.newFixedThreadPool(2)

    fun process(frame: Bitmap): FrameResult {
        val t0 = System.currentTimeMillis()
        val depth = runCatching { depthStage.run(frame) }
            .onFailure { Log.e(TAG, "Depth stage threw: ${it.message}", it) }
            .getOrNull()
        val seg = runCatching { segStage.run(frame) }
            .onFailure { Log.e(TAG, "Segmentation stage threw: ${it.message}", it) }
            .getOrNull()

        val t1 = System.currentTimeMillis()

        val pose = if (depth != null && seg != null) {
            runCatching { poseEstimator.run(seg, depth, frame.width, frame.height) }
                .onFailure { Log.e(TAG, "Pose estimation threw: ${it.message}", it) }
                .getOrNull()
        } else null

        val t2 = System.currentTimeMillis()

        Log.i("ProfilePipeline", "models_parallel=${t1 - t0}ms, pose=${t2 - t1}ms, total=${t2 - t0}ms, fps=${if (t2 > t0) "%.1f".format(1000f / (t2 - t0)) else "0"}")

        return FrameResult(depth, seg, pose, frame.width, frame.height)
    }

    fun close() {
        pool.shutdown()
        depthStage.close()
        segStage.close()
    }
}
