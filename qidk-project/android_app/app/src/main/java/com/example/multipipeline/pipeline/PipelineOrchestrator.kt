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
        val depthFuture = pool.submit<DepthResult> { depthStage.run(frame) }
        val segFuture = pool.submit<SegResult> { segStage.run(frame) }

        val depth = runCatching { depthFuture.get() }
            .onFailure { Log.e(TAG, "Depth stage threw: ${it.message}", it) }
            .getOrNull()

        val seg = runCatching { segFuture.get() }
            .onFailure { Log.e(TAG, "Segmentation stage threw: ${it.message}", it) }
            .getOrNull()

        Log.i(
            TAG,
            "depth=${depth != null} (size=${depth?.width}x${depth?.height}), " +
                    "seg instances=${seg?.instances?.size ?: "null"}"
        )

        val pose = if (depth != null && seg != null) {
            runCatching { poseEstimator.run(seg, depth, frame.width, frame.height) }
                .onFailure { Log.e(TAG, "Pose estimation threw: ${it.message}", it) }
                .getOrNull()
        } else null

        Log.i(TAG, "pose objects=${pose?.objects?.size ?: "null (depth or seg was null)"}")

        return FrameResult(depth, seg, pose, frame.width, frame.height)
    }

    fun close() {
        pool.shutdown()
        depthStage.close()
        segStage.close()
    }
}
