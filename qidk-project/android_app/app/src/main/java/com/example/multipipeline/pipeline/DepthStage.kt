package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import com.example.multipipeline.snpe.SnpeModelRunner

/**
 * Depth Anything V2 (Large). Static input 518x518 - must match
 * model_conversion/1_export_depth_anything_v2_large.py's INPUT_SIZE.
 * Uses ImageNet mean/std normalization (see ImageUtils.bitmapToChwFloatImageNet),
 * matching Depth Anything V2's own NormalizeImage preprocessing step.
 *
 * Note: the Large variant's ViT-L/DINOv2 backbone is significantly heavier
 * than Small - watch the on-screen fps counter once running on-device. If
 * it's too slow for live camera use, swap to the Small or Base export
 * script and update INPUT_SIZE/MODEL_CONFIG accordingly - the rest of this
 * file (and the app) doesn't need to change either way.
 */
class DepthStage(context: Context) {
    companion object {
        const val INPUT_SIZE = 392 // Depth Anything V2 Small (vits) 28x28 patches static square input
        const val OUTPUT_TENSOR_NAME = "depth"
    }

    private val runner = SnpeModelRunner(context, "depth.dlc")

    fun run(frame: Bitmap): DepthResult {
        val t0 = System.currentTimeMillis()
        val resized = ImageUtils.resizeSquare(frame, INPUT_SIZE)
        val t1 = System.currentTimeMillis()
        val input = ImageUtils.bitmapToHwcFloatImageNet(resized)
        val t2 = System.currentTimeMillis()
        val outputs = runner.run(input)
        val t3 = System.currentTimeMillis()

        val raw = outputs[OUTPUT_TENSOR_NAME]
            ?: outputs.entries.firstOrNull { it.key.contains("depth", ignoreCase = true) }?.value
            ?: outputs.values.firstOrNull()
            ?: FloatArray(INPUT_SIZE * INPUT_SIZE)

        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (v in raw) {
            if (v < min) min = v
            if (v > max) max = v
        }
        val range = (max - min).takeIf { it > 1e-6f } ?: 1f
        val normalized = FloatArray(raw.size) { (raw[it] - min) / range }
        val t4 = System.currentTimeMillis()

        android.util.Log.i("ProfileDepth", "resize=${t1-t0}ms float=${t2-t1}ms npu=${t3-t2}ms post=${t4-t3}ms total=${t4-t0}ms")

        return DepthResult(INPUT_SIZE, INPUT_SIZE, raw, normalized)
    }

    fun close() = runner.close()
}
