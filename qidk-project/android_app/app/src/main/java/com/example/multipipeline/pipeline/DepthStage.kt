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
        const val INPUT_SIZE = 392 // Depth Anything V2 Small (vits) static square input
        const val OUTPUT_TENSOR_NAME = "depth"
    }

    private val runner = SnpeModelRunner(context, "depth.dlc")

    fun run(frame: Bitmap): DepthResult {
        val resized = ImageUtils.resizeSquare(frame, INPUT_SIZE)
        val input = ImageUtils.bitmapToChwFloatImageNet(resized)
        val outputs = runner.run(input)

        val raw = outputs[OUTPUT_TENSOR_NAME]
            ?: outputs.entries.firstOrNull { it.key.contains("depth", ignoreCase = true) }?.value
            ?: outputs.values.firstOrNull()
            ?: FloatArray(INPUT_SIZE * INPUT_SIZE)

        // Depth Anything V2 outputs relative (inverse) depth, arbitrary scale.
        // Normalize to 0-1 across this frame for display and for the
        // pseudo-metric mapping pose estimation uses (0.3m-2.5m range).
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (v in raw) {
            if (v < min) min = v
            if (v > max) max = v
        }
        val range = (max - min).takeIf { it > 1e-6f } ?: 1f
        val normalized = FloatArray(raw.size) { (raw[it] - min) / range }

        return DepthResult(INPUT_SIZE, INPUT_SIZE, raw, normalized)
    }

    fun close() = runner.close()
}
