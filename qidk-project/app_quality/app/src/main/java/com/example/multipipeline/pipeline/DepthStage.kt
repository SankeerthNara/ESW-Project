package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import com.example.multipipeline.snpe.SnpeModelRunner

/**
 * Depth Anything V2 Small (vits) from assets/depth.dlc - a static square
 * input whose side is a multiple of the 14 px ViT patch. ImageNet mean/std
 * input normalization, matching the model's own NormalizeImage step; the
 * network outputs relative disparity (larger = closer).
 */
class DepthStage(context: Context) {
    private val runner = SnpeModelRunner(context, "depth.dlc")
    private val output = runner.outputs.values.single()
    private val packer = ModelInputPacker(runner.inputs.values.single(), imagenet = true)

    /** Side of the square depth map (== model input size). */
    val size = packer.size

    init {
        require(output.size == size * size) { "depth output ${output.shape.joinToString("x")} != ${size}x$size" }
    }

    private val raw = FloatArray(size * size)
    private val normalized = FloatArray(size * size)
    private val normalizedBytes = ByteArray(size * size)

    var lastNpuMs = 0L; private set

    /**
     * The returned arrays are reused on the next call; consume them (pose,
     * colorization) before running this stage again.
     */
    fun run(frame: Bitmap): DepthResult {
        packer.pack(frame)
        val t0 = System.nanoTime()
        runner.execute()
        lastNpuMs = (System.nanoTime() - t0) / 1_000_000

        output.readFloats(raw)
        // Per-frame min-max stretch to 0-1 for display and pose.
        var lo = Float.MAX_VALUE; var hi = -Float.MAX_VALUE
        for (v in raw) {
            if (v < lo) lo = v
            if (v > hi) hi = v
        }
        val inv = 1f / (hi - lo).coerceAtLeast(1e-6f)
        for (i in raw.indices) {
            val n = ((raw[i] - lo) * inv).coerceIn(0f, 1f)
            normalized[i] = n
            normalizedBytes[i] = (n * 255f + 0.5f).toInt().toByte()
        }
        return DepthResult(size, size, normalized, normalizedBytes)
    }

    fun close() = runner.close()
}
