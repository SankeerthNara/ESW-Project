package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import com.example.multipipeline.snpe.QuantEncoding
import com.example.multipipeline.snpe.SnpeModelRunner

/**
 * Depth Anything V2 Small (vits), 392x392 (28x28 patches of 14px) INT8 with
 * an SM8650 HTP cache (depth_small_htp.dlc). ImageNet mean/std input
 * normalization, matching the model's own NormalizeImage step; the network
 * outputs relative disparity (larger = closer).
 */
class DepthStage(context: Context) {
    companion object {
        const val INPUT_SIZE = 392

        // From `snpe-dlc-info -i depth_small_htp.dlc`.
        private val ENC_IMAGE = QuantEncoding(0.018658448011f, -114)
        private val ENC_DEPTH = QuantEncoding(0.016817899421f, 0)
    }

    private val runner = SnpeModelRunner(
        context, "depth.dlc",
        inputEncodings = mapOf("image" to ENC_IMAGE),
        outputEncodings = mapOf("depth" to ENC_DEPTH),
    )
    private val packer = ModelInputPacker(INPUT_SIZE, ENC_IMAGE, imagenet = true)
    private val raw = ByteArray(INPUT_SIZE * INPUT_SIZE)

    private val normalized = FloatArray(INPUT_SIZE * INPUT_SIZE)
    private val normalizedBytes = ByteArray(INPUT_SIZE * INPUT_SIZE)

    var lastNpuMs = 0L; private set

    /**
     * The returned arrays are reused on the next call; consume them (pose,
     * colorization) before running this stage again.
     */
    fun run(frame: Bitmap): DepthResult {
        packer.pack(frame, runner.inputBuffers.getValue("image"))
        val t0 = System.nanoTime()
        runner.execute()
        lastNpuMs = (System.nanoTime() - t0) / 1_000_000

        val buf = runner.outputBuffers.getValue("depth")
        buf.rewind()
        buf.get(raw)

        // Per-frame min-max stretch, done on the quantized values directly
        // (dequantization is affine, so it cancels out of the ratio).
        var lo = 255; var hi = 0
        for (b in raw) {
            val q = b.toInt() and 0xFF
            if (q < lo) lo = q
            if (q > hi) hi = q
        }
        val range = (hi - lo).coerceAtLeast(1).toFloat()
        val lut = FloatArray(256) { ((it - lo) / range).coerceIn(0f, 1f) }
        for (i in raw.indices) {
            val q = raw[i].toInt() and 0xFF
            normalized[i] = lut[q]
            normalizedBytes[i] = (lut[q] * 255f + 0.5f).toInt().toByte()
        }
        return DepthResult(INPUT_SIZE, INPUT_SIZE, normalized, normalizedBytes)
    }

    fun close() = runner.close()
}
