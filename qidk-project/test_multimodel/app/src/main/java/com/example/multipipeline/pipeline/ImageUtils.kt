package com.example.multipipeline.pipeline

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import com.example.multipipeline.snpe.QuantEncoding
import java.nio.ByteBuffer

/**
 * Turns a camera frame into a model's quantized NHWC uint8 input with no
 * per-frame allocation: a bilinear (Skia) stretch into a reused square
 * bitmap, then one pass that packs RGB bytes through per-channel lookup
 * tables. The tables fold 0-1 scaling, optional ImageNet mean/std, and the
 * model's input quantization into a single byte->byte map.
 */
class ModelInputPacker(
    private val size: Int,
    inputEncoding: QuantEncoding,
    imagenet: Boolean,
) {
    private val scaled = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(scaled)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels = IntArray(size * size)
    private val packed = ByteArray(size * size * 3)

    private val lutR = buildLut(inputEncoding, if (imagenet) 0 else -1)
    private val lutG = buildLut(inputEncoding, if (imagenet) 1 else -1)
    private val lutB = buildLut(inputEncoding, if (imagenet) 2 else -1)

    fun pack(frame: Bitmap, dst: ByteBuffer) {
        matrix.setScale(size.toFloat() / frame.width, size.toFloat() / frame.height)
        canvas.drawBitmap(frame, matrix, paint)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)

        val r = lutR; val g = lutG; val b = lutB; val out = packed
        var o = 0
        for (p in pixels) {
            out[o] = r[(p shr 16) and 0xFF]
            out[o + 1] = g[(p shr 8) and 0xFF]
            out[o + 2] = b[p and 0xFF]
            o += 3
        }
        dst.clear()
        dst.put(out)
        dst.rewind()
    }

    companion object {
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        /** channel = -1 for plain 0-1 scaling, 0/1/2 for ImageNet R/G/B. */
        private fun buildLut(enc: QuantEncoding, channel: Int) = ByteArray(256) { v ->
            var x = v / 255f
            if (channel >= 0) x = (x - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
            enc.quant(x).toByte()
        }
    }
}
