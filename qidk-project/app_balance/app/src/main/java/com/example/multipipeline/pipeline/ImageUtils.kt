package com.example.multipipeline.pipeline

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import com.example.multipipeline.snpe.QuantTensor

/**
 * Turns a camera frame into a model's quantized NHWC input with no
 * per-frame allocation: a bilinear (Skia) stretch into a reused square
 * bitmap, then one pass through per-channel lookup tables that fold 0-1
 * scaling, optional ImageNet mean/std and the input quantization (8- or
 * 16-bit) into a single 256-entry map per channel.
 */
class ModelInputPacker(private val input: QuantTensor, imagenet: Boolean) {
    val size: Int

    init {
        val s = input.shape
        require(s.size == 4 && s[3] == 3 && s[1] == s[2]) {
            "expected a square NHWC [1,S,S,3] input, got ${s.joinToString("x")} for '${input.name}'"
        }
        size = s[1]
    }

    private val scaled = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(scaled)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels = IntArray(size * size)

    private val lut = Array(3) { c -> IntArray(256) { v ->
        var x = v / 255f
        if (imagenet) x = (x - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        input.enc.quant(x)
    } }
    private val packed8 = if (input.enc.bits == 8) ByteArray(size * size * 3) else null
    private val packed16 = if (input.enc.bits == 16) ShortArray(size * size * 3) else null

    fun pack(frame: Bitmap) {
        matrix.setScale(size.toFloat() / frame.width, size.toFloat() / frame.height)
        canvas.drawBitmap(frame, matrix, paint)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)

        val r = lut[0]; val g = lut[1]; val b = lut[2]
        val dst = input.buffer
        dst.clear()
        if (packed8 != null) {
            var o = 0
            for (p in pixels) {
                packed8[o] = r[(p shr 16) and 0xFF].toByte()
                packed8[o + 1] = g[(p shr 8) and 0xFF].toByte()
                packed8[o + 2] = b[p and 0xFF].toByte()
                o += 3
            }
            dst.put(packed8)
        } else {
            val out = packed16!!
            var o = 0
            for (p in pixels) {
                out[o] = r[(p shr 16) and 0xFF].toShort()
                out[o + 1] = g[(p shr 8) and 0xFF].toShort()
                out[o + 2] = b[p and 0xFF].toShort()
                o += 3
            }
            dst.asShortBuffer().put(out)
        }
        dst.rewind()
    }

    companion object {
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
