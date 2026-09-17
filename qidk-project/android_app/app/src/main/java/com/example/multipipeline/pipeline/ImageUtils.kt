package com.example.multipipeline.pipeline

import android.graphics.Bitmap
import android.graphics.Matrix

object ImageUtils {

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** Resize (simple stretch) to a square of [size]x[size]. */
    fun resizeSquare(src: Bitmap, size: Int): Bitmap {
        if (src.width == size && src.height == size) return src
        val m = Matrix()
        m.setScale(size.toFloat() / src.width, size.toFloat() / src.height)
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    /**
     * Bitmap -> NCHW float32 array, normalized to 0-1.
     * Use for YOLOv8 models (seg) - keep in sync with make_raw_inputs.py
     * --normalize unit (the default) from the model conversion phase.
     */
    fun bitmapToChwFloat(bmp: Bitmap): FloatArray = toChw(bmp, imagenet = false)

    /**
     * Bitmap -> NCHW float32 array, normalized to 0-1 then ImageNet mean/std.
     * Use for MiDaS depth model - keep in sync with make_raw_inputs.py
     * --normalize imagenet from the model conversion phase.
     */
    fun bitmapToChwFloatImageNet(bmp: Bitmap): FloatArray = toChw(bmp, imagenet = true)

    private fun toChw(bmp: Bitmap, imagenet: Boolean): FloatArray {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)

        val out = FloatArray(3 * w * h)
        val plane = w * h
        for (i in 0 until plane) {
            val p = pixels[i]
            var r = ((p shr 16) and 0xFF) / 255f
            var g = ((p shr 8) and 0xFF) / 255f
            var b = (p and 0xFF) / 255f
            if (imagenet) {
                r = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                g = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                b = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            }
            out[i] = r
            out[plane + i] = g
            out[2 * plane + i] = b
        }
        return out
    }
}
