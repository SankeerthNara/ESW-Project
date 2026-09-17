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
     * Use for models expecting planar NCHW input.
     */
    fun bitmapToChwFloat(bmp: Bitmap): FloatArray = toChw(bmp, imagenet = false)

    /**
     * Bitmap -> NCHW float32 array, normalized to 0-1 then ImageNet mean/std.
     * Use for models expecting planar NCHW input.
     */
    fun bitmapToChwFloatImageNet(bmp: Bitmap): FloatArray = toChw(bmp, imagenet = true)

    /**
     * Bitmap -> NHWC float32 array (interleaved RGB), normalized to 0-1.
     * Use for YOLOv8 models in SNPE/QNN.
     */
    fun bitmapToHwcFloat(bmp: Bitmap): FloatArray = toHwc(bmp, imagenet = false)

    /**
     * Bitmap -> NHWC float32 array (interleaved RGB), normalized to 0-1 then ImageNet mean/std.
     * Use for Depth Anything V2 models in SNPE/QNN.
     */
    fun bitmapToHwcFloatImageNet(bmp: Bitmap): FloatArray = toHwc(bmp, imagenet = true)

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

    private fun toHwc(bmp: Bitmap, imagenet: Boolean): FloatArray {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)

        val out = FloatArray(3 * w * h)
        val totalPixels = w * h
        for (i in 0 until totalPixels) {
            val p = pixels[i]
            var r = ((p shr 16) and 0xFF) / 255f
            var g = ((p shr 8) and 0xFF) / 255f
            var b = (p and 0xFF) / 255f
            if (imagenet) {
                r = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                g = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                b = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            }
            val offset = i * 3
            out[offset] = r
            out[offset + 1] = g
            out[offset + 2] = b
        }
        return out
    }
}
