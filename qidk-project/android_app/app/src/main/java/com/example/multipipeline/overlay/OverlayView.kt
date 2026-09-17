package com.example.multipipeline.overlay

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.example.multipipeline.pipeline.FrameResult
import com.example.multipipeline.pipeline.ObjectPose
import com.example.multipipeline.pipeline.SegInstance
import com.example.multipipeline.pipeline.SegmentationStage
import com.example.multipipeline.pipeline.ViewMode

/**
 * Owns ALL rendering - the camera frame itself plus every model's output -
 * so the 2x2 "ALL" layout can show the live frame in exactly one quadrant
 * while the others show colormap/black-background content, matching the
 * reference pipeline's 4-screen dashboard. CameraX's PreviewView is not used
 * for display; MainActivity keeps it hidden and only uses it to drive the
 * camera pipeline.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    @Volatile private var result: FrameResult? = null
    @Volatile private var frameBitmap: Bitmap? = null
    @Volatile private var mode: ViewMode = ViewMode.PIPELINE
    private var depthBitmap: Bitmap? = null

    private val fillPaint = Paint().apply { style = Paint.Style.FILL }
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 3f; color = Color.YELLOW }
    private val textPaint = Paint().apply {
        color = Color.WHITE; textSize = 24f; isAntiAlias = true
        setShadowLayer(3f, 0f, 0f, Color.BLACK)
    }
    private val axisXPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.RED }
    private val axisYPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.GREEN }
    private val axisZPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.rgb(255, 150, 0) }
    private val originPaint = Paint().apply { style = Paint.Style.FILL; color = Color.YELLOW }

    fun setMode(newMode: ViewMode) {
        mode = newMode
        postInvalidate()
    }

    fun update(newResult: FrameResult, newFrameBitmap: Bitmap) {
        result = newResult
        frameBitmap = newFrameBitmap
        newResult.depth?.let { depthBitmap = buildDepthBitmap(it.normalizedDepth, it.width, it.height) }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val frame = frameBitmap
        if (frame == null) {
            fillPaint.color = Color.BLACK
            canvas.drawRect(fullRect(), fillPaint)
            canvas.drawText("Waiting for camera feed...", 32f, height / 2f, textPaint)
            return
        }

        val r = result
        if (r == null) {
            drawBitmapInto(canvas, frame, fullRect())
            canvas.drawText("Initializing models on NPU...", 32f, 60f, textPaint)
            return
        }

        when (mode) {
            ViewMode.DEPTH -> drawDepthLayer(canvas, fullRect(), opaque = true)
            ViewMode.SEGMENTATION -> drawSegLayer(canvas, fullRect(), frame, r, blackBg = false)
            ViewMode.POSE -> drawPoseLayer(canvas, fullRect(), frame, r, blackBg = false)
            ViewMode.PIPELINE -> drawPipelineLayer(canvas, fullRect(), frame, r)
            ViewMode.ALL -> {
                val halfW = width / 2f
                val halfH = height / 2f
                val topLeft = RectF(0f, 0f, halfW, halfH)
                val topRight = RectF(halfW, 0f, width.toFloat(), halfH)
                val bottomLeft = RectF(0f, halfH, halfW, height.toFloat())
                val bottomRight = RectF(halfW, halfH, width.toFloat(), height.toFloat())

                drawDepthLayer(canvas, topLeft, opaque = true)
                drawSegLayer(canvas, topRight, frame, r, blackBg = false)
                drawPoseLayer(canvas, bottomLeft, frame, r, blackBg = false)
                drawPipelineLayer(canvas, bottomRight, frame, r)

                // quadrant header labels for clarity
                canvas.drawText("DEPTH (DA-V2 Small)", topLeft.left + 16f, topLeft.top + 32f, textPaint)
                canvas.drawText("SEGMENTATION", topRight.left + 16f, topRight.top + 32f, textPaint)
                canvas.drawText("POSE (6-DoF)", bottomLeft.left + 16f, bottomLeft.top + 32f, textPaint)
                canvas.drawText("PIPELINE", bottomRight.left + 16f, bottomRight.top + 32f, textPaint)

                // thin divider lines between quadrants
                val dividerPaint = Paint().apply { color = Color.DKGRAY; strokeWidth = 3f }
                canvas.drawLine(halfW, 0f, halfW, height.toFloat(), dividerPaint)
                canvas.drawLine(0f, halfH, width.toFloat(), halfH, dividerPaint)
            }
        }
    }

    private fun fullRect() = RectF(0f, 0f, width.toFloat(), height.toFloat())

    private fun mapX(nx: Float, rect: RectF) = rect.left + nx * rect.width()
    private fun mapY(ny: Float, rect: RectF) = rect.top + ny * rect.height()

    private fun drawBitmapInto(canvas: Canvas, bmp: Bitmap, rect: RectF) {
        canvas.drawBitmap(bmp, Rect(0, 0, bmp.width, bmp.height), rect, fillPaint)
    }

    private fun drawDepthLayer(canvas: Canvas, rect: RectF, opaque: Boolean) {
        val bmp = depthBitmap ?: run {
            fillPaint.color = Color.BLACK
            canvas.drawRect(rect, fillPaint)
            return
        }
        fillPaint.alpha = 255
        drawBitmapInto(canvas, bmp, rect)
    }

    private fun drawSegLayer(canvas: Canvas, rect: RectF, frame: Bitmap, r: FrameResult, blackBg: Boolean) {
        if (blackBg) {
            fillPaint.color = Color.BLACK
            canvas.drawRect(rect, fillPaint)
        } else {
            drawBitmapInto(canvas, frame, rect)
        }
        val instances = r.seg?.instances
        if (instances.isNullOrEmpty()) {
            val label = if (r.seg == null) "Seg: Initializing..." else "Seg: 0 objects detected"
            canvas.drawText(label, rect.left + 16f, rect.top + 36f, textPaint)
        } else {
            instances.forEach { inst -> drawSegInstance(canvas, rect, inst, drawLabel = true) }
        }
    }

    private fun drawSegInstance(canvas: Canvas, rect: RectF, inst: SegInstance, drawLabel: Boolean) {
        val size = SegmentationStage.INPUT_SIZE.toFloat()
        fillPaint.color = colorForClass(inst.classId)
        fillPaint.alpha = 130
        val cellW = rect.width() / inst.maskWidth
        val cellH = rect.height() / inst.maskHeight
        for (y in 0 until inst.maskHeight) {
            for (x in 0 until inst.maskWidth) {
                if (inst.mask[y * inst.maskWidth + x] > 0.5f) {
                    val left = rect.left + x * cellW
                    val top = rect.top + y * cellH
                    canvas.drawRect(left, top, left + cellW, top + cellH, fillPaint)
                }
            }
        }
        boxPaint.color = colorForClass(inst.classId)
        canvas.drawRect(
            mapX(inst.box[0] / size, rect), mapY(inst.box[1] / size, rect),
            mapX(inst.box[2] / size, rect), mapY(inst.box[3] / size, rect),
            boxPaint,
        )
        if (drawLabel) {
            val label = com.example.multipipeline.pipeline.CocoLabels.NAMES.getOrElse(inst.classId) { "?" }
            canvas.drawText(
                "$label ${"%.2f".format(inst.score)}",
                mapX(inst.box[0] / size, rect), mapY(inst.box[1] / size, rect) - 6f,
                textPaint,
            )
        }
    }

    private fun drawPoseLayer(canvas: Canvas, rect: RectF, frame: Bitmap, r: FrameResult, blackBg: Boolean) {
        if (blackBg) {
            fillPaint.color = Color.BLACK
            canvas.drawRect(rect, fillPaint)
        } else {
            drawBitmapInto(canvas, frame, rect)
        }
        val objects = r.pose?.objects
        if (objects.isNullOrEmpty()) {
            val label = if (r.pose == null) "Pose: Initializing..." else "Pose: 0 objects detected"
            canvas.drawText(label, rect.left + 16f, rect.top + 36f, textPaint)
        } else {
            objects.forEach { obj -> drawPoseAxes(canvas, rect, obj) }
        }
    }

    private fun drawPoseAxes(canvas: Canvas, rect: RectF, obj: ObjectPose) {
        val (ox, oy) = obj.originNorm
        val (xx, xy) = obj.xAxisEndNorm
        val (yx, yy) = obj.yAxisEndNorm
        val (zx, zy) = obj.zAxisEndNorm

        canvas.drawLine(mapX(ox, rect), mapY(oy, rect), mapX(xx, rect), mapY(xy, rect), axisXPaint)
        canvas.drawLine(mapX(ox, rect), mapY(oy, rect), mapX(yx, rect), mapY(yy, rect), axisYPaint)
        canvas.drawLine(mapX(ox, rect), mapY(oy, rect), mapX(zx, rect), mapY(zy, rect), axisZPaint)
        canvas.drawCircle(mapX(ox, rect), mapY(oy, rect), 7f, originPaint)

        val posTxt = "${obj.label} [X:%.2f Y:%.2f Z:%.2f m]".format(obj.x, obj.y, obj.z)
        val rotTxt = "Rot [R:%.0f P:%.0f Y:%.0f deg]".format(obj.roll, obj.pitch, obj.yaw)
        val labelX = mapX(obj.boxNorm[0], rect)
        val labelY = (mapY(obj.boxNorm[1], rect) - 30f).coerceAtLeast(rect.top + 20f)
        canvas.drawText(posTxt, labelX, labelY, textPaint)
        canvas.drawText(rotTxt, labelX, labelY + 22f, textPaint)
    }

    private fun drawPipelineLayer(canvas: Canvas, rect: RectF, frame: Bitmap, r: FrameResult) {
        drawBitmapInto(canvas, frame, rect)
        r.seg?.instances?.forEach { inst -> drawSegInstance(canvas, rect, inst, drawLabel = false) }
        r.pose?.objects?.forEach { obj -> drawPoseAxes(canvas, rect, obj) }
    }

    private fun buildDepthBitmap(depth: FloatArray, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        for (i in depth.indices) pixels[i] = turbo(depth[i])
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return bmp
    }

    /** Cheap blue->red colormap approximation of matplotlib's "turbo". */
    private fun turbo(v: Float): Int {
        val t = v.coerceIn(0f, 1f)
        val r = (t * 255).toInt()
        val b = ((1f - t) * 255).toInt()
        val g = (128 * (1f - kotlin.math.abs(t - 0.5f) * 2)).toInt()
        return Color.argb(255, r, g, b)
    }

    private fun colorForClass(classId: Int): Int {
        val palette = intArrayOf(
            Color.rgb(0, 255, 100), Color.rgb(255, 100, 0), Color.rgb(0, 150, 255), Color.rgb(200, 0, 255),
            Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.rgb(128, 255, 0),
        )
        return palette[classId % palette.size]
    }
}
