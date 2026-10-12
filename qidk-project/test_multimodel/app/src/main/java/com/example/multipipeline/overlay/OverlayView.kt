package com.example.multipipeline.overlay

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.example.multipipeline.pipeline.CocoLabels
import com.example.multipipeline.pipeline.FramePool
import com.example.multipipeline.pipeline.ObjectPose
import com.example.multipipeline.pipeline.RenderFrame
import com.example.multipipeline.pipeline.SegInstance
import com.example.multipipeline.pipeline.SegmentationStage
import com.example.multipipeline.pipeline.ViewMode
import com.example.multipipeline.pipeline.classColor

/**
 * Owns ALL rendering - the camera frame itself plus every model's output -
 * so the 2x2 "ALL" layout can show the live frame in each quadrant next to
 * the depth colormap, matching the reference pipeline's 4-screen dashboard.
 *
 * Everything heavy (depth colormap, combined instance masks) arrives as
 * prebuilt bitmaps in a [RenderFrame], so onDraw is a few scaled bitmap
 * draws plus lines and text.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    var framePool: FramePool? = null
    @Volatile private var mode: ViewMode = ViewMode.PIPELINE

    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val fillPaint = Paint().apply { style = Paint.Style.FILL; color = Color.BLACK }
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 3f }
    private val textPaint = Paint().apply {
        color = Color.WHITE; textSize = 24f; isAntiAlias = true
        setShadowLayer(3f, 0f, 0f, Color.BLACK)
    }
    private val axisXPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.RED }
    private val axisYPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.GREEN }
    private val axisZPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.rgb(255, 150, 0) }
    private val originPaint = Paint().apply { style = Paint.Style.FILL; color = Color.YELLOW }
    private val dividerPaint = Paint().apply { color = Color.DKGRAY; strokeWidth = 3f }

    private val full = RectF()
    private val quad = Array(4) { RectF() }

    fun setMode(newMode: ViewMode) {
        mode = newMode
        postInvalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        val hw = w / 2f; val hh = h / 2f
        full.set(0f, 0f, w.toFloat(), h.toFloat())
        quad[0].set(0f, 0f, hw, hh)
        quad[1].set(hw, 0f, w.toFloat(), hh)
        quad[2].set(0f, hh, hw, h.toFloat())
        quad[3].set(hw, hh, w.toFloat(), h.toFloat())
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = framePool?.latestForDraw()
        val cam = f?.camera
        if (f == null || cam == null) {
            canvas.drawRect(full, fillPaint)
            canvas.drawText("Waiting for camera feed...", 32f, height / 2f, textPaint)
            return
        }

        when (mode) {
            ViewMode.DEPTH -> drawDepthLayer(canvas, full, f)
            ViewMode.SEGMENTATION -> drawSegLayer(canvas, full, cam, f)
            ViewMode.POSE -> drawPoseLayer(canvas, full, cam, f)
            ViewMode.PIPELINE -> drawPipelineLayer(canvas, full, cam, f)
            ViewMode.ALL -> {
                drawDepthLayer(canvas, quad[0], f)
                drawSegLayer(canvas, quad[1], cam, f)
                drawPoseLayer(canvas, quad[2], cam, f)
                drawPipelineLayer(canvas, quad[3], cam, f)

                canvas.drawText("DEPTH (DA-V2 Small)", quad[0].left + 16f, quad[0].top + 32f, textPaint)
                canvas.drawText("SEGMENTATION", quad[1].left + 16f, quad[1].top + 32f, textPaint)
                canvas.drawText("POSE (6-DoF)", quad[2].left + 16f, quad[2].top + 32f, textPaint)
                canvas.drawText("PIPELINE", quad[3].left + 16f, quad[3].top + 32f, textPaint)

                canvas.drawLine(full.centerX(), 0f, full.centerX(), full.bottom, dividerPaint)
                canvas.drawLine(0f, full.centerY(), full.right, full.centerY(), dividerPaint)
            }
        }
    }

    private fun mapX(nx: Float, rect: RectF) = rect.left + nx * rect.width()
    private fun mapY(ny: Float, rect: RectF) = rect.top + ny * rect.height()

    private fun drawBitmapInto(canvas: Canvas, bmp: Bitmap, rect: RectF) {
        canvas.drawBitmap(bmp, null, rect, bitmapPaint)
    }

    private fun drawDepthLayer(canvas: Canvas, rect: RectF, f: RenderFrame) {
        if (f.hasDepth) drawBitmapInto(canvas, f.depthColor, rect) else canvas.drawRect(rect, fillPaint)
    }

    private fun drawSegLayer(canvas: Canvas, rect: RectF, cam: Bitmap, f: RenderFrame) {
        drawBitmapInto(canvas, cam, rect)
        if (f.instances.isEmpty()) {
            canvas.drawText("Seg: 0 objects detected", rect.left + 16f, rect.top + 64f, textPaint)
            return
        }
        drawBitmapInto(canvas, f.maskOverlay, rect)
        f.instances.forEach { drawSegBox(canvas, rect, it, drawLabel = true) }
    }

    private fun drawSegBox(canvas: Canvas, rect: RectF, inst: SegInstance, drawLabel: Boolean) {
        val size = SegmentationStage.INPUT_SIZE.toFloat()
        val left = mapX(inst.box[0] / size, rect)
        val top = mapY(inst.box[1] / size, rect)
        boxPaint.color = classColor(inst.classId)
        canvas.drawRect(left, top, mapX(inst.box[2] / size, rect), mapY(inst.box[3] / size, rect), boxPaint)
        if (drawLabel) {
            val label = CocoLabels.NAMES.getOrElse(inst.classId) { "?" }
            canvas.drawText("$label ${"%.2f".format(inst.score)}", left, top - 6f, textPaint)
        }
    }

    private fun drawPoseLayer(canvas: Canvas, rect: RectF, cam: Bitmap, f: RenderFrame) {
        drawBitmapInto(canvas, cam, rect)
        if (f.poses.isEmpty()) {
            canvas.drawText("Pose: 0 objects detected", rect.left + 16f, rect.top + 64f, textPaint)
        } else {
            f.poses.forEach { drawPoseAxes(canvas, rect, it) }
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

    private fun drawPipelineLayer(canvas: Canvas, rect: RectF, cam: Bitmap, f: RenderFrame) {
        drawBitmapInto(canvas, cam, rect)
        if (f.instances.isNotEmpty()) {
            drawBitmapInto(canvas, f.maskOverlay, rect)
            f.instances.forEach { drawSegBox(canvas, rect, it, drawLabel = false) }
        }
        f.poses.forEach { drawPoseAxes(canvas, rect, it) }
    }
}
