package com.example.rs1.overlay

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.example.rs1.AppConfig
import com.example.rs1.pipeline.CocoLabels
import com.example.rs1.pipeline.FramePool
import com.example.rs1.pipeline.ObjectPose
import com.example.rs1.pipeline.RenderFrame
import com.example.rs1.pipeline.SegInstance
import com.example.rs1.pipeline.ViewMode
import com.example.rs1.pipeline.classColor

/**
 * Owns ALL rendering - the camera frame itself plus every model's output -
 * so the 2x2 "ALL" layout can show the live frame in each quadrant next to
 * the depth colormap, matching the reference pipeline's 4-screen dashboard.
 *
 * Everything heavy (depth colormap, combined instance masks) arrives as
 * prebuilt bitmaps in a [RenderFrame], so onDraw is a few scaled bitmap
 * draws plus lines and text. Each panel shows the frame center-cropped at
 * its true aspect ratio; all overlays are mapped through the same crop.
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
    private val content = RectF()   // frame area of the current panel (may extend past it)
    private val panelRect = RectF()  // visible area of the current panel

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
        canvas.drawRect(full, fillPaint)
        val f = framePool?.latestForDraw()
        val cam = f?.camera
        if (f == null || cam == null) {
            canvas.drawText("Waiting for camera feed...", 32f, height / 2f, textPaint)
            return
        }

        when (mode) {
            ViewMode.DEPTH -> panel(canvas, full, cam) { drawDepthLayer(canvas, f) }
            ViewMode.SEGMENTATION -> panel(canvas, full, cam) { drawSegLayer(canvas, cam, f) }
            ViewMode.POSE -> panel(canvas, full, cam) { drawPoseLayer(canvas, cam, f) }
            ViewMode.PIPELINE -> panel(canvas, full, cam) { drawPipelineLayer(canvas, cam, f) }
            ViewMode.ALL -> {
                panel(canvas, quad[0], cam) { drawDepthLayer(canvas, f) }
                panel(canvas, quad[1], cam) { drawSegLayer(canvas, cam, f) }
                panel(canvas, quad[2], cam) { drawPoseLayer(canvas, cam, f) }
                panel(canvas, quad[3], cam) { drawPipelineLayer(canvas, cam, f) }

                // Labels hug the center divider: the fps readout covers the top
                // edge and the mode buttons cover the bottom one.
                canvas.drawText("DEPTH (DA-V2 Small)", quad[0].left + 16f, quad[0].bottom - 16f, textPaint)
                canvas.drawText("SEGMENTATION", quad[1].left + 16f, quad[1].bottom - 16f, textPaint)
                canvas.drawText("POSE (6-DoF)", quad[2].left + 16f, quad[2].top + 36f, textPaint)
                canvas.drawText("PIPELINE", quad[3].left + 16f, quad[3].top + 36f, textPaint)

                canvas.drawLine(full.centerX(), 0f, full.centerX(), full.bottom, dividerPaint)
                canvas.drawLine(0f, full.centerY(), full.right, full.centerY(), dividerPaint)
            }
        }
        val tag = if (f.configLabel.isEmpty()) AppConfig.VARIANT else "${AppConfig.VARIANT}  ${f.configLabel}"
        canvas.drawText(tag, full.right - textPaint.measureText(tag) - 16f, full.top + 150f, textPaint)
    }

    /** Sets [content] to [frame] center-cropped into [rect] and draws [body] clipped to [rect]. */
    private inline fun panel(canvas: Canvas, rect: RectF, frame: Bitmap, body: () -> Unit) {
        val scale = maxOf(rect.width() / frame.width, rect.height() / frame.height)
        val w = frame.width * scale; val h = frame.height * scale
        panelRect.set(rect)
        content.set(rect.centerX() - w / 2f, rect.centerY() - h / 2f, rect.centerX() + w / 2f, rect.centerY() + h / 2f)
        canvas.save()
        canvas.clipRect(rect)
        body()
        canvas.restore()
    }

    private fun mapX(nx: Float) = content.left + nx * content.width()
    private fun mapY(ny: Float) = content.top + ny * content.height()

    private fun drawBitmap(canvas: Canvas, bmp: Bitmap) {
        canvas.drawBitmap(bmp, null, content, bitmapPaint)
    }

    private fun drawDepthLayer(canvas: Canvas, f: RenderFrame) {
        val depth = f.depthColor
        if (f.hasDepth && depth != null) drawBitmap(canvas, depth)
    }

    private fun drawSegLayer(canvas: Canvas, cam: Bitmap, f: RenderFrame) {
        drawBitmap(canvas, cam)
        if (f.instances.isEmpty()) {
            canvas.drawText("Seg: 0 objects detected", panelRect.left + 16f, panelRect.top + 100f, textPaint)
            return
        }
        f.maskOverlay?.let { drawBitmap(canvas, it) }
        f.instances.forEach { drawSegBox(canvas, it, drawLabel = true) }
    }

    private fun drawSegBox(canvas: Canvas, inst: SegInstance, drawLabel: Boolean) {
        val left = mapX(inst.box[0])
        val top = mapY(inst.box[1])
        boxPaint.color = classColor(inst.classId)
        canvas.drawRect(left, top, mapX(inst.box[2]), mapY(inst.box[3]), boxPaint)
        if (drawLabel) {
            val label = CocoLabels.NAMES.getOrElse(inst.classId) { "?" }
            canvas.drawText("$label ${"%.2f".format(inst.score)}", left, top - 6f, textPaint)
        }
    }

    private fun drawPoseLayer(canvas: Canvas, cam: Bitmap, f: RenderFrame) {
        drawBitmap(canvas, cam)
        if (f.poses.isEmpty()) {
            canvas.drawText("Pose: 0 objects detected", panelRect.left + 16f, panelRect.top + 100f, textPaint)
        } else {
            f.poses.forEach { drawPoseAxes(canvas, it) }
        }
    }

    private fun drawPoseAxes(canvas: Canvas, obj: ObjectPose) {
        val (ox, oy) = obj.originNorm
        val (xx, xy) = obj.xAxisEndNorm
        val (yx, yy) = obj.yAxisEndNorm
        val (zx, zy) = obj.zAxisEndNorm

        canvas.drawLine(mapX(ox), mapY(oy), mapX(xx), mapY(xy), axisXPaint)
        canvas.drawLine(mapX(ox), mapY(oy), mapX(yx), mapY(yy), axisYPaint)
        canvas.drawLine(mapX(ox), mapY(oy), mapX(zx), mapY(zy), axisZPaint)
        canvas.drawCircle(mapX(ox), mapY(oy), 7f, originPaint)

        val posTxt = "${obj.label} [X:%.2f Y:%.2f Z:%.2f m]".format(obj.x, obj.y, obj.z)
        val rotTxt = "Rot [R:%.0f P:%.0f Y:%.0f deg]".format(obj.roll, obj.pitch, obj.yaw)
        val labelX = mapX(obj.boxNorm[0])
        val labelY = mapY(obj.boxNorm[1]) - 30f
        canvas.drawText(posTxt, labelX, labelY, textPaint)
        canvas.drawText(rotTxt, labelX, labelY + 22f, textPaint)
    }

    private fun drawPipelineLayer(canvas: Canvas, cam: Bitmap, f: RenderFrame) {
        drawBitmap(canvas, cam)
        if (f.instances.isNotEmpty()) {
            f.maskOverlay?.let { drawBitmap(canvas, it) }
            f.instances.forEach { drawSegBox(canvas, it, drawLabel = false) }
        }
        f.poses.forEach { drawPoseAxes(canvas, it) }
    }
}
