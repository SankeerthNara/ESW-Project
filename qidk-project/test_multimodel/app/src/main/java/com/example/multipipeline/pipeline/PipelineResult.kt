package com.example.multipipeline.pipeline

import android.graphics.Bitmap

enum class ViewMode { DEPTH, SEGMENTATION, POSE, PIPELINE, ALL }

/** Dense relative depth, min-max normalized to 0-1 (1 = closest). */
data class DepthResult(
    val width: Int,
    val height: Int,
    val normalizedDepth: FloatArray,
    val normalizedBytes: ByteArray, // same values as 0-255, for colorization
)

data class SegInstance(
    val classId: Int,
    val score: Float,
    val box: FloatArray,        // [x1, y1, x2, y2] in model input space (0..SegmentationStage.INPUT_SIZE)
    val maskWidth: Int,
    val maskHeight: Int,
    val mask: FloatArray,       // per-pixel probability, maskWidth*maskHeight, in proto space
)

data class SegResult(val instances: List<SegInstance>)

/**
 * Geometric 6-DoF pose for one segmented object, computed via pinhole
 * back-projection + PCA orientation from its mask + the depth map - not a
 * neural network output. Mirrors compute_pose_pca()/project_axes_to_image()
 * from the reference pipeline.
 */
data class ObjectPose(
    val classId: Int,
    val label: String,
    // position in pseudo-metric camera space (meters, approximate)
    val x: Float, val y: Float, val z: Float,
    // orientation in degrees
    val roll: Float, val pitch: Float, val yaw: Float,
    // normalized [0,1] image-space points for drawing the 3D axes overlay
    val originNorm: Pair<Float, Float>,
    val xAxisEndNorm: Pair<Float, Float>,
    val yAxisEndNorm: Pair<Float, Float>,
    val zAxisEndNorm: Pair<Float, Float>,
    // normalized [0,1] box, for label placement
    val boxNorm: FloatArray,
)

data class PoseResult(val objects: List<ObjectPose>)

data class StageTimings(
    val totalMs: Long,
    val depthNpuMs: Long,
    val segNpuMs: Long,
)

/**
 * Everything OverlayView needs to draw one processed frame. Instances are
 * pooled (see FramePool) so the bitmaps are allocated once, not per frame:
 * the pipeline fills a free one while the UI draws another.
 */
class RenderFrame {
    var camera: Bitmap? = null
    val depthColor: Bitmap = Bitmap.createBitmap(DepthStage.INPUT_SIZE, DepthStage.INPUT_SIZE, Bitmap.Config.ARGB_8888)
    val maskOverlay: Bitmap = Bitmap.createBitmap(SegmentationStage.PROTO_SIZE, SegmentationStage.PROTO_SIZE, Bitmap.Config.ARGB_8888)
    var hasDepth = false
    var instances: List<SegInstance> = emptyList()
    var poses: List<ObjectPose> = emptyList()
    var timings = StageTimings(0, 0, 0)

    /** Reuses the camera bitmap when the frame size is unchanged. */
    fun cameraBitmap(width: Int, height: Int): Bitmap {
        val cur = camera
        if (cur != null && cur.width == width && cur.height == height) return cur
        cur?.recycle()
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { camera = it }
    }
}

/**
 * Triple buffer between the pipeline thread and the UI thread: one frame
 * on screen, one waiting to be drawn, one being filled.
 */
class FramePool {
    private val frames = List(3) { RenderFrame() }
    private var displayed: RenderFrame? = null
    private var pending: RenderFrame? = null

    @Synchronized
    fun acquire(): RenderFrame = frames.first { it !== displayed && it !== pending }

    /** Pipeline thread: hand a filled frame to the UI. */
    @Synchronized
    fun publish(frame: RenderFrame) { pending = frame }

    /** UI thread: newest frame to draw (and keep drawing until a newer one arrives). */
    @Synchronized
    fun latestForDraw(): RenderFrame? {
        pending?.let { displayed = it; pending = null }
        return displayed
    }
}
