package com.example.multipipeline.pipeline

enum class ViewMode { DEPTH, SEGMENTATION, POSE, PIPELINE, ALL }

/** Dense depth map. rawDepth is model output (relative), normalizedDepth is 0-1 for display. */
data class DepthResult(
    val width: Int,
    val height: Int,
    val rawDepth: FloatArray,
    val normalizedDepth: FloatArray,
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

data class FrameResult(
    val depth: DepthResult?,
    val seg: SegResult?,
    val pose: PoseResult?,
    val sourceWidth: Int,
    val sourceHeight: Int,
)
