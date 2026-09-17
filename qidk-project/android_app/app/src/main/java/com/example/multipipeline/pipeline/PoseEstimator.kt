package com.example.multipipeline.pipeline

import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Geometric 6-DoF pose estimation: pinhole back-projection of each
 * segmented object's mask into 3D using the depth map, then PCA to find its
 * dominant orientation. This is NOT a neural network - it's a direct port of
 * compute_pose_pca() / project_axes_to_image() from the reference pipeline.
 *
 * Camera intrinsics follow the reference pipeline's convention exactly:
 * fx = fy = frame width, cx = width/2, cy = height/2. Because of that choice,
 * all the pixel-space math collapses to simple normalized-coordinate math
 * (see comments below) - no need to carry actual pixel dimensions around.
 */
class PoseEstimator {

    companion object {
        // Pseudo-metric depth range the normalized [0,1] depth output is
        // mapped into, matching the reference pipeline's 0.3-2.5m assumption.
        const val DEPTH_MIN_M = 0.3f
        const val DEPTH_MAX_M = 2.5f
        const val AXIS_LENGTH_M = 0.15f
        const val MAX_SAMPLE_POINTS = 400
        const val MIN_MASK_PIXELS = 15
        const val MIN_VALID_POINTS = 5
    }

    fun run(seg: SegResult, depth: DepthResult, sourceWidth: Int, sourceHeight: Int): PoseResult {
        val aspect = sourceHeight.toFloat() / sourceWidth.toFloat()
        val objects = ArrayList<ObjectPose>()

        for (inst in seg.instances) {
            val pose = estimateOne(inst, depth, aspect) ?: continue
            objects.add(pose)
        }
        return PoseResult(objects)
    }

    private fun estimateOne(inst: SegInstance, depth: DepthResult, aspect: Float): ObjectPose? {
        val mw = inst.maskWidth
        val mh = inst.maskHeight

        // Collect "on" mask pixel coordinates (normalized 0-1 in image space)
        val onPixels = ArrayList<FloatArray>() // [nx, ny]
        for (y in 0 until mh) {
            for (x in 0 until mw) {
                if (inst.mask[y * mw + x] > 0.5f) {
                    onPixels.add(floatArrayOf((x + 0.5f) / mw, (y + 0.5f) / mh))
                }
            }
        }
        if (onPixels.size < MIN_MASK_PIXELS) return null

        val step = maxOf(1, onPixels.size / MAX_SAMPLE_POINTS)
        val points3d = ArrayList<DoubleArray>() // [x, y, z] in pseudo-meters

        var i = 0
        while (i < onPixels.size) {
            val (nx, ny) = onPixels[i]
            val dx = (nx * depth.width).toInt().coerceIn(0, depth.width - 1)
            val dy = (ny * depth.height).toInt().coerceIn(0, depth.height - 1)
            val zNorm = depth.normalizedDepth[dy * depth.width + dx]
            // Depth Anything V2 outputs disparity (1 = closest, 0 = farthest).
            // Invert to map to distance in meters: 1 -> DEPTH_MIN_M, 0 -> DEPTH_MAX_M
            val z = DEPTH_MIN_M + (1f - zNorm) * (DEPTH_MAX_M - DEPTH_MIN_M)
            if (z > 0f) {
                // Pinhole back-projection collapses to this simple form because
                // fx = fy = width in the reference pipeline's intrinsics.
                val x = (nx - 0.5) * z
                val y = (ny - 0.5) * z * aspect
                points3d.add(doubleArrayOf(x.toDouble(), y, z.toDouble()))
            }
            i += step
        }
        if (points3d.size < MIN_VALID_POINTS) return null

        // Centroid
        var cx = 0.0; var cy = 0.0; var cz = 0.0
        for (p in points3d) { cx += p[0]; cy += p[1]; cz += p[2] }
        val n = points3d.size
        cx /= n; cy /= n; cz /= n

        // Covariance matrix
        var xx = 0.0; var xy = 0.0; var xz = 0.0
        var yy = 0.0; var yz = 0.0; var zz = 0.0
        for (p in points3d) {
            val dx0 = p[0] - cx; val dy0 = p[1] - cy; val dz0 = p[2] - cz
            xx += dx0 * dx0; xy += dx0 * dy0; xz += dx0 * dz0
            yy += dy0 * dy0; yz += dy0 * dz0; zz += dz0 * dz0
        }
        xx /= n; xy /= n; xz /= n; yy /= n; yz /= n; zz /= n

        val cov = arrayOf(
            doubleArrayOf(xx, xy, xz),
            doubleArrayOf(xy, yy, yz),
            doubleArrayOf(xz, yz, zz),
        )
        val (eigenvalues, eigenvectors) = jacobiEigenSymmetric3x3(cov)
        // eigenvectors columns sorted ascending by eigenvalue, matching numpy.linalg.eigh
        var normal = doubleArrayOf(eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0])
        if (normal[2] > 0) normal = normal.map { -it }.toDoubleArray()
        val majorAxis = doubleArrayOf(eigenvectors[0][2], eigenvectors[1][2], eigenvectors[2][2])

        val zAxis = normalize(normal)
        var xAxis = normalize(majorAxis)
        var yAxis = normalize(cross(zAxis, xAxis))
        xAxis = normalize(cross(yAxis, zAxis))

        // R columns = [xAxis, yAxis, zAxis]
        val r00 = xAxis[0]; val r01 = yAxis[0]; val r02 = zAxis[0]
        val r10 = xAxis[1]; val r11 = yAxis[1]; val r12 = zAxis[1]
        val r20 = xAxis[2]; val r21 = yAxis[2]; val r22 = zAxis[2]

        val sy = sqrt(r00 * r00 + r10 * r10)
        val roll: Double; val pitch: Double; val yaw: Double
        if (sy >= 1e-6) {
            roll = atan2(r21, r22)
            pitch = atan2(-r20, sy)
            yaw = atan2(r10, r00)
        } else {
            roll = atan2(-r12, r11)
            pitch = atan2(-r20, sy)
            yaw = 0.0
        }

        val centroid = doubleArrayOf(cx, cy, cz)
        val origin = project(centroid, aspect) ?: return null
        val xEnd = project(addScaled(centroid, xAxis, AXIS_LENGTH_M.toDouble()), aspect) ?: origin
        val yEnd = project(addScaled(centroid, yAxis, AXIS_LENGTH_M.toDouble()), aspect) ?: origin
        val zEnd = project(addScaled(centroid, zAxis, AXIS_LENGTH_M.toDouble()), aspect) ?: origin

        val boxScale = 1f / SegmentationStage.INPUT_SIZE.toFloat()
        val boxNorm = floatArrayOf(
            inst.box[0] * boxScale, inst.box[1] * boxScale,
            inst.box[2] * boxScale, inst.box[3] * boxScale,
        )

        return ObjectPose(
            classId = inst.classId,
            label = CocoLabels.NAMES.getOrElse(inst.classId) { "obj${inst.classId}" },
            x = cx.toFloat(), y = cy.toFloat(), z = cz.toFloat(),
            roll = Math.toDegrees(roll).toFloat(),
            pitch = Math.toDegrees(pitch).toFloat(),
            yaw = Math.toDegrees(yaw).toFloat(),
            originNorm = origin,
            xAxisEndNorm = xEnd,
            yAxisEndNorm = yEnd,
            zAxisEndNorm = zEnd,
            boxNorm = boxNorm,
        )
    }

    /** Projects a 3D pseudo-metric point to normalized [0,1] image coordinates. */
    private fun project(pt: DoubleArray, aspect: Float): Pair<Float, Float>? {
        val z = pt[2]
        if (z <= 0.05) return null
        val nu = (pt[0] / z) + 0.5
        val nv = (pt[1] / z) / aspect + 0.5
        return nu.toFloat() to nv.toFloat()
    }

    private fun addScaled(base: DoubleArray, dir: DoubleArray, scale: Double) =
        doubleArrayOf(base[0] + dir[0] * scale, base[1] + dir[1] * scale, base[2] + dir[2] * scale)

    private fun normalize(v: DoubleArray): DoubleArray {
        val len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).takeIf { it > 1e-12 } ?: 1.0
        return doubleArrayOf(v[0] / len, v[1] / len, v[2] / len)
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

    /**
     * Classic cyclic Jacobi eigenvalue algorithm for a symmetric 3x3 matrix.
     * Returns (eigenvalues ascending, eigenvectors as columns) matching the
     * convention of numpy.linalg.eigh, which the reference pipeline relies on.
     */
    private fun jacobiEigenSymmetric3x3(input: Array<DoubleArray>): Pair<DoubleArray, Array<DoubleArray>> {
        val a = Array(3) { r -> DoubleArray(3) { c -> input[r][c] } }
        val v = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0),
        )

        repeat(50) {
            // Find largest off-diagonal element
            var p = 0; var q = 1; var maxVal = kotlin.math.abs(a[0][1])
            if (kotlin.math.abs(a[0][2]) > maxVal) { maxVal = kotlin.math.abs(a[0][2]); p = 0; q = 2 }
            if (kotlin.math.abs(a[1][2]) > maxVal) { maxVal = kotlin.math.abs(a[1][2]); p = 1; q = 2 }
            if (maxVal < 1e-12) return@repeat

            val theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            val t = (if (theta >= 0) 1.0 else -1.0) / (kotlin.math.abs(theta) + sqrt(theta * theta + 1.0))
            val c = 1.0 / sqrt(t * t + 1.0)
            val s = t * c

            val app = a[p][p]; val aqq = a[q][q]; val apq = a[p][q]
            a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
            a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
            a[p][q] = 0.0; a[q][p] = 0.0

            for (k in 0 until 3) {
                if (k != p && k != q) {
                    val akp = a[k][p]; val akq = a[k][q]
                    a[k][p] = c * akp - s * akq; a[p][k] = a[k][p]
                    a[k][q] = s * akp + c * akq; a[q][k] = a[k][q]
                }
                val vkp = v[k][p]; val vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }

        val eigenvalues = doubleArrayOf(a[0][0], a[1][1], a[2][2])
        // Sort ascending, reordering eigenvector columns to match
        val order = (0..2).sortedBy { eigenvalues[it] }
        val sortedValues = DoubleArray(3) { eigenvalues[order[it]] }
        val sortedVectors = Array(3) { r -> DoubleArray(3) { c -> v[r][order[c]] } }
        return sortedValues to sortedVectors
    }
}
