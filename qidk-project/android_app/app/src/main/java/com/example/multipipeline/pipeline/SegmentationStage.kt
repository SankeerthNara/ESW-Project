package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import com.example.multipipeline.snpe.SnpeModelRunner
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8n-seg. Static input 640x640 - must match
 * model_conversion/2_export_yolov8_seg.py's IMG_SIZE.
 *
 * Ultralytics ONNX export for -seg models produces two outputs:
 *   "output0": [1, 4 + numClasses + 32, 8400]   (box xywh + class scores + 32 mask coeffs)
 *   "output1": [1, 32, 160, 160]                (mask prototypes)
 * Exact tensor names can vary by opset/export settings - check with Netron
 * (netron.app) on your actual seg.onnx and adjust OUTPUT0_NAME/OUTPUT1_NAME
 * below if they differ.
 */
class SegmentationStage(
    context: Context,
    private val numClasses: Int = 80,          // change if you trained a custom class set
    private val confThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f,
) {
    companion object {
        const val INPUT_SIZE = 640
        const val PROTO_SIZE = 160
        const val NUM_MASK_COEFF = 32
        const val OUTPUT0_NAME = "output0"
        const val OUTPUT1_NAME = "output1"
    }

    private val runner = SnpeModelRunner(context, "seg.dlc")
    private val numAnchors = 8400
    private val bestScores = FloatArray(numAnchors)
    private val bestClasses = IntArray(numAnchors)

    private data class RawCandidate(
        val classId: Int,
        val score: Float,
        val box: FloatArray,
        val coeffs: FloatArray,
    )

    fun run(frame: Bitmap): SegResult {
        val t0 = System.currentTimeMillis()
        val resized = ImageUtils.resizeSquare(frame, INPUT_SIZE)
        val t1 = System.currentTimeMillis()
        val input = ImageUtils.bitmapToHwcFloat(resized)
        val t2 = System.currentTimeMillis()
        val outputs = runner.run(input)
        val t3 = System.currentTimeMillis()

        val decoupledScores = outputs["scores"] ?: outputs.entries.firstOrNull { it.key.contains("scores", ignoreCase = true) }?.value
        val decoupledBoxes = outputs["boxes"] ?: outputs.entries.firstOrNull { it.key.contains("boxes", ignoreCase = true) }?.value
        val decoupledCoeffs = outputs["coeffs"] ?: outputs.entries.firstOrNull { it.key.contains("coeffs", ignoreCase = true) }?.value
        val proto = outputs["proto"]
            ?: outputs[OUTPUT1_NAME]
            ?: outputs.entries.firstOrNull { it.key.contains("proto", ignoreCase = true) || it.key.contains("output1", ignoreCase = true) }?.value
            ?: return SegResult(emptyList())

        val rawCandidates = ArrayList<RawCandidate>()
        var topScore = 0f
        var topClass = -1

        if (decoupledScores != null && decoupledBoxes != null && decoupledCoeffs != null) {
            // Decoupled heads branch (INT8 high-precision scale per head)
            bestScores.fill(0f)
            for (c in 0 until numClasses) {
                val offset = c * numAnchors
                for (a in 0 until numAnchors) {
                    val score = decoupledScores[offset + a]
                    if (score > bestScores[a]) {
                        bestScores[a] = score
                        bestClasses[a] = c
                    }
                }
            }

            for (a in 0 until numAnchors) {
                val score = bestScores[a]
                if (score > topScore) {
                    topScore = score
                    topClass = bestClasses[a]
                }
                if (score < confThreshold) continue

                val cx = decoupledBoxes[0 * numAnchors + a]
                val cy = decoupledBoxes[1 * numAnchors + a]
                val w = decoupledBoxes[2 * numAnchors + a]
                val h = decoupledBoxes[3 * numAnchors + a]
                val x1 = cx - w / 2f
                val y1 = cy - h / 2f
                val x2 = cx + w / 2f
                val y2 = cy + h / 2f

                val coeffs = FloatArray(NUM_MASK_COEFF) { k -> decoupledCoeffs[k * numAnchors + a] }
                rawCandidates.add(RawCandidate(bestClasses[a], score, floatArrayOf(x1, y1, x2, y2), coeffs))
            }
            android.util.Log.i("SegmentationStage", "[Decoupled INT8] scores size: ${decoupledScores.size}, proto size: ${proto.size}, Top score: ${"%.4f".format(topScore)} (class $topClass), candidates: ${rawCandidates.size}")
        } else {
            // Legacy concatenated output0 branch
            val pred = outputs[OUTPUT0_NAME]
                ?: outputs.entries.firstOrNull { it.key.contains("output0", ignoreCase = true) }?.value
                ?: return SegResult(emptyList())

            bestScores.fill(0f)
            for (c in 0 until numClasses) {
                val offset = (4 + c) * numAnchors
                for (a in 0 until numAnchors) {
                    val score = pred[offset + a]
                    if (score > bestScores[a]) {
                        bestScores[a] = score
                        bestClasses[a] = c
                    }
                }
            }

            val cxRow = 0 * numAnchors
            val cyRow = 1 * numAnchors
            val wRow = 2 * numAnchors
            val hRow = 3 * numAnchors
            val coeffBase = (4 + numClasses) * numAnchors

            for (a in 0 until numAnchors) {
                val score = bestScores[a]
                if (score > topScore) {
                    topScore = score
                    topClass = bestClasses[a]
                }
                if (score < confThreshold) continue

                val cx = pred[cxRow + a]
                val cy = pred[cyRow + a]
                val w = pred[wRow + a]
                val h = pred[hRow + a]
                val x1 = cx - w / 2f
                val y1 = cy - h / 2f
                val x2 = cx + w / 2f
                val y2 = cy + h / 2f

                val coeffs = FloatArray(NUM_MASK_COEFF) { k -> pred[coeffBase + k * numAnchors + a] }
                rawCandidates.add(RawCandidate(bestClasses[a], score, floatArrayOf(x1, y1, x2, y2), coeffs))
            }
            android.util.Log.i("SegmentationStage", "[Legacy FP16] pred size: ${pred.size}, proto size: ${proto.size}, Top score: ${"%.4f".format(topScore)} (class $topClass), candidates: ${rawCandidates.size}")
        }

        // Run NMS on bounding boxes first to avoid expensive mask computations on suppressed candidates
        val kept = nmsCandidates(rawCandidates)

        // Build masks only for surviving detections
        val instances = kept.map { c ->
            val mask = buildMask(c.coeffs, proto, c.box[0], c.box[1], c.box[2], c.box[3])
            SegInstance(
                classId = c.classId,
                score = c.score,
                box = c.box,
                maskWidth = PROTO_SIZE,
                maskHeight = PROTO_SIZE,
                mask = mask,
            )
        }

        val t4 = System.currentTimeMillis()
        android.util.Log.i("ProfileSeg", "resize=${t1-t0}ms float=${t2-t1}ms npu=${t3-t2}ms post=${t4-t3}ms total=${t4-t0}ms")

        return SegResult(instances)
    }

    private fun buildMask(coeffs: FloatArray, proto: FloatArray, x1: Float, y1: Float, x2: Float, y2: Float): FloatArray {
        val planeSize = PROTO_SIZE * PROTO_SIZE
        val mask = FloatArray(planeSize)

        // Zero out anything outside the box (proto space is INPUT_SIZE/4 = 160, same scale as PROTO_SIZE here)
        val scale = PROTO_SIZE.toFloat() / INPUT_SIZE
        val bx1 = (x1 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val by1 = (y1 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val bx2 = (x2 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val by2 = (y2 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)

        val isNhwc = proto.size == planeSize * NUM_MASK_COEFF
        for (yy in by1..by2) {
            val yOffset = yy * PROTO_SIZE
            for (xx in bx1..bx2) {
                val p = yOffset + xx
                var sum = 0f
                if (isNhwc) {
                    val protoOffset = p * NUM_MASK_COEFF
                    for (k in 0 until NUM_MASK_COEFF) {
                        sum += coeffs[k] * proto[protoOffset + k]
                    }
                } else {
                    for (k in 0 until NUM_MASK_COEFF) {
                        sum += coeffs[k] * proto[k * planeSize + p]
                    }
                }
                mask[p] = sigmoid(sum)
            }
        }
        return mask
    }

    private fun sigmoid(x: Float) = (1f / (1f + exp(-x)))

    private fun nmsCandidates(instances: List<RawCandidate>): List<RawCandidate> {
        val sorted = instances.sortedByDescending { it.score }.toMutableList()
        val kept = ArrayList<RawCandidate>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(it.box, best.box) > iouThreshold && it.classId == best.classId }
        }
        return kept
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ix1 = max(a[0], b[0]); val iy1 = max(a[1], b[1])
        val ix2 = min(a[2], b[2]); val iy2 = min(a[3], b[3])
        val iw = max(0f, ix2 - ix1); val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() = runner.close()
}
