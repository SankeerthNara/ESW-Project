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
    private val numAnchors = 8400 // standard for 640 input across P3/P4/P5 - verify against your export

    fun run(frame: Bitmap): SegResult {
        val resized = ImageUtils.resizeSquare(frame, INPUT_SIZE)
        val input = ImageUtils.bitmapToChwFloat(resized)
        val outputs = runner.run(input)

        val pred = outputs[OUTPUT0_NAME]
            ?: outputs.entries.firstOrNull { it.key.contains("output0", ignoreCase = true) }?.value
            ?: return SegResult(emptyList())
        val proto = outputs[OUTPUT1_NAME]
            ?: outputs.entries.firstOrNull { it.key.contains("output1", ignoreCase = true) }?.value
            ?: return SegResult(emptyList())

        val rowLen = 4 + numClasses + NUM_MASK_COEFF
        val candidates = ArrayList<SegInstance>()

        // pred is laid out [rowLen, numAnchors] (channels-first, as exported by ultralytics)
        for (a in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until numClasses) {
                val score = pred[(4 + c) * numAnchors + a]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }
            if (bestScore < confThreshold) continue

            val cx = pred[0 * numAnchors + a]
            val cy = pred[1 * numAnchors + a]
            val w = pred[2 * numAnchors + a]
            val h = pred[3 * numAnchors + a]
            val x1 = cx - w / 2f
            val y1 = cy - h / 2f
            val x2 = cx + w / 2f
            val y2 = cy + h / 2f

            val coeffs = FloatArray(NUM_MASK_COEFF) { pred[(4 + numClasses + it) * numAnchors + a] }
            val mask = buildMask(coeffs, proto, x1, y1, x2, y2)

            candidates.add(
                SegInstance(
                    classId = bestClass,
                    score = bestScore,
                    box = floatArrayOf(x1, y1, x2, y2),
                    maskWidth = PROTO_SIZE,
                    maskHeight = PROTO_SIZE,
                    mask = mask,
                )
            )
        }

        return SegResult(nms(candidates))
    }

    private fun buildMask(coeffs: FloatArray, proto: FloatArray, x1: Float, y1: Float, x2: Float, y2: Float): FloatArray {
        val planeSize = PROTO_SIZE * PROTO_SIZE
        val mask = FloatArray(planeSize)
        for (p in 0 until planeSize) {
            var sum = 0f
            for (k in 0 until NUM_MASK_COEFF) {
                sum += coeffs[k] * proto[k * planeSize + p]
            }
            mask[p] = sigmoid(sum)
        }
        // Zero out anything outside the box (proto space is INPUT_SIZE/4 = 160, same scale as PROTO_SIZE here)
        val scale = PROTO_SIZE.toFloat() / INPUT_SIZE
        val bx1 = (x1 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val by1 = (y1 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val bx2 = (x2 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val by2 = (y2 * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        for (yy in 0 until PROTO_SIZE) {
            for (xx in 0 until PROTO_SIZE) {
                if (xx < bx1 || xx > bx2 || yy < by1 || yy > by2) {
                    mask[yy * PROTO_SIZE + xx] = 0f
                }
            }
        }
        return mask
    }

    private fun sigmoid(x: Float) = (1f / (1f + exp(-x)))

    private fun nms(instances: List<SegInstance>): List<SegInstance> {
        val sorted = instances.sortedByDescending { it.score }.toMutableList()
        val kept = ArrayList<SegInstance>()
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
