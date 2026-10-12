package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import com.example.multipipeline.AppConfig
import com.example.multipipeline.snpe.SnpeModelRunner
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8n-seg from assets/seg.dlc, any static square input size. The detect
 * head is exported "decoupled" so each output gets its own quantization
 * scale (a single concatenated output0 would share one scale across box
 * pixels, class scores and mask coefficients and crush the scores):
 *   images [1,S,S,3]          NHWC, 0-1
 *   boxes  [1,4,N]            cx, cy, w, h in input pixels
 *   scores [1,80,N]           sigmoid class scores
 *   coeffs [1,32,N]           mask coefficients
 *   proto  [1,P,P,32] or [1,32,P,P]   mask prototypes, P = S/4
 */
class SegmentationStage(context: Context) {
    companion object {
        const val NUM_MASK_COEFF = 32
    }

    private val runner = SnpeModelRunner(context, "seg.dlc")
    private val packer = ModelInputPacker(runner.inputs.getValue("images"), imagenet = false)
    private val boxesT = runner.outputs.getValue("boxes")
    private val scoresT = runner.outputs.getValue("scores")
    private val coeffsT = runner.outputs.getValue("coeffs")
    private val protoT = runner.outputs.getValue("proto")

    private val inputSize = packer.size.toFloat()
    private val numAnchors = boxesT.shape.last()
    private val numClasses = scoresT.shape[1]
    private val protoNhwc = protoT.shape[3] == NUM_MASK_COEFF
    /** Side of the square prototype / mask grid. */
    val protoSize = if (protoNhwc) protoT.shape[1] else protoT.shape[2]
    private val protoPlane = protoSize * protoSize
    // Index of (pixel p, coefficient k) = p * pixelStride + k * coeffStride
    private val pixelStride = if (protoNhwc) NUM_MASK_COEFF else 1
    private val coeffStride = if (protoNhwc) 1 else protoPlane

    private val boxes = FloatArray(boxesT.size)
    private val scores = FloatArray(scoresT.size)
    private val coeffs = FloatArray(coeffsT.size)
    private val proto = FloatArray(protoT.size)
    private val bestScore = FloatArray(numAnchors)
    private val bestClass = IntArray(numAnchors)

    private class Candidate(val anchor: Int, val classId: Int, val score: Float, val box: FloatArray)

    var lastNpuMs = 0L; private set

    fun run(frame: Bitmap): SegResult {
        packer.pack(frame)
        val t0 = System.nanoTime()
        runner.execute()
        lastNpuMs = (System.nanoTime() - t0) / 1_000_000

        scoresT.readFloats(scores)
        // Best class per anchor, class-major so reads are sequential.
        java.util.Arrays.fill(bestScore, 0f)
        for (c in 0 until numClasses) {
            val base = c * numAnchors
            for (a in 0 until numAnchors) {
                val s = scores[base + a]
                if (s > bestScore[a]) { bestScore[a] = s; bestClass[a] = c }
            }
        }

        boxesT.readFloats(boxes)
        val candidates = ArrayList<Candidate>()
        for (a in 0 until numAnchors) {
            val s = bestScore[a]
            if (s <= AppConfig.SEG_CONF_THRESHOLD) continue
            val cx = boxes[a]; val cy = boxes[numAnchors + a]
            val w = boxes[2 * numAnchors + a]; val h = boxes[3 * numAnchors + a]
            candidates.add(Candidate(a, bestClass[a], s, floatArrayOf(
                (cx - w / 2f) / inputSize, (cy - h / 2f) / inputSize,
                (cx + w / 2f) / inputSize, (cy + h / 2f) / inputSize,
            )))
        }
        val kept = nms(candidates)
        if (kept.isEmpty()) return SegResult(emptyList())

        // Only dequantize coefficients and prototypes when something survived NMS.
        coeffsT.readFloats(coeffs)
        protoT.readFloats(proto)
        return SegResult(kept.map { c ->
            SegInstance(c.classId, c.score, c.box, protoSize, protoSize, buildMask(c))
        })
    }

    /** sigmoid(coeffs . proto), cropped to the box, at prototype resolution. */
    private fun buildMask(c: Candidate): FloatArray {
        val k = FloatArray(NUM_MASK_COEFF) { coeffs[it * numAnchors + c.anchor] }
        val mask = FloatArray(protoPlane)
        val last = protoSize - 1
        val bx1 = (c.box[0] * protoSize).toInt().coerceIn(0, last)
        val by1 = (c.box[1] * protoSize).toInt().coerceIn(0, last)
        val bx2 = (c.box[2] * protoSize).toInt().coerceIn(0, last)
        val by2 = (c.box[3] * protoSize).toInt().coerceIn(0, last)
        for (y in by1..by2) {
            for (x in bx1..bx2) {
                val p = y * protoSize + x
                val base = p * pixelStride
                var sum = 0f
                for (i in 0 until NUM_MASK_COEFF) sum += k[i] * proto[base + i * coeffStride]
                mask[p] = 1f / (1f + exp(-sum))
            }
        }
        return mask
    }

    private fun nms(cands: List<Candidate>): List<Candidate> {
        val sorted = cands.sortedByDescending { it.score }
        val kept = ArrayList<Candidate>()
        for (c in sorted) {
            if (kept.size >= AppConfig.SEG_MAX_DETECTIONS) break
            if (kept.none { it.classId == c.classId && iou(it.box, c.box) > AppConfig.SEG_IOU_THRESHOLD }) kept.add(c)
        }
        return kept
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = iw * ih
        val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() = runner.close()
}
