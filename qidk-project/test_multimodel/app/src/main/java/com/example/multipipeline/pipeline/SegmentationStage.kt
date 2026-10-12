package com.example.multipipeline.pipeline

import android.content.Context
import android.graphics.Bitmap
import com.example.multipipeline.snpe.QuantEncoding
import com.example.multipipeline.snpe.SnpeModelRunner
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8n-seg, 640x640 INT8 with an SM8650 HTP cache (seg_decoupled_htp.dlc).
 *
 * The detect head is exported "decoupled" so each output gets its own INT8
 * scale (a single concatenated output0 would share one scale across box
 * pixels, class scores and mask coefficients and crush the scores):
 *   images [1,640,640,3]   NHWC, 0-1
 *   boxes  [1,4,8400]      cx, cy, w, h in input pixels
 *   scores [1,80,8400]     sigmoid class scores
 *   coeffs [1,32,8400]     mask coefficients
 *   proto  [1,160,160,32]  mask prototypes, NHWC
 */
class SegmentationStage(
    context: Context,
    private val confThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f,
    private val maxDetections: Int = 20,
) {
    companion object {
        const val INPUT_SIZE = 640
        const val PROTO_SIZE = 160
        const val NUM_ANCHORS = 8400
        const val NUM_CLASSES = 80
        const val NUM_MASK_COEFF = 32

        // From `snpe-dlc-info -i seg_decoupled_htp.dlc`.
        private val ENC_IMAGES = QuantEncoding(1f / 255f, 0)
        private val ENC_BOXES = QuantEncoding(2.506506919861f, 0)
        private val ENC_SCORES = QuantEncoding(0.002667539055f, 0)
        private val ENC_COEFFS = QuantEncoding(0.017189435661f, -152)
        private val ENC_PROTO = QuantEncoding(0.021236443892f, -13)
    }

    private val runner = SnpeModelRunner(
        context, "seg.dlc",
        inputEncodings = mapOf("images" to ENC_IMAGES),
        outputEncodings = mapOf(
            "boxes" to ENC_BOXES, "scores" to ENC_SCORES,
            "coeffs" to ENC_COEFFS, "proto" to ENC_PROTO,
        ),
    )
    private val packer = ModelInputPacker(INPUT_SIZE, ENC_IMAGES, imagenet = false)

    private val boxes = ByteArray(4 * NUM_ANCHORS)
    private val scores = ByteArray(NUM_CLASSES * NUM_ANCHORS)
    private val coeffs = ByteArray(NUM_MASK_COEFF * NUM_ANCHORS)
    private val protoQ = ByteArray(PROTO_SIZE * PROTO_SIZE * NUM_MASK_COEFF)
    private val proto = FloatArray(protoQ.size)
    private val protoLut = FloatArray(256) { ENC_PROTO.dequant(it) }
    private val coeffLut = FloatArray(256) { ENC_COEFFS.dequant(it) }

    private val bestScoreQ = IntArray(NUM_ANCHORS)
    private val bestClass = IntArray(NUM_ANCHORS)
    // Compare scores as raw bytes: q > threshold  <=>  dequant(q) > confThreshold.
    private val scoreThresholdQ = (confThreshold / ENC_SCORES.scale).toInt()

    private class Candidate(val anchor: Int, val classId: Int, val score: Float, val box: FloatArray)

    var lastNpuMs = 0L; private set

    fun run(frame: Bitmap): SegResult {
        packer.pack(frame, runner.inputBuffers.getValue("images"))
        val t0 = System.nanoTime()
        runner.execute()
        lastNpuMs = (System.nanoTime() - t0) / 1_000_000

        read("boxes", boxes); read("scores", scores); read("coeffs", coeffs)

        // Best class per anchor, class-major so reads are sequential.
        java.util.Arrays.fill(bestScoreQ, 0)
        for (c in 0 until NUM_CLASSES) {
            val base = c * NUM_ANCHORS
            for (a in 0 until NUM_ANCHORS) {
                val q = scores[base + a].toInt() and 0xFF
                if (q > bestScoreQ[a]) { bestScoreQ[a] = q; bestClass[a] = c }
            }
        }

        val candidates = ArrayList<Candidate>()
        for (a in 0 until NUM_ANCHORS) {
            val q = bestScoreQ[a]
            if (q <= scoreThresholdQ) continue
            val cx = ENC_BOXES.dequant(boxes[a].toInt() and 0xFF)
            val cy = ENC_BOXES.dequant(boxes[NUM_ANCHORS + a].toInt() and 0xFF)
            val w = ENC_BOXES.dequant(boxes[2 * NUM_ANCHORS + a].toInt() and 0xFF)
            val h = ENC_BOXES.dequant(boxes[3 * NUM_ANCHORS + a].toInt() and 0xFF)
            candidates.add(Candidate(a, bestClass[a], ENC_SCORES.dequant(q),
                floatArrayOf(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)))
        }
        val kept = nms(candidates)
        if (kept.isEmpty()) return SegResult(emptyList())

        // Only dequantize the prototypes when something survived NMS.
        read("proto", protoQ)
        for (i in protoQ.indices) proto[i] = protoLut[protoQ[i].toInt() and 0xFF]

        return SegResult(kept.map { c ->
            SegInstance(c.classId, c.score, c.box, PROTO_SIZE, PROTO_SIZE, buildMask(c))
        })
    }

    private fun read(name: String, dst: ByteArray) {
        val buf = runner.outputBuffers.getValue(name)
        buf.rewind()
        buf.get(dst)
    }

    /** sigmoid(coeffs . proto), cropped to the box, at prototype resolution. */
    private fun buildMask(c: Candidate): FloatArray {
        val k = FloatArray(NUM_MASK_COEFF) { coeffLut[coeffs[it * NUM_ANCHORS + c.anchor].toInt() and 0xFF] }
        val mask = FloatArray(PROTO_SIZE * PROTO_SIZE)
        val scale = PROTO_SIZE.toFloat() / INPUT_SIZE
        val bx1 = (c.box[0] * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val by1 = (c.box[1] * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val bx2 = (c.box[2] * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        val by2 = (c.box[3] * scale).toInt().coerceIn(0, PROTO_SIZE - 1)
        for (y in by1..by2) {
            for (x in bx1..bx2) {
                val p = y * PROTO_SIZE + x
                val base = p * NUM_MASK_COEFF // NHWC: 32 coefficients contiguous per pixel
                var sum = 0f
                for (i in 0 until NUM_MASK_COEFF) sum += k[i] * proto[base + i]
                mask[p] = 1f / (1f + exp(-sum))
            }
        }
        return mask
    }

    private fun nms(cands: List<Candidate>): List<Candidate> {
        val sorted = cands.sortedByDescending { it.score }
        val kept = ArrayList<Candidate>()
        for (c in sorted) {
            if (kept.size >= maxDetections) break
            if (kept.none { it.classId == c.classId && iou(it.box, c.box) > iouThreshold }) kept.add(c)
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
