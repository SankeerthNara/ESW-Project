package com.example.multipipeline.snpe

import android.app.Application
import android.content.Context
import android.util.Log
import com.qualcomm.qti.snpe.NeuralNetwork
import com.qualcomm.qti.snpe.SNPE
import com.qualcomm.qti.snpe.TF8UserBufferTensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 8-bit affine quantization of a tensor: real = (q + offset) * scale.
 * `offset` uses the same sign convention snpe-dlc-info prints (e.g. -114).
 */
data class QuantEncoding(val scale: Float, val offset: Int) {
    fun dequant(q: Int): Float = (q + offset) * scale
    fun quant(real: Float): Int = (Math.round(real / scale) - offset).coerceIn(0, 255)
}

/**
 * Runs a quantized, HTP-cached DLC on the Hexagon NPU through SNPE's TF8
 * user-buffer API. Inputs and outputs live in preallocated direct ByteBuffers
 * whose encodings match the DLC's native uint8 encodings, so execute() moves
 * raw bytes only: no per-frame Java float arrays, no float<->uint8
 * conversion on the CPU, and no tensor allocation per call.
 *
 * The FloatTensor API this replaces quantized every input float and
 * dequantized every output on a JNI thread, which cost ~15 ms per model per
 * frame on top of the ~6 ms (depth) / ~1.5 ms (seg) the HTP actually needs.
 *
 * Encodings must match `snpe-dlc-info -i <model>.dlc` ("Encoding Info"
 * column). A mismatch is still correct - SNPE requantizes - just slower.
 */
class SnpeModelRunner(
    context: Context,
    private val assetModelName: String,
    inputEncodings: Map<String, QuantEncoding>,
    outputEncodings: Map<String, QuantEncoding>,
) {
    companion object {
        private const val TAG = "SnpeModelRunner"
    }

    private val modelFile: File = copyAssetToFile(context, assetModelName)

    private val network: NeuralNetwork = SNPE.NeuralNetworkBuilder(context.applicationContext as Application)
        .setDebugEnabled(false)
        .setPerformanceProfile(NeuralNetwork.PerformanceProfile.BURST)
        .setExecutionPriorityHint(NeuralNetwork.ExecutionPriorityHint.HIGH)
        .setModel(modelFile)
        // Qualcomm's shipped skel libs are unsigned; without both of these the
        // DSP load fails with "dlerror signature verify start failed".
        .setUnsignedPD(true)
        .setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK)
        .setRuntimeOrder(NeuralNetwork.Runtime.DSP)
        .setUseUserSuppliedBuffers(true)
        .build()

    /** Direct buffers holding raw uint8 tensor data, keyed by tensor name. */
    val inputBuffers: Map<String, ByteBuffer>
    val outputBuffers: Map<String, ByteBuffer>
    val inputShapes: Map<String, IntArray> = network.inputTensorsShapes
    val outputShapes: Map<String, IntArray> = network.outputTensorsShapes

    private val inputTensors: Map<String, TF8UserBufferTensor>
    private val outputTensors: Map<String, TF8UserBufferTensor>

    init {
        Log.i(TAG, "[$assetModelName] runtime=${network.runtime} inputs=${describe(inputShapes)} outputs=${describe(outputShapes)}")
        val (inBufs, inTensors) = createBuffers(inputShapes, inputEncodings)
        val (outBufs, outTensors) = createBuffers(outputShapes, outputEncodings)
        inputBuffers = inBufs; inputTensors = inTensors
        outputBuffers = outBufs; outputTensors = outTensors
    }

    private fun createBuffers(
        shapes: Map<String, IntArray>,
        encodings: Map<String, QuantEncoding>,
    ): Pair<Map<String, ByteBuffer>, Map<String, TF8UserBufferTensor>> {
        val buffers = HashMap<String, ByteBuffer>()
        val tensors = HashMap<String, TF8UserBufferTensor>()
        for ((name, shape) in shapes) {
            val enc = encodings[name] ?: error("[$assetModelName] no encoding configured for tensor '$name'")
            val size = shape.fold(1) { acc, d -> acc * d }
            val buffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
            // Packed strides in bytes, innermost dimension = 1 byte.
            val strides = IntArray(shape.size)
            var stride = 1
            for (i in shape.indices.reversed()) {
                strides[i] = stride
                stride *= shape[i]
            }
            // SNPE's stepExactly0 is the quantized value representing real 0,
            // i.e. the negated offset printed by snpe-dlc-info.
            tensors[name] = network.createTF8UserBufferTensor(size, strides, -enc.offset, enc.scale, buffer)
            buffers[name] = buffer
        }
        return buffers to tensors
    }

    /** Executes on the NPU; read results from [outputBuffers] afterwards. */
    fun execute() {
        if (!network.execute(inputTensors, outputTensors)) {
            error("[$assetModelName] SNPE execute() failed")
        }
    }

    fun close() {
        inputTensors.values.forEach { runCatching { it.release() } }
        outputTensors.values.forEach { runCatching { it.release() } }
        network.release()
    }

    private fun describe(shapes: Map<String, IntArray>) =
        shapes.entries.joinToString { "${it.key}[${it.value.joinToString("x")}]" }

    private fun copyAssetToFile(context: Context, assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        // Re-extract whenever the APK was (re)installed after the cached copy
        // was written - a size check alone misses same-size model swaps.
        val apkUpdated = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        if (outFile.exists() && outFile.length() > 0L && outFile.lastModified() >= apkUpdated) {
            return outFile
        }
        Log.i(TAG, "Extracting $assetName to internal storage...")
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output -> input.copyTo(output) }
        }
        return outFile
    }
}
