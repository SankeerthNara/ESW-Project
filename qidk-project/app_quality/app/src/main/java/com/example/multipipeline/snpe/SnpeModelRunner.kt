package com.example.multipipeline.snpe

import android.app.Application
import android.content.Context
import android.util.Log
import com.qualcomm.qti.snpe.NeuralNetwork
import com.qualcomm.qti.snpe.SNPE
import com.qualcomm.qti.snpe.UserBufferTensor
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Affine quantization of a tensor: real = (q + offset) * scale, with q an
 * unsigned [bits]-bit integer. `offset` uses snpe-dlc-info's sign convention.
 */
data class QuantEncoding(val scale: Float, val offset: Int, val bits: Int) {
    val maxQ = (1 shl bits) - 1
    fun dequant(q: Int): Float = (q + offset) * scale
    fun quant(real: Float): Int = (Math.round(real / scale) - offset).coerceIn(0, maxQ)
}

/**
 * A quantized tensor in a direct ByteBuffer, 8- or 16-bit per element.
 * [shape] is the DLC's (NHWC for images) shape.
 */
class QuantTensor(val name: String, val shape: IntArray, val enc: QuantEncoding) {
    val size = shape.fold(1) { acc, d -> acc * d }
    val bytesPerElement = enc.bits / 8
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(size * bytesPerElement).order(ByteOrder.nativeOrder())

    private val lut8 = if (enc.bits == 8) FloatArray(256) { enc.dequant(it) } else null
    private val raw8 = if (enc.bits == 8) ByteArray(size) else null
    private val raw16 = if (enc.bits == 16) ShortArray(size) else null

    /** Dequantizes the whole tensor into [dst] (size >= [size]). */
    fun readFloats(dst: FloatArray) {
        buffer.rewind()
        if (lut8 != null && raw8 != null) {
            buffer.get(raw8)
            for (i in 0 until size) dst[i] = lut8[raw8[i].toInt() and 0xFF]
        } else {
            val r = raw16!!
            buffer.asShortBuffer().get(r)
            val s = enc.scale; val o = enc.offset
            for (i in 0 until size) dst[i] = ((r[i].toInt() and 0xFFFF) + o) * s
        }
    }
}

/**
 * Runs a quantized, HTP-cached DLC on the Hexagon NPU through SNPE's
 * user-buffer API. Every input/output is a preallocated direct buffer in the
 * tensor's native encoding (from `<asset>.json`, written at conversion time
 * by model_conversion/variants/dump_encodings.py), so execute() moves raw
 * bytes only - no per-frame float arrays and no requantization on the CPU.
 */
class SnpeModelRunner(context: Context, private val assetModelName: String) {
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

    val inputs: Map<String, QuantTensor>
    val outputs: Map<String, QuantTensor>
    private val inputBuffers: Map<String, UserBufferTensor>
    private val outputBuffers: Map<String, UserBufferTensor>

    init {
        val json = JSONObject(context.assets.open(assetModelName.removeSuffix(".dlc") + ".json").bufferedReader().use { it.readText() })
        inputs = tensors(network.inputTensorsShapes, json.getJSONObject("inputs"))
        outputs = tensors(network.outputTensorsShapes, json.getJSONObject("outputs"))
        inputBuffers = inputs.mapValues { userBuffer(it.value) }
        outputBuffers = outputs.mapValues { userBuffer(it.value) }
        Log.i(TAG, "[$assetModelName] runtime=${network.runtime} inputs=${describe(inputs)} outputs=${describe(outputs)}")
    }

    private fun tensors(shapes: Map<String, IntArray>, encodings: JSONObject) = shapes.mapValues { (name, shape) ->
        val e = encodings.optJSONObject(name) ?: error("[$assetModelName] no encoding for tensor '$name' in sidecar json")
        QuantTensor(name, shape, QuantEncoding(e.getDouble("scale").toFloat(), e.getInt("offset"), e.getInt("bits")))
    }

    private fun userBuffer(t: QuantTensor): UserBufferTensor {
        // Packed strides in bytes, innermost dimension = one element.
        val strides = IntArray(t.shape.size)
        var stride = t.bytesPerElement
        for (i in t.shape.indices.reversed()) {
            strides[i] = stride
            stride *= t.shape[i]
        }
        // SNPE's stepExactly0 is the quantized value representing real 0,
        // i.e. the negated snpe-dlc-info offset.
        val bytes = t.size * t.bytesPerElement
        return when (t.enc.bits) {
            8 -> network.createTF8UserBufferTensor(bytes, strides, -t.enc.offset, t.enc.scale, t.buffer)
            16 -> network.createTF16UserBufferTensor(bytes, strides, -t.enc.offset, t.enc.scale, t.buffer)
            else -> error("[$assetModelName] unsupported bitwidth ${t.enc.bits} for '${t.name}'")
        }
    }

    /** Executes on the NPU; read results from [outputs] afterwards. */
    fun execute() {
        if (!network.execute(inputBuffers, outputBuffers)) error("[$assetModelName] SNPE execute() failed")
    }

    fun close() {
        inputBuffers.values.forEach { runCatching { it.release() } }
        outputBuffers.values.forEach { runCatching { it.release() } }
        network.release()
    }

    private fun describe(t: Map<String, QuantTensor>) =
        t.values.joinToString { "${it.name}[${it.shape.joinToString("x")}]u${it.enc.bits}" }

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
