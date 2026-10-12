package com.example.rs1.snpe

import android.app.Application
import android.content.Context
import android.util.Log
import com.example.rs1.research.ModelChoice
import com.example.rs1.research.Proc
import com.qualcomm.qti.snpe.FloatTensor
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
 * bits == 32 means plain float32 (GPU / CPU runtimes).
 */
data class QuantEncoding(val scale: Float, val offset: Int, val bits: Int) {
    val isFloat get() = bits == 32
    val maxQ = if (isFloat) 0 else (1 shl bits) - 1
    fun dequant(q: Int): Float = (q + offset) * scale
    fun quant(real: Float): Int = (Math.round(real / scale) - offset).coerceIn(0, maxQ)
}

/**
 * A tensor in a direct ByteBuffer: quantized 8/16-bit, or float32.
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
        if (enc.isFloat) {
            buffer.asFloatBuffer().get(dst, 0, size)
        } else if (lut8 != null && raw8 != null) {
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
 * Runs one DLC on the processor in [choice].
 *
 * NPU: the quantized HTP graph through SNPE's user-buffer API. Every
 * input/output is a preallocated direct buffer in the tensor's native
 * encoding (from `<asset>.json`, written at conversion time by
 * model_conversion/variants/dump_encodings.py), so execute() moves raw
 * bytes only.
 *
 * GPU / CPU: the same DLC in floating point. SNPE's Java API has no float
 * user buffers, so tensors are float32 direct buffers copied through
 * FloatTensors around execute().
 */
class SnpeModelRunner(context: Context, private val assetModelName: String, val choice: ModelChoice) {
    companion object {
        private const val TAG = "SnpeModelRunner"
    }

    private val npu = choice.proc == Proc.NPU
    private val modelFile: File = copyAssetToFile(context, assetModelName)

    /** Wall time to build the network (model load + graph init on the processor). */
    val loadMs: Float

    private val network: NeuralNetwork

    init {
        val t0 = System.nanoTime()
        val builder = SNPE.NeuralNetworkBuilder(context.applicationContext as Application)
            .setDebugEnabled(false)
            .setPerformanceProfile(choice.power.profile)
            .setExecutionPriorityHint(NeuralNetwork.ExecutionPriorityHint.HIGH)
            .setModel(modelFile)
            .setRuntimeOrder(choice.proc.runtime)
            // No silent fallback: a config that cannot run where it says must fail.
            .setCpuFallbackEnabled(false)
            .setUseUserSuppliedBuffers(npu)
        if (!npu) {
            // Without this, SNPE exposes only the DLC's last output (seg: proto)
            // on GPU/CPU; the HTP cache got the full list at graph-prepare time.
            builder.setUnconsumedTensorsOutput(true)
        }
        if (npu) {
            // Qualcomm's shipped skel libs are unsigned; without both of these the
            // DSP load fails with "dlerror signature verify start failed".
            builder.setUnsignedPD(true).setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK)
        }
        network = builder.build()
        loadMs = (System.nanoTime() - t0) / 1e6f
    }

    val inputs: Map<String, QuantTensor>
    val outputs: Map<String, QuantTensor>
    private val inputBuffers: Map<String, UserBufferTensor>
    private val outputBuffers: Map<String, UserBufferTensor>
    private val floatInputs: Map<String, FloatTensor>
    private val scratch: FloatArray

    init {
        if (npu) {
            val json = JSONObject(context.assets.open(assetModelName.removeSuffix(".dlc") + ".json").bufferedReader().use { it.readText() })
            inputs = tensors(network.inputTensorsShapes, json.getJSONObject("inputs"))
            outputs = tensors(network.outputTensorsShapes, json.getJSONObject("outputs"))
            inputBuffers = inputs.mapValues { userBuffer(it.value) }
            outputBuffers = outputs.mapValues { userBuffer(it.value) }
            floatInputs = emptyMap()
        } else {
            val f32 = QuantEncoding(1f, 0, 32)
            inputs = network.inputTensorsShapes.mapValues { (name, shape) -> QuantTensor(name, shape, f32) }
            outputs = network.outputTensorsShapes.mapValues { (name, shape) -> QuantTensor(name, shape, f32) }
            inputBuffers = emptyMap()
            outputBuffers = emptyMap()
            floatInputs = inputs.mapValues { network.createFloatTensor(*it.value.shape) }
        }
        scratch = FloatArray(if (npu) 0 else (inputs.values + outputs.values).maxOf { it.size })
        Log.i(TAG, "[$assetModelName] ${choice.label} runtime=${network.runtime} load=%.0f ms inputs=${describe(inputs)} outputs=${describe(outputs)}".format(loadMs))
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

    /** Executes on the chosen processor; read results from [outputs] afterwards. */
    fun execute() {
        if (npu) {
            if (!network.execute(inputBuffers, outputBuffers)) error("[$assetModelName] SNPE execute() failed")
            return
        }
        for ((name, t) in inputs) {
            t.buffer.rewind()
            t.buffer.asFloatBuffer().get(scratch, 0, t.size)
            floatInputs.getValue(name).write(scratch, 0, t.size)
        }
        val result = network.execute(floatInputs)
        try {
            for ((name, ft) in result) {
                val t = outputs[name] ?: continue
                ft.read(scratch, 0, t.size)
                t.buffer.rewind()
                t.buffer.asFloatBuffer().put(scratch, 0, t.size)
            }
        } finally {
            result.values.forEach { runCatching { it.release() } }
        }
    }

    fun describe() = "$assetModelName@${network.runtime}#${System.identityHashCode(network).toString(16)}"

    fun close() {
        inputBuffers.values.forEach { runCatching { it.release() } }
        outputBuffers.values.forEach { runCatching { it.release() } }
        floatInputs.values.forEach { runCatching { it.release() } }
        network.release()
    }

    private fun describe(t: Map<String, QuantTensor>) =
        t.values.joinToString { "${it.name}[${it.shape.joinToString("x")}]${if (it.enc.isFloat) "f32" else "u${it.enc.bits}"}" }

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
