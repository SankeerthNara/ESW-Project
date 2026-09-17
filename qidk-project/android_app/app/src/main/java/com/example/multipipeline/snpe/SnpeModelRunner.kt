package com.example.multipipeline.snpe

import android.app.Application
import android.content.Context
import android.util.Log
import com.qualcomm.qti.snpe.NeuralNetwork
import com.qualcomm.qti.snpe.SNPE
import java.io.File
import java.io.FileOutputStream

class SnpeModelRunner(
    context: Context,
    private val assetModelName: String
) {
    companion object {
        private const val TAG = "SnpeModelRunner"
    }

    private val appContext = context.applicationContext as Application
    private val modelFile: File = copyAssetToFile(context, assetModelName)

    private val network: NeuralNetwork = run {
        Log.i(TAG, "[$assetModelName] Initializing SNPE Builder...")

        val probeBuilder = SNPE.NeuralNetworkBuilder(appContext)
            .setDebugEnabled(false)
            .setPerformanceProfile(NeuralNetwork.PerformanceProfile.BURST)
            .setModel(modelFile)
            .setUnsignedPD(true)
            .setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK)
        // Without this, isRuntimeSupported() defaults to NORMAL_CHECK, which
        // expects a Signed PD - and Qualcomm's shipped skel libs are unsigned
        // by default, so that check (and the later real DSP load) fails with
        // "dlerror signature verify start failed" even with setUnsignedPD(true).

        val dspSupported = try {
            probeBuilder.isRuntimeSupported(NeuralNetwork.Runtime.DSP)
        } catch (t: Throwable) {
            Log.e(TAG, "[$assetModelName] DSP runtime probe failed: ${t.message}", t)
            false
        }
        Log.i(TAG, "[$assetModelName] Runtime Support -> DSP/HTP (unsigned PD check): $dspSupported")

        if (dspSupported) {
            try {
                val net = SNPE.NeuralNetworkBuilder(appContext)
                    .setDebugEnabled(false)
                    .setPerformanceProfile(NeuralNetwork.PerformanceProfile.BURST)
                    .setModel(modelFile)
                    .setUnsignedPD(true)
                    .setRuntimeCheckOption(NeuralNetwork.RuntimeCheckOption.UNSIGNEDPD_CHECK)
                    .setRuntimeOrder(NeuralNetwork.Runtime.DSP, NeuralNetwork.Runtime.CPU)
                    .build()
                Log.i(TAG, "[$assetModelName] Built with runtime order: DSP (unsigned PD) -> CPU")
                return@run net
            } catch (t: Throwable) {
                Log.e(TAG, "[$assetModelName] DSP build failed, falling back to CPU: ${t.message}", t)
            }
        }

        val net = SNPE.NeuralNetworkBuilder(appContext)
            .setDebugEnabled(false)
            .setPerformanceProfile(NeuralNetwork.PerformanceProfile.BURST)
            .setModel(modelFile)
            .setRuntimeOrder(NeuralNetwork.Runtime.CPU)
            .build()
        Log.i(TAG, "[$assetModelName] Built with runtime order: CPU only")
        net
    }

    val inputName: String = network.inputTensorsNames.first()
    val inputShape: IntArray = network.inputTensorsShapes[inputName]!!

    init {
        Log.i(TAG, "[$assetModelName] Input: $inputName, Shape: ${inputShape.joinToString("x")}")
        Log.i(TAG, "[$assetModelName] Outputs: ${network.outputTensorsNames.joinToString(", ")}")
    }

    fun run(inputChw: FloatArray): Map<String, FloatArray> {
        val inputTensor = network.createFloatTensor(*inputShape)
        inputTensor.write(inputChw, 0, inputChw.size)
        val inputs = mapOf(inputName to inputTensor)
        val outputs = network.execute(inputs)
        val result = HashMap<String, FloatArray>()
        for ((name, tensor) in outputs) {
            val arr = FloatArray(tensor.size)
            tensor.read(arr, 0, arr.size)
            result[name] = arr
        }
        inputTensor.release()
        outputs.values.forEach { it.release() }
        return result
    }

    fun close() { network.release() }

    private fun copyAssetToFile(context: Context, assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        var assetLength = -1L
        try {
            context.assets.openFd(assetName).use { fd ->
                assetLength = fd.length
            }
        } catch (_: Throwable) {}

        if (outFile.exists() && outFile.length() > 0L) {
            if (assetLength <= 0L || outFile.length() == assetLength) {
                Log.i(TAG, "Asset $assetName already present and up to date (${outFile.length() / (1024 * 1024)} MB)")
                return outFile
            } else {
                Log.i(TAG, "Asset $assetName size changed (cached=${outFile.length()} vs asset=$assetLength), re-extracting...")
            }
        }
        Log.i(TAG, "Extracting $assetName to internal storage...")
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output -> input.copyTo(output) }
        }
        Log.i(TAG, "Successfully extracted $assetName (${outFile.length() / (1024 * 1024)} MB)")
        return outFile
    }
}