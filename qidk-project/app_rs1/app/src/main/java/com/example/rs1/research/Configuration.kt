package com.example.rs1.research

import com.qualcomm.qti.snpe.NeuralNetwork

/**
 * The knobs of the proposal's configuration c (Table 1), as far as RS-1
 * implements them so far: model size/resolution/precision (bundled as a
 * [Variant]), processor per network ([Proc]), parallel vs. sequential
 * placement, and the SNPE performance profile (the clock / NPU power-mode
 * knob).
 */

/** One converted model pair; assets are `<asset>_depth.dlc` and `<asset>_seg.dlc`. */
enum class Variant(val asset: String, val short: String) {
    FPS("fps", "S"),         // depth 280² W8A8, seg 384² W8A8
    BALANCE("balance", "M"), // depth 392² W8A8, seg 640² W8A8
    QUALITY("quality", "L"), // depth 518² W8A8, seg 640² W8A16
}

/**
 * Where a network runs. The NPU executes the quantized HTP graph; GPU and CPU
 * run the same DLC in floating point (GPU_FLOAT16 = fp16 math).
 */
enum class Proc(val runtime: NeuralNetwork.Runtime, val short: String) {
    NPU(NeuralNetwork.Runtime.DSP, "N"),
    GPU(NeuralNetwork.Runtime.GPU_FLOAT16, "G"),
    CPU(NeuralNetwork.Runtime.CPU, "C"),
}

enum class PowerMode(val profile: NeuralNetwork.PerformanceProfile, val short: String) {
    BURST(NeuralNetwork.PerformanceProfile.BURST, "burst"),
    HIGH(NeuralNetwork.PerformanceProfile.HIGH_PERFORMANCE, "high"),
    BALANCED(NeuralNetwork.PerformanceProfile.BALANCED, "bal"),
    SAVER(NeuralNetwork.PerformanceProfile.POWER_SAVER, "saver"),
}

/** One network instance: which variant, on which processor, at which power mode. */
data class ModelChoice(val variant: Variant, val proc: Proc, val power: PowerMode = PowerMode.BURST) {
    val label get() = "${variant.short}@${proc.short}"
}

data class PipelineConfig(
    val depth: ModelChoice,
    val seg: ModelChoice,
    val parallel: Boolean = true,
) {
    /** e.g. `D:L@N S:M@G par burst` */
    val label: String
        get() {
            val power = if (depth.power == seg.power) depth.power.short else "${depth.power.short}/${seg.power.short}"
            return "D:${depth.label} S:${seg.label} ${if (parallel) "par" else "seq"} $power"
        }

    companion object {
        fun uniform(v: Variant, depthProc: Proc, segProc: Proc, parallel: Boolean = true, power: PowerMode = PowerMode.BURST) =
            PipelineConfig(ModelChoice(v, depthProc, power), ModelChoice(v, segProc, power), parallel)

        /** Configurations the live "Cfg" button cycles through. */
        val PRESETS = listOf(
            uniform(Variant.FPS, Proc.NPU, Proc.NPU),
            uniform(Variant.BALANCE, Proc.NPU, Proc.NPU),
            uniform(Variant.QUALITY, Proc.NPU, Proc.NPU),
            uniform(Variant.QUALITY, Proc.NPU, Proc.NPU, parallel = false),
            uniform(Variant.BALANCE, Proc.GPU, Proc.NPU),
            uniform(Variant.QUALITY, Proc.NPU, Proc.GPU),
            uniform(Variant.QUALITY, Proc.GPU, Proc.GPU),
        )

        /**
         * Configuration the bench uses as its quality reference: the largest
         * models in fp16 on the GPU, the closest thing to the float model on
         * the board until ground-truth depth is available.
         */
        val REFERENCE = uniform(Variant.QUALITY, Proc.GPU, Proc.GPU)

        /**
         * Bench sweeps. "default": every variant x {NPU, GPU} per network x
         * {parallel, sequential}, plus CPU for the smallest variant only.
         * "quick": the presets. "power": NPU-only presets across power modes.
         */
        fun sweep(name: String): List<PipelineConfig> = when {
            // "presets:0,6" -> those PRESETS indices
            name.startsWith("presets:") -> name.removePrefix("presets:").split(',').map { PRESETS[it.trim().toInt()] }
            name == "quick" -> PRESETS
            name == "power" -> Variant.values().flatMap { v ->
                PowerMode.values().map { p -> uniform(v, Proc.NPU, Proc.NPU, power = p) }
            }
            else -> buildList {
                val procs = listOf(Proc.NPU, Proc.GPU)
                for (v in Variant.values()) for (d in procs) for (s in procs) for (par in listOf(true, false)) {
                    add(uniform(v, d, s, par))
                }
                add(uniform(Variant.FPS, Proc.CPU, Proc.CPU))
                add(uniform(Variant.FPS, Proc.CPU, Proc.NPU))
                // Mixed sizes: cheap segmentation with accurate depth and vice versa.
                add(PipelineConfig(ModelChoice(Variant.QUALITY, Proc.NPU), ModelChoice(Variant.FPS, Proc.NPU)))
                add(PipelineConfig(ModelChoice(Variant.FPS, Proc.NPU), ModelChoice(Variant.QUALITY, Proc.NPU)))
                add(PipelineConfig(ModelChoice(Variant.QUALITY, Proc.GPU), ModelChoice(Variant.FPS, Proc.NPU)))
            }
        }
    }
}
