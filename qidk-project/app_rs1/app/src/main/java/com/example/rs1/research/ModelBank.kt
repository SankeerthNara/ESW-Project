package com.example.rs1.research

import android.content.Context
import android.util.Log
import com.example.rs1.pipeline.DepthStage
import com.example.rs1.pipeline.SegmentationStage

/**
 * Every network instance the runtime has used, kept loaded so switching
 * configuration costs a map lookup rather than a model load (Agile3D's
 * finding, adopted in proposal §3.4). Instances are created on first use;
 * [loadMs] of each is kept for the overhead measurement.
 *
 * A choice that cannot be built (e.g. an op unsupported on the GPU) is
 * remembered as failed and throws [Unavailable] on every later request.
 */
class ModelBank(private val context: Context) {
    companion object {
        private const val TAG = "ModelBank"
    }

    class Unavailable(what: String, cause: Throwable?) : Exception("$what unavailable: ${cause?.message}", cause)

    private val depth = HashMap<ModelChoice, DepthStage>()
    private val seg = HashMap<ModelChoice, SegmentationStage>()
    private val failed = HashMap<String, Throwable>()

    @Synchronized
    fun depth(c: ModelChoice): DepthStage = depth[c] ?: load("depth ${c.label}/${c.power.short}") { DepthStage(context, c) }.also { depth[c] = it }

    @Synchronized
    fun seg(c: ModelChoice): SegmentationStage = seg[c] ?: load("seg ${c.label}/${c.power.short}") { SegmentationStage(context, c) }.also { seg[c] = it }

    @Synchronized
    fun isLoaded(cfg: PipelineConfig) = depth.containsKey(cfg.depth) && seg.containsKey(cfg.seg)

    private fun <T> load(what: String, make: () -> T): T {
        failed[what]?.let { throw Unavailable(what, it) }
        return try {
            make()
        } catch (t: Throwable) {
            Log.e(TAG, "$what failed to load: ${t.message}", t)
            failed[what] = t
            throw Unavailable(what, t)
        }
    }

    @Synchronized
    fun close() {
        depth.values.forEach { runCatching { it.close() } }
        seg.values.forEach { runCatching { it.close() } }
        depth.clear()
        seg.clear()
    }
}
