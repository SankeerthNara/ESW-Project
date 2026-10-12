package com.example.rs1.research

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import java.io.File

/**
 * The chip state h of proposal Eq. 4, as far as an app can see it: the
 * hottest sensor of each processor (SM8650 thermal zones), Android's
 * thermal status / headroom, and battery current and voltage for a coarse
 * power reading. Unreadable values come back as NaN.
 */
class ChipMonitor(context: Context) {

    data class Sample(
        val cpuC: Float, val gpuC: Float, val npuC: Float, val ddrC: Float, val skinC: Float,
        val thermalStatus: Int,
        val headroom: Float,   // PowerManager.getThermalHeadroom(0); 1.0 = throttling starts
        val currentMa: Float,  // battery current, + = discharging (sign varies by board)
        val voltageV: Float,
    ) {
        fun short() = "CPU %.0f° GPU %.0f° NPU %.0f°".format(cpuC, gpuC, npuC)
    }

    private val app = context.applicationContext
    private val power = app.getSystemService(PowerManager::class.java)
    private val battery = app.getSystemService(BatteryManager::class.java)

    private val zones: Map<String, List<File>> = run {
        val all = File("/sys/class/thermal").listFiles { f -> f.name.startsWith("thermal_zone") }.orEmpty()
        val typed = all.mapNotNull { z -> runCatching { File(z, "type").readText().trim() to File(z, "temp") }.getOrNull() }
        fun group(vararg prefixes: String) = typed.filter { (t, _) -> prefixes.any { t.startsWith(it) } }.map { it.second }
        mapOf(
            "cpu" to group("cpu-", "cpuss-"),
            "gpu" to group("gpuss-"),
            "npu" to group("nsphvx-", "nsphmx-"),
            "ddr" to group("ddr"),
            "skin" to group("skin-msm-therm"),
        )
    }

    private fun maxC(group: String): Float {
        var best = Float.NaN
        for (f in zones[group].orEmpty()) {
            val milli = runCatching { f.readText().trim().toFloat() }.getOrNull() ?: continue
            val c = milli / 1000f
            if (best.isNaN() || c > best) best = c
        }
        return best
    }

    fun sample(): Sample {
        val volt = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
        val currentUa = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        return Sample(
            cpuC = maxC("cpu"), gpuC = maxC("gpu"), npuC = maxC("npu"), ddrC = maxC("ddr"), skinC = maxC("skin"),
            thermalStatus = power.currentThermalStatus,
            headroom = power.getThermalHeadroom(0),
            currentMa = if (currentUa == Int.MIN_VALUE) Float.NaN else currentUa / 1000f,
            voltageV = if (volt > 0) volt / 1000f else Float.NaN,
        )
    }
}
