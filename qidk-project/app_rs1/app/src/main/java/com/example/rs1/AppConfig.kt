package com.example.rs1

import android.util.Size

/**
 * RS-1 (research 1): the physical-error-budgeted runtime of
 * QIDK_CoDesign_Proposal.pdf. Unlike the fps / balance / quality apps, the
 * models and processors are not fixed here - they are a runtime
 * configuration (research/Configuration.kt); this file holds only what is
 * common to every configuration.
 */
object AppConfig {
    const val VARIANT = "RS-1"
    const val CAMERA_FPS = 30
    val CAMERA_RESOLUTION = Size(640, 480)

    const val SEG_CONF_THRESHOLD = 0.25f
    const val SEG_IOU_THRESHOLD = 0.45f
    const val SEG_MAX_DETECTIONS = 20
}
