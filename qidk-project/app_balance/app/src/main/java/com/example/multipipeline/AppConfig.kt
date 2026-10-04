package com.example.multipipeline

import android.util.Size

/**
 * The only source file that differs between the fps / balance / quality
 * apps. Model input sizes, tensor layouts and quantization come from the
 * DLCs in assets/ (plus their .json encoding sidecars), not from here.
 *
 * BALANCE: Depth Anything V2 Small @392 + YOLOv8n-seg @640, both INT8 (W8A8),
 * camera locked at 30 fps.
 */
object AppConfig {
    const val VARIANT = "BALANCE"
    const val CAMERA_FPS = 30
    val CAMERA_RESOLUTION = Size(640, 480)

    const val SEG_CONF_THRESHOLD = 0.25f
    const val SEG_IOU_THRESHOLD = 0.45f
    const val SEG_MAX_DETECTIONS = 20
}
