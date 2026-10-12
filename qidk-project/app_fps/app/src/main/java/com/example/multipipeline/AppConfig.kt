package com.example.multipipeline

import android.util.Size

/**
 * The only source file that differs between the fps / balance / quality
 * apps. Model input sizes, tensor layouts and quantization come from the
 * DLCs in assets/ (plus their .json encoding sidecars), not from here.
 *
 * FPS: Depth Anything V2 Small @280 + YOLOv8n-seg @384, both INT8 (W8A8),
 * camera locked at 60 fps.
 */
object AppConfig {
    const val VARIANT = "FPS"
    const val CAMERA_FPS = 60
    val CAMERA_RESOLUTION = Size(640, 480)

    const val SEG_CONF_THRESHOLD = 0.25f
    const val SEG_IOU_THRESHOLD = 0.45f
    const val SEG_MAX_DETECTIONS = 20
}
