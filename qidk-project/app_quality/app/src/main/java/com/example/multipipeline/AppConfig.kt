package com.example.multipipeline

import android.util.Size

/**
 * The only source file that differs between the fps / balance / quality
 * apps. Model input sizes, tensor layouts and quantization come from the
 * DLCs in assets/ (plus their .json encoding sidecars), not from here.
 *
 * QUALITY: Depth Anything V2 Small @518 + YOLOv8n-seg @640, both W8A16
 * (8-bit weights, 16-bit activations), 720p camera locked at 30 fps.
 */
object AppConfig {
    const val VARIANT = "QUALITY"
    const val CAMERA_FPS = 30
    val CAMERA_RESOLUTION = Size(1280, 720)

    const val SEG_CONF_THRESHOLD = 0.25f
    const val SEG_IOU_THRESHOLD = 0.45f
    const val SEG_MAX_DETECTIONS = 20
}
