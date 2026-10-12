package com.example.rs1.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.hardware.camera2.CaptureRequest
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

/**
 * ImageAnalysis-only CameraX setup. OverlayView draws the analyzed frame
 * itself, so there is no Preview use case competing for the camera.
 *
 * Frames arrive as upright RGBA (CameraX rotates them natively), which copy
 * straight into a reused Bitmap - no YUV->Bitmap conversion or per-frame
 * rotation in Kotlin.
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val targetFps: Int,
    private val targetResolution: Size,
    /** Called on the analysis thread; must return a bitmap of the given size to copy the frame into. */
    private val acquireBitmap: (width: Int, height: Int) -> Bitmap,
    /** Called on the analysis thread once the frame has been copied into the bitmap. */
    private val onFrame: (Bitmap) -> Unit,
) {
    companion object {
        private const val TAG = "CameraManager"
        private const val STALL_TIMEOUT_MS = 3000L
    }

    private val analysisExecutor = Executors.newSingleThreadExecutor { Thread(it, "camera-analysis") }
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile private var lastFrameMs = 0L
    private var watchdogRunning = false

    /**
     * CameraX does not always reopen the camera after the camera HAL dies or
     * another client takes the device while this activity stays resumed - the
     * app then shows a frozen frame forever. Rebind whenever frames stop.
     */
    private val watchdog = object : Runnable {
        override fun run() {
            val resumed = lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)
            if (resumed && SystemClock.elapsedRealtime() - lastFrameMs > STALL_TIMEOUT_MS) {
                Log.w(TAG, "No camera frames for ${STALL_TIMEOUT_MS} ms - rebinding camera")
                bind()
            }
            mainHandler.postDelayed(this, STALL_TIMEOUT_MS)
        }
    }

    fun start() {
        bind()
        if (!watchdogRunning) {
            watchdogRunning = true
            mainHandler.postDelayed(watchdog, STALL_TIMEOUT_MS)
        }
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun bind() {
        lastFrameMs = SystemClock.elapsedRealtime() // grace period while the camera opens
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val builder = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(targetResolution, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                        )
                        .build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setOutputImageRotationEnabled(true)

            // Without a fixed range, auto-exposure is free to drop to 10-15 fps
            // indoors, which caps the whole pipeline no matter how fast it is.
            Camera2Interop.Extender(builder)
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(targetFps, targetFps))

            val analysis = builder.build().also {
                it.setAnalyzer(analysisExecutor) { proxy -> handleFrame(proxy) }
            }

            provider.unbindAll()
            val camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            camera.cameraInfo.cameraState.removeObservers(lifecycleOwner)
            camera.cameraInfo.cameraState.observe(lifecycleOwner) { state ->
                Log.i(TAG, "Camera state: ${state.type}${state.error?.let { " error=${it.code}" } ?: ""}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun handleFrame(proxy: ImageProxy) {
        lastFrameMs = SystemClock.elapsedRealtime()
        try {
            val target = proxy.use { copyFrame(it) }
            // The camera buffer is already released here, so the camera can
            // fill the next frame while this one is being processed.
            onFrame(target)
        } catch (t: Throwable) {
            Log.e(TAG, "Frame handling failed: ${t.message}", t)
        }
    }

    private fun copyFrame(proxy: ImageProxy): Bitmap {
        val plane = proxy.planes[0]
        val target = acquireBitmap(proxy.width, proxy.height)
        if (plane.rowStride == proxy.width * 4 && plane.pixelStride == 4) {
            plane.buffer.rewind()
            target.copyPixelsFromBuffer(plane.buffer)
        } else {
            // Padded rows: let CameraX unpack, then copy into the reused bitmap.
            val tmp = proxy.toBitmap()
            Canvas(target).drawBitmap(tmp, 0f, 0f, null)
            tmp.recycle()
        }
        return target
    }

    fun shutdown() {
        mainHandler.removeCallbacks(watchdog)
        analysisExecutor.shutdown()
    }
}
