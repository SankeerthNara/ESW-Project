package com.example.multipipeline

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.multipipeline.camera.CameraManager
import com.example.multipipeline.overlay.OverlayView
import com.example.multipipeline.pipeline.FramePool
import com.example.multipipeline.pipeline.PipelineOrchestrator
import com.example.multipipeline.pipeline.RenderFrame
import com.example.multipipeline.pipeline.ViewMode

class MainActivity : AppCompatActivity() {

    private lateinit var orchestrator: PipelineOrchestrator
    private lateinit var cameraManager: CameraManager
    private lateinit var overlayView: OverlayView
    private lateinit var fpsText: TextView
    private val framePool = FramePool()

    // Analysis-thread only.
    private var currentFrame: RenderFrame? = null
    private var windowStartNs = 0L
    private var windowFrames = 0
    private var windowLatencyMs = 0L

    private val requestPermission = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else fpsText.text = "Camera permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        overlayView = findViewById(R.id.overlayView)
        overlayView.framePool = framePool
        fpsText = findViewById(R.id.fpsText)

        setupModeButtons()

        orchestrator = PipelineOrchestrator(this)
        cameraManager = CameraManager(
            this, this,
            acquireBitmap = { w, h ->
                framePool.acquire().also { currentFrame = it }.cameraBitmap(w, h)
            },
            onFrame = { onCameraFrame() },
        )

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun onCameraFrame() {
        val frame = currentFrame ?: return
        orchestrator.process(frame)
        framePool.publish(frame)
        overlayView.postInvalidate()
        recordFps(frame)
    }

    /** Displayed-frame rate over ~1 s windows - what the user actually sees. */
    private fun recordFps(frame: RenderFrame) {
        val now = System.nanoTime()
        if (windowStartNs == 0L) windowStartNs = now
        windowFrames++
        windowLatencyMs += frame.timings.totalMs
        val elapsedNs = now - windowStartNs
        if (elapsedNs < 1_000_000_000L) return

        val fps = windowFrames * 1e9f / elapsedNs
        val avgMs = windowLatencyMs.toFloat() / windowFrames
        val t = frame.timings
        val text = "%.1f fps | pipeline %.1f ms (NPU depth %d, seg %d) | %d obj"
            .format(fps, avgMs, t.depthNpuMs, t.segNpuMs, frame.instances.size)
        Log.i("ProfilePipeline", text)
        runOnUiThread { fpsText.text = text }
        windowStartNs = now
        windowFrames = 0
        windowLatencyMs = 0
    }

    private fun setupModeButtons() {
        findViewById<Button>(R.id.btnDepth).setOnClickListener { overlayView.setMode(ViewMode.DEPTH) }
        findViewById<Button>(R.id.btnSeg).setOnClickListener { overlayView.setMode(ViewMode.SEGMENTATION) }
        findViewById<Button>(R.id.btnPose).setOnClickListener { overlayView.setMode(ViewMode.POSE) }
        findViewById<Button>(R.id.btnPipeline).setOnClickListener { overlayView.setMode(ViewMode.PIPELINE) }
        findViewById<Button>(R.id.btnAll).setOnClickListener { overlayView.setMode(ViewMode.ALL) }
    }

    private fun startCamera() {
        cameraManager.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        orchestrator.close()
    }
}
