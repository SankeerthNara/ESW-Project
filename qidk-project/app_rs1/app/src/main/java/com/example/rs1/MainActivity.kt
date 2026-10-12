package com.example.rs1

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.rs1.camera.CameraManager
import com.example.rs1.overlay.OverlayView
import com.example.rs1.pipeline.CocoLabels
import com.example.rs1.pipeline.FramePool
import com.example.rs1.pipeline.PipelineOrchestrator
import com.example.rs1.pipeline.RenderFrame
import com.example.rs1.pipeline.ViewMode
import com.example.rs1.research.Bench
import com.example.rs1.research.ChipMonitor
import com.example.rs1.research.ModelBank
import com.example.rs1.research.PipelineConfig
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        private const val EXTRA_TEST_IMAGE = "test_image"
        private const val EXTRA_BENCH = "bench"           // image file or directory under the app's external files dir
        private const val EXTRA_SWEEP = "sweep"           // default | quick | power
        private const val EXTRA_ITERS = "iters"
        private const val EXTRA_COOLDOWN = "cooldown"     // seconds between configurations
        private const val EXTRA_CONFIG = "config"         // index into PipelineConfig.PRESETS
        private const val TAG_TEST = "StillImageTest"
    }

    private lateinit var bank: ModelBank
    private lateinit var chip: ChipMonitor
    private lateinit var orchestrator: PipelineOrchestrator
    private lateinit var cameraManager: CameraManager
    private lateinit var overlayView: OverlayView
    private lateinit var fpsText: TextView
    private val framePool = FramePool()

    // Touched by both the UI and the analysis thread.
    @Volatile private var presetIndex = 0
    @Volatile private var lastSwitchMs = 0f
    @Volatile private var lastError: String? = null

    // Analysis-thread only.
    private var currentFrame: RenderFrame? = null
    private var windowStartNs = 0L
    private var windowFrames = 0
    private var windowLatencyMs = 0f

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

        // Create the bench input dir as the app, so files adb pushes into it are readable.
        getExternalFilesDir("bench")
        bank = ModelBank(this)
        chip = ChipMonitor(this)
        presetIndex = intent.getIntExtra(EXTRA_CONFIG, 0).coerceIn(0, PipelineConfig.PRESETS.size - 1)
        orchestrator = PipelineOrchestrator(bank, PipelineConfig.PRESETS[presetIndex])

        setupButtons()

        cameraManager = CameraManager(
            this, this,
            targetFps = AppConfig.CAMERA_FPS,
            targetResolution = AppConfig.CAMERA_RESOLUTION,
            acquireBitmap = { w, h ->
                framePool.acquire().also { currentFrame = it }.cameraBitmap(w, h)
            },
            onFrame = { onCameraFrame() },
        )

        val testImage = intent.getStringExtra(EXTRA_TEST_IMAGE)
        val bench = intent.getStringExtra(EXTRA_BENCH)
        if (bench != null) {
            // A dozing board runs the app in the background cgroup on throttled
            // cores; measurements are only meaningful with the screen on.
            setShowWhenLocked(true)
            setTurnScreenOn(true)
            runBench(bench)
        } else if (testImage != null) {
            runStillImage(testImage)
        } else if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * `adb shell am start -n com.example.qidk.rs1/com.example.rs1.MainActivity
     *   --es bench bench [--es sweep default|quick|power] [--ei iters 40] [--ei cooldown 0]`
     * profiles every configuration of the sweep on the images in
     * <external files>/bench and writes CSVs to <external files>/results.
     */
    private fun runBench(path: String) {
        val root = getExternalFilesDir(null)!!
        val target = File(root, path)
        val images = if (target.isDirectory) {
            target.listFiles { f -> f.extension.lowercase() in setOf("jpg", "jpeg", "png") }.orEmpty().sortedBy { it.name }
        } else listOf(target)
        val opt = Bench.Options(
            configs = PipelineConfig.sweep(intent.getStringExtra(EXTRA_SWEEP) ?: "default"),
            iters = intent.getIntExtra(EXTRA_ITERS, 40),
            cooldownS = intent.getIntExtra(EXTRA_COOLDOWN, 0),
        )
        fpsText.text = "Benchmark: ${opt.configs.size} configs on ${images.size} images\nadb logcat -s ${Bench.TAG}"
        Thread({
            val result = runCatching { Bench(orchestrator, chip, File(root, "results")).run(images, opt) }
            result.exceptionOrNull()?.let { Log.e(Bench.TAG, "bench failed: ${it.message}", it) }
            runOnUiThread { fpsText.text = if (result.isSuccess) "Benchmark done - results/ in app files" else "Benchmark failed: ${result.exceptionOrNull()?.message}" }
        }, "bench").start()
    }

    /**
     * QA hook: `adb shell am start -n <pkg>/com.example.rs1.MainActivity
     * --es test_image <name>` runs a still image from the app's external files
     * dir through the exact camera pipeline, and logs the detections.
     */
    private fun runStillImage(name: String) {
        Thread({
            val opts = BitmapFactory.Options().apply { inPreferredConfig = android.graphics.Bitmap.Config.ARGB_8888 }
            val img = BitmapFactory.decodeFile(File(getExternalFilesDir(null), name).path, opts)
            if (img == null) { Log.e(TAG_TEST, "cannot decode $name"); return@Thread }
            repeat(30) {
                val frame = framePool.acquire()
                val target = frame.cameraBitmap(img.width, img.height)
                android.graphics.Canvas(target).drawBitmap(img, 0f, 0f, null)
                currentFrame = frame
                onCameraFrame()
            }
            val f = currentFrame ?: return@Thread
            Log.i(TAG_TEST, "$name [${f.configLabel}]: ${f.instances.size} detections, ${f.poses.size} poses, pipeline %.1f ms".format(f.timings.totalMs))
            f.instances.forEach {
                Log.i(TAG_TEST, "  ${CocoLabels.NAMES.getOrElse(it.classId) { "?" }} %.2f box=[%.2f %.2f %.2f %.2f] maskPx=%d".format(
                    it.score, it.box[0], it.box[1], it.box[2], it.box[3], it.mask.count { p -> p > 0.5f }))
            }
        }, "still-image").start()
    }

    private fun onCameraFrame() {
        val frame = currentFrame ?: return
        try {
            orchestrator.process(frame)
        } catch (e: ModelBank.Unavailable) {
            // Fall back to the first preset; the HUD says why.
            lastError = e.message
            presetIndex = 0
            orchestrator.config = PipelineConfig.PRESETS[0]
            return
        }
        if (frame.timings.switchMs > 1f) lastSwitchMs = frame.timings.switchMs
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
        val avgMs = windowLatencyMs / windowFrames
        val t = frame.timings
        val c = chip.sample()
        val text = buildString {
            append("%.1f fps | pipeline %.1f ms | D %.1f S %.1f pose %.1f | %d obj\n".format(
                fps, avgMs, t.depthExecMs, t.segExecMs, t.poseMs, frame.instances.size))
            append("[${presetIndex + 1}/${PipelineConfig.PRESETS.size}] ${frame.configLabel} | ${c.short()}")
            if (lastSwitchMs > 0f) append(" | load %.0f ms".format(lastSwitchMs))
            lastError?.let { append("\n$it") }
        }
        Log.i("ProfilePipeline", text.replace('\n', ' '))
        runOnUiThread { fpsText.text = text }
        windowStartNs = now
        windowFrames = 0
        windowLatencyMs = 0f
    }

    private fun setupButtons() {
        findViewById<Button>(R.id.btnDepth).setOnClickListener { overlayView.setMode(ViewMode.DEPTH) }
        findViewById<Button>(R.id.btnSeg).setOnClickListener { overlayView.setMode(ViewMode.SEGMENTATION) }
        findViewById<Button>(R.id.btnPose).setOnClickListener { overlayView.setMode(ViewMode.POSE) }
        findViewById<Button>(R.id.btnPipeline).setOnClickListener { overlayView.setMode(ViewMode.PIPELINE) }
        findViewById<Button>(R.id.btnAll).setOnClickListener { overlayView.setMode(ViewMode.ALL) }
        findViewById<Button>(R.id.btnConfig).setOnClickListener {
            presetIndex = (presetIndex + 1) % PipelineConfig.PRESETS.size
            lastError = null
            lastSwitchMs = 0f
            orchestrator.config = PipelineConfig.PRESETS[presetIndex]
            fpsText.text = "switching to ${PipelineConfig.PRESETS[presetIndex].label}..."
        }
    }

    private fun startCamera() {
        cameraManager.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        orchestrator.close()
        bank.close()
    }
}
