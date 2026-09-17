package com.example.multipipeline

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.multipipeline.camera.CameraManager
import com.example.multipipeline.overlay.OverlayView
import com.example.multipipeline.pipeline.PipelineOrchestrator
import com.example.multipipeline.pipeline.ViewMode

class MainActivity : AppCompatActivity() {

    private lateinit var orchestrator: PipelineOrchestrator
    private lateinit var cameraManager: CameraManager
    private lateinit var overlayView: OverlayView
    private lateinit var fpsText: TextView

    private val requestPermission = registerForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else fpsText.text = "Camera permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        overlayView = findViewById(R.id.overlayView)
        fpsText = findViewById(R.id.fpsText)
        val previewView: PreviewView = findViewById(R.id.previewView)

        setupModeButtons()

        orchestrator = PipelineOrchestrator(this)
        cameraManager = CameraManager(this, this, previewView) { bitmap ->
            val t0 = System.currentTimeMillis()
            val result = orchestrator.process(bitmap)
            val t1 = System.currentTimeMillis()

            runOnUiThread {
                overlayView.update(result, bitmap)
                val fps = if (t1 > t0) 1000f / (t1 - t0) else 0f
                fpsText.text = "pipeline: ${t1 - t0}ms (~${"%.1f".format(fps)} fps)"
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
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
        orchestrator.close()
    }
}
