#!/usr/bin/env python3
import json
import time
import qai_hub as hub

jobs = {
    "Pose Estimation (YOLOv8n-Pose DLC)": "j5wl93y6p",
    "Segmentation (YOLOv8n-Seg DLC)": "jg9z4yolp",
    "Depth Estimation (Depth Anything V2 Small DLC)": "jp1n8ez7g"
}

results = {}

for name, jid in jobs.items():
    job = hub.get_job(jid)
    status = job.get_status()
    print(f"[{name}] ID: {jid} | Status: {status.code}")
    results[name] = {
        "job_id": jid,
        "url": job.url,
        "status": status.code,
        "message": status.message
    }
    if status.code == "SUCCESS":
        try:
            prof = job.download_profile()
            exec_sum = prof.get("execution_summary", {})
            inf_time_us = exec_sum.get("estimated_inference_time")
            peak_mem_bytes = exec_sum.get("estimated_inference_peak_memory")
            results[name]["inference_time_ms"] = (inf_time_us / 1000.0) if inf_time_us else None
            results[name]["peak_memory_mb"] = (peak_mem_bytes / (1024 * 1024.0)) if peak_mem_bytes else None
            results[name]["fps"] = (1000.0 / (inf_time_us / 1000.0)) if inf_time_us else None
            results[name]["compute_units"] = prof.get("compute_unit_counts", {})
        except Exception as e:
            results[name]["profile_error"] = str(e)

with open("/work/ai_hub_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to /work/ai_hub_benchmark_results.json")
