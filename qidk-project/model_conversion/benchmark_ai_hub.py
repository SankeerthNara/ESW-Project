#!/usr/bin/env python3
"""
Qualcomm AI Hub Benchmarking Utility
Allows querying status and detailed profiling results for models running on Snapdragon 8 Gen 3 (SM8650).
"""
import sys
import json
import qai_hub as hub

def print_job_summary(job_id: str):
    print(f"\n==================== JOB: {job_id} ====================")
    job = hub.get_job(job_id)
    status = job.get_status()
    print(f"Status  : {status.code}")
    print(f"Message : {status.message}")
    print(f"URL     : {job.url}")

    if status.code == "SUCCESS":
        try:
            profile = job.download_profile()
            print("\n--- Execution Summary ---")
            if "execution_summary" in profile:
                summary = profile["execution_summary"]
                print(f"Estimated Inference Time: {summary.get('estimated_inference_time', 'N/A')} us")
                print(f"Compute Units Breakdown: {summary.get('compute_unit_counts', {})}")
            print(json.dumps(profile, indent=2)[:1000] + "...")
        except Exception as e:
            print(f"Could not parse full profile details: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_ai_hub.py <job_id_1> [job_id_2 ...]")
        sys.exit(1)
    for jid in sys.argv[1:]:
        print_job_summary(jid)
