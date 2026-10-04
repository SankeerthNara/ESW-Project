"""Writes <model>.json next to a DLC with its input/output quantization encodings.

The Android apps read this sidecar to create SNPE user buffers in each
tensor's native encoding, so encodings never have to be copied by hand.
Usage: python dump_encodings.py out/balance_seg.dlc   (QAIRT env sourced)
"""
import json, re, subprocess, sys

dlc = sys.argv[1]
info = subprocess.run(["snpe-dlc-info", "-i", dlc], capture_output=True, text=True, check=True).stdout
row = re.compile(r"^\|\s*(\S+)\s*\|\s*([\d,]+)\s*\|\s*uFxp_(\d+)\s*\|\s*bitwidth \d+, min \S+, max \S+, scale ([\d.eE+-]+), offset ([\d.eE+-]+)")

result, section = {"inputs": {}, "outputs": {}}, None
for line in info.splitlines():
    if line.startswith("| Input Name"):
        section = "inputs"
    elif line.startswith("| Output Name"):
        section = "outputs"
    elif section and (m := row.match(line)):
        name, dims, bits, scale, offset = m.groups()
        result[section][name] = {
            "shape": [int(d) for d in dims.split(",")],
            "bits": int(bits), "scale": float(scale), "offset": int(float(offset)),
        }
for sec in ("inputs", "outputs"):
    if not result[sec]:
        sys.exit(f"no quantized {sec} found in {dlc} - is it a quantized DLC?")

out = dlc[:-4] + ".json"
with open(out, "w") as fh:
    json.dump(result, fh, indent=2)
print(out, json.dumps(result))
