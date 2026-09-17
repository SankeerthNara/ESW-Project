#!/usr/bin/env bash
# Convert an ONNX model to a Hexagon-NPU-ready quantized .dlc using the QNN SDK.
#
# Usage:
#   ./convert_to_dlc.sh <input.onnx> <name> <width> <height>
# Example:
#   ./convert_to_dlc.sh depth.onnx depth 384 384
#
# Requires: QNN_SDK_ROOT set and QNN SDK tools on PATH (see README.md step 0).

set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 <input.onnx> <name> <width> <height>"
  exit 1
fi

ONNX_IN="$1"
NAME="$2"
W="$3"
H="$4"
CALIB_DIR="calibration/${NAME}"

if [ -z "${QNN_SDK_ROOT:-}" ]; then
  echo "ERROR: QNN_SDK_ROOT is not set. See README.md step 0."
  exit 1
fi

if [ ! -d "$CALIB_DIR" ] || [ -z "$(ls -A "$CALIB_DIR" 2>/dev/null)" ]; then
  echo "ERROR: put 20-100 representative images (resized to ${W}x${H}) in $CALIB_DIR/ first."
  echo "These calibrate the INT8 quantizer so the model runs correctly on the NPU."
  exit 1
fi

mkdir -p build/"$NAME"

echo "== Step 1/3: build calibration input list =="
python3 make_raw_inputs.py --src "$CALIB_DIR" --dst build/"$NAME"/raw --width "$W" --height "$H" \
  --list_out build/"$NAME"/input_list.txt

echo "== Step 2/3: ONNX -> float DLC =="
qnn-onnx-converter \
  --input_network "$ONNX_IN" \
  --input_dim image "1,3,${H},${W}" \
  --out_node "" \
  --output_path build/"$NAME"/"${NAME}"_float.dlc

echo "== Step 3/3: quantize DLC (INT8, HTP-ready) =="
qnn-model-lib-generator -c build/"$NAME"/"${NAME}"_float.dlc \
  --input_list build/"$NAME"/input_list.txt \
  -o "${NAME}.dlc" || \
qnn-dlc-quantize \
  --input_dlc build/"$NAME"/"${NAME}"_float.dlc \
  --input_list build/"$NAME"/input_list.txt \
  --output_dlc "${NAME}.dlc"

echo "Done -> ${NAME}.dlc"
echo "NOTE: exact tool names (qnn-model-lib-generator vs qnn-dlc-quantize vs"
echo "snpe-dlc-quantize) vary between QNN SDK releases. If both commands above"
echo "fail, run 'ls \$QNN_SDK_ROOT/bin/x86_64-linux-clang' and use whichever"
echo "quantize tool is actually present in your installed version."
