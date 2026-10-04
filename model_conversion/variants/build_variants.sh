#!/bin/bash
# Builds the depth/seg DLCs for the fps / balance / quality apps.
# Outputs out/<name>.dlc plus out/<name>.json (encodings the apps read).
# Per-channel weights + 32-bit bias; per-row quantization was measured ~15%
# slower on the HTP for little gain, so it is not used.
# NOTE: --set_output_tensors is required for seg; graph-prepare otherwise
# compiles the HTP cache with only the last layer's output (proto).
# Run inside the QAIRT container:  bash /work/variants/build_variants.sh
source /root/qenv.sh   # before strict mode: envsetup.sh trips over set -eu
set -euo pipefail
cd /work/variants
mkdir -p build out
CALIB="/work/calibration/depth /work/calibration/seg /work/calibration/pose /work/Depth-Anything-V2/assets/examples /opt/qnn_venv/lib/python3.10/site-packages/ultralytics/assets"

# name  onnx  input  size  normalize  act_bits  outputs
build() {
  local name=$1 onnx=$2 input=$3 size=$4 norm=$5 bits=$6 outputs=$7
  echo "===== $name"
  python make_raw_nhwc.py --src $CALIB --dst build/raw_${norm}_${size} --size $size \
      --normalize $norm --input_name $input --list_out build/list_${name}.txt
  snpe-onnx-to-dlc --input_network $onnx -d $input 1,3,$size,$size -o build/${name}_float.dlc > build/${name}_convert.log 2>&1
  snpe-dlc-quant --input_dlc build/${name}_float.dlc --input_list build/list_${name}.txt \
      --output_dlc build/${name}_q.dlc --use_per_channel_quantization \
      --act_bitwidth $bits --bias_bitwidth 32 > build/${name}_quant.log 2>&1
  snpe-dlc-graph-prepare --input_dlc build/${name}_q.dlc --output_dlc out/${name}.dlc --htp_socs sm8650 \
      --set_output_tensors $outputs > build/${name}_prepare.log 2>&1
  python dump_encodings.py out/${name}.dlc
  ls -la out/${name}.dlc
}

[ -f build/depth_280.onnx ] || python export_depth.py 280
[ -f build/depth_392.onnx ] || python export_depth.py 392
[ -f build/depth_518.onnx ] || python export_depth.py 518

build fps_depth     build/depth_280.onnx               image  280 imagenet 8 depth
build fps_seg       /work/yolov8n_seg_384.onnx         images 384 unit     8 boxes,scores,coeffs,proto
build balance_depth build/depth_392.onnx               image  392 imagenet 8 depth
build balance_seg   /work/yolov8n_seg_decoupled.onnx   images 640 unit     8 boxes,scores,coeffs,proto
build quality_depth build/depth_518.onnx               image  518 imagenet 8 depth
build quality_seg   /work/yolov8n_seg_decoupled.onnx   images 640 unit     16 boxes,scores,coeffs,proto
echo ALL_DONE
