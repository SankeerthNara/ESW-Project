"""
Export Depth Anything V2 (Small) to a static-shape ONNX file ready for the
QNN converter.

Prereqs:
    git clone https://github.com/DepthAnything/Depth-Anything-V2
    cd Depth-Anything-V2 && pip install -r requirements.txt
    # download checkpoints/depth_anything_v2_vits.pth per that repo's README

Run this script from inside the cloned Depth-Anything-V2 repo folder (or add
it to PYTHONPATH) so `from depth_anything_v2.dpt import DepthAnythingV2` works.
"""
import torch
from depth_anything_v2.dpt import DepthAnythingV2

INPUT_SIZE = 392  # static square input (28x14 patches) - keep in sync with DepthStage.kt
CHECKPOINT = "checkpoints/depth_anything_v2_vits.pth"
OUTPUT_ONNX = "depth.onnx"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


def main():
    model = DepthAnythingV2(**MODEL_CONFIG)
    state_dict = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        OUTPUT_ONNX,
        input_names=["image"],
        output_names=["depth"],
        opset_version=17,
        dynamic_axes=None,   # static shape - required for QNN
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Wrote {OUTPUT_ONNX} with static input [1,3,{INPUT_SIZE},{INPUT_SIZE}]")


if __name__ == "__main__":
    main()
