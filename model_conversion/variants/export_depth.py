"""Export Depth Anything V2 Small (vits) to ONNX at a static square size (multiple of 14)."""
import sys
import torch
sys.path.append("/work/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

size = int(sys.argv[1])
assert size % 14 == 0, "ViT patch size is 14"
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load("/work/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth", map_location="cpu"))
model.eval()
torch.onnx.export(model, torch.randn(1, 3, size, size), f"/work/variants/build/depth_{size}.onnx",
                  input_names=["image"], output_names=["depth"], opset_version=17,
                  do_constant_folding=True, dynamo=False)
print(f"depth_{size}.onnx")
