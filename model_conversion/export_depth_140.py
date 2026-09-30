import sys
import torch
sys.path.append('/work/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load('/work/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
model.eval()

IMG_SIZE = 140
dummy_in = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(
    model,
    dummy_in,
    f'/work/build/depth/depth_{IMG_SIZE}.onnx',
    opset_version=17,
    input_names=['image'],
    output_names=['depth'],
    dynamo=False
)
print('Exported depth 140x140')
