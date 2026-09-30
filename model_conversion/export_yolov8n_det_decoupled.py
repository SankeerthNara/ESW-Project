import torch
import torch.nn as nn
from ultralytics import YOLO

class DecoupledYOLO(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        
    def forward(self, x):
        preds = self.model(x)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        # preds is [1, 84, 8400]
        # boxes are first 4 rows: cx, cy, w, h
        # scores are next 80 rows
        boxes = preds[:, :4, :]
        scores = preds[:, 4:, :]
        return boxes, scores

base_model = YOLO('yolov8n.pt')
wrapped_model = DecoupledYOLO(base_model)
wrapped_model.eval()

dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    wrapped_model,
    dummy,
    '/work/build/seg/yolov8n_det_decoupled.onnx',
    input_names=['image'],
    output_names=['boxes', 'scores'],
    opset_version=17
)
print('Exported decoupled det onnx')
