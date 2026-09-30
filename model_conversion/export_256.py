import torch
from ultralytics import YOLO

model = YOLO('/work/yolov8n-seg.pt')
head = model.model.model[-1]

def decoupled_forward(self, x):
    preds = self.forward_head(x, self.cv2, self.cv3, self.cv4)
    dbox = self._get_decode_boxes(preds)
    scores = preds['scores'].sigmoid()
    mask_coeffs = preds['mask_coefficient']
    proto = self.proto(x[0])
    return dbox, scores, mask_coeffs, proto

head.forward = decoupled_forward.__get__(head, type(head))
model.model.eval()

IMG_SIZE = 128
dummy_in = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(
    model.model,
    dummy_in,
    f'/work/yolov8n_seg_{IMG_SIZE}.onnx',
    opset_version=17,
    input_names=['images'],
    output_names=['boxes', 'scores', 'coeffs', 'proto'],
    dynamo=False
)
print('Exported seg 256x256')
