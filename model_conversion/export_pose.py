from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")
model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17)
