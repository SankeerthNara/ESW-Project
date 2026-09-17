"""
Export a YOLOv8 segmentation model to static-shape ONNX.

If you already trained your OWN YOLOv8-seg weights, just point WEIGHTS at your
`best.pt` file. Otherwise this grabs the stock pretrained nano-seg model so you
have something to build/test the pipeline against.

pip install ultralytics
"""
from ultralytics import YOLO

WEIGHTS = "yolov8n-seg.pt"   # <-- replace with your own best.pt if you have one
IMG_SIZE = 640               # static square input - keep in sync with SegmentationStage.kt
OUTPUT_ONNX = "seg.onnx"


def main():
    model = YOLO(WEIGHTS)
    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,     # static shape - required for QNN
        simplify=True,
        opset=17,
    )
    # Ultralytics writes <weights-stem>.onnx next to the weights file
    print(f"Exported. Rename/move the produced .onnx to {OUTPUT_ONNX}")


if __name__ == "__main__":
    main()
