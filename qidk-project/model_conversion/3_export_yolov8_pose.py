"""
Export a YOLOv8 pose model to static-shape ONNX.

You mentioned a "pinhole" pose model without being sure of the exact
architecture. Since nothing was pinned down, this uses YOLOv8-pose, which:
  - shares the exact same export/quantization toolchain as the seg model above
    (one less thing to debug on-device),
  - is well-tested on Hexagon NPUs,
  - is easy to swap for your own weights if "pinhole" turns out to be a
    custom-trained YOLOv8-pose model (just change WEIGHTS below).

If your real pose model is NOT YOLOv8-based (e.g. RTMPose, OpenPose, a fully
custom network), replace the body of this script with that model's own
PyTorch export code — everything else in this project (convert_to_dlc.sh,
PoseStage.kt) only cares that you end up with a static-shape ONNX file named
pose.onnx; you'll just need to update PoseStage.kt's output parsing to match
your model's actual output tensor layout instead of YOLOv8-pose's.

pip install ultralytics
"""
from ultralytics import YOLO

WEIGHTS = "yolov8n-pose.pt"   # <-- replace with your own weights if applicable
IMG_SIZE = 640                # static square input - keep in sync with PoseStage.kt
OUTPUT_ONNX = "pose.onnx"


def main():
    model = YOLO(WEIGHTS)
    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,
        simplify=True,
        opset=17,
    )
    print(f"Exported. Rename/move the produced .onnx to {OUTPUT_ONNX}")


if __name__ == "__main__":
    main()
