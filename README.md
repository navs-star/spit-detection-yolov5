# Real-Time Spit Detection System — YOLOv5

A computer vision project that detects spitting in real time using a laptop webcam and displays a "DO NOT SPIT" warning on screen.

## Results
- mAP@50: 88%
- Precision: 84.2%
- Recall: 80.3%
- Training: 3766 images, Google Colab Tesla T4 GPU

## How to Run
```bash
pip install torch torchvision opencv-python ultralytics
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
python 3_live_detect.py --weights best.pt
```

## Tech Stack
- Python 3.10.11
- YOLOv5
- PyTorch
- OpenCV
- Google Colab (training)
- Roboflow (dataset)

## Dataset
- Dataset 1: Roboflow — Manas (528 images)
- Dataset 2: Roboflow — Jaydev Zala (1116 images)
- Merged total: 3766 training images
