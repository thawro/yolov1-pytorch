# About
PyTorch implementation of [YOLOv1](https://arxiv.org/pdf/1506.02640.pdf) architecture

# Details
Model training is split into two parts:
1. Train classifier using backbone (`yolov1` or `yolov1-tiny`)
2. Train object detector using classifier backbone
