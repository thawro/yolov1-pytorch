# About
PyTorch implementation of [YOLOv1](https://arxiv.org/pdf/1506.02640.pdf) architecture

# Details
Model training is split into two parts:
1. Train classifier using backbone ([YOLOv1](https://arxiv.org/pdf/1506.02640.pdf) or [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf))
2. Train object detector using classifier backbone