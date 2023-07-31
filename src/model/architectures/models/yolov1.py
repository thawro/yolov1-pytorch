from src.model.architectures.backbones.yolov1 import BACKBONE_CFG, YOLOv1Backbone
from src.model.yolov1 import YOLOv1DetectionHead
from torch import nn

def create_yolov1_backbone():
    return YOLOv1Backbone(config=BACKBONE_CFG[:20], in_channels=3)

def create_detection_head(S: int, C: int, B: int):
    return nn.Sequential(
            YOLOv1Backbone(config=BACKBONE_CFG[20:], in_channels=BACKBONE_CFG[19][0]),
            YOLOv1DetectionHead(BACKBONE_CFG[-1][0], S, C, B)
        )

