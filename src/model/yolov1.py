from torch import nn
from src.utils.ops import cellboxes_to_boxes, NMS
from .architectures.backbones.yolov1 import BACKBONE_CFG, YOLOv1Backbone
from .architectures.backbones.squeeze_net import SqueezeNetBackbone
from src.model.architectures.helpers import Backbone

import torch
from typing import Literal


class YOLOv1DetectionHead(nn.Module):
    def __init__(self, backbone_out_channels: int, S: int, C: int, B: int):
        super().__init__()
        in_features = backbone_out_channels * S * S
        mid_features = 496
        out_features = S * S * (C + B * 5)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=mid_features),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(mid_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


class YOLOv1ClassificationHead(nn.Module):
    def __init__(self, backbone_out_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_out_channels, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class YOLOv1Detector(nn.Module):
    def __init__(self, S: int, C: int, B: int, backbone: Backbone, head: nn.Module):
        self.S = S
        self.C = C
        self.B = B
        super().__init__()
        self.backbone = backbone
        self.head = head


    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.reshape(-1, self.S, self.S, self.C + self.B * 5)

    def inference(self, x):
        """
        cell_boxes shape: [batch_size, S, S, C + B * 5]
        preds shape: [batch_size, S * S, 6]
        best_class, objectness, x, y, w, h
        """
        cell_boxes = self(x).to("cpu")
        boxes_preds = cellboxes_to_boxes(cell_boxes, self.S, self.C, self.B)
        return boxes_preds

    def perform_nms(
        self,
        boxes_preds: torch.Tensor,
        iou_threshold: float = 0.5,
        objectness_threshold: float = 0.4,
    ):
        all_nms_boxes = []
        for i, boxes in enumerate(boxes_preds):
            nms_boxes = NMS(
                boxes,
                iou_threshold=iou_threshold,
                objectness_threshold=objectness_threshold,
            )
            all_nms_boxes.append(nms_boxes)
        return all_nms_boxes


class YOLOv1Classifier(nn.Module):
    def __init__(self, num_classes: int, backbone: Backbone):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.head = YOLOv1ClassificationHead(self.backbone.out_channels, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def create_backbone(mode: Literal["yolov1", "squeeze_net"]):
    if mode == "yolov1":
            return YOLOv1Backbone(config=BACKBONE_CFG[:20], in_channels=3)
    else:
        return SqueezeNetBackbone(
            in_channels=3, 
            version="squeezenet1_0", 
            load_from_torch=False, 
            pretrained=False, 
            freeze_extractor=False
        )
    
def create_detection_head(S: int, C: int, B: int, mode: Literal["yolov1", "squeeze_net"], backbone_out_channels: int):
    if mode == "yolov1":
        pre_head = Backbone(
            net = YOLOv1Backbone(BACKBONE_CFG[20:], in_channels=backbone_out_channels),
            out_channels=BACKBONE_CFG[-1][0],
            name="yolov1"
        )
        
    else:
        pre_head = Backbone(
            net = nn.Identity(),
            out_channels=backbone_out_channels,
            name="yolov1"
        )
    return nn.Sequential(
        pre_head,
        YOLOv1DetectionHead(pre_head.out_channels, S, C, B)
    )