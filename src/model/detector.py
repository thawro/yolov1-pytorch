from torch import nn
from src.utils.ops import cellboxes_to_boxes, NMS
from .helpers import Backbone
import torch
import torch.nn.functional as F


class YOLOv1DetectionHead(nn.Module):
    def __init__(
        self, backbone_out_channels: int, S: int, C: int, B: int, mid_features: int = 2048
    ):
        super().__init__()
        in_features = backbone_out_channels * S * S
        out_features = S * S * (C + B * 5)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=mid_features),
            nn.LeakyReLU(0.1),
            nn.Linear(mid_features, out_features),  # include [0, 1] for xy classes? TODO
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class YOLOv1Detector(nn.Module):
    def __init__(self, S: int, C: int, B: int, backbone: Backbone, pre_head: Backbone):
        self.S = S
        self.C = C
        self.B = B
        super().__init__()
        self.backbone = backbone
        self.pre_head = pre_head
        self.head = YOLOv1DetectionHead(pre_head.out_channels, S, C, B, mid_features=2048)

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        out = self.pre_head(out)
        out = self.head(out)
        # S * S * (C + B * 5) => S x S x (C + B * 5)
        out = out.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # make x and y in range [0, 1]
        out[..., : self.C][..., 1::5] = F.sigmoid(out[..., : self.C][..., 1::5])
        out[..., : self.C][..., 2::5] = F.sigmoid(out[..., : self.C][..., 2::5])
        return out

    def inference(self, x: torch.Tensor):
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
