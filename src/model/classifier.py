from torch import nn
from src.model.helpers import Backbone


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


class YOLOv1Classifier(nn.Module):
    def __init__(self, num_classes: int, backbone: Backbone):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.head = YOLOv1ClassificationHead(self.backbone.out_channels, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)