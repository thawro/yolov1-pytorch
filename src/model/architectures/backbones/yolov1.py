from torch import nn
from collections import OrderedDict
from src.model.architectures.helpers import CNNBlock, Backbone


BACKBONE_CFG = [
    # channels, kernel_size, stride, maxpool
    # 1
    (64, 7, 2, True),
    # 2
    (192, 3, 1, True),
    # 3
    (128, 1, 1, False),
    (256, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, True),
    # 4
    (256, 1, 1, False),
    (512, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, False),
    (256, 1, 1, False),
    (512, 3, 1, False),
    (512, 1, 1, False),
    (1024, 3, 1, True),
    # 5
    (512, 1, 1, False),
    (1024, 3, 1, False),
    (512, 1, 1, False),
    (1024, 3, 1, False),
    (1024, 3, 1, False),
    (1024, 3, 2, False),
    # 6
    (1024, 3, 1, False),
    (1024, 3, 1, False),
]


class YOLOv1Backbone(Backbone):
    def __init__(self, config, in_channels: int = 3):
        layers = []
        batch_norm = True
        for i, layer_cfg in enumerate(config):
            out_channels, kernel_size, stride, maxpool = layer_cfg
            layers.append(
                (
                    f"layer_{i}",
                    CNNBlock(in_channels, out_channels, kernel_size, stride, batch_norm, maxpool),
                )
            )
            in_channels = out_channels
        net = nn.Sequential(OrderedDict(layers))
        super().__init__(net=net, out_channels=config[-1][0], name="yolov1_backbone")