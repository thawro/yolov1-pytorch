from torch import nn
from collections import OrderedDict
from .helpers import CNNBlock, Backbone


YOLOV1_CFG = [
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

YOLOV1_TINY_CFG = [
    # channels, kernel_size, stride, maxpool
    # 1
    (16, 3, 1, True),
    # 2
    (32, 3, 1, True),
    # 3
    (64, 3, 1, True),
    # 4
    (128, 3, 1, True),
    # 5
    (256, 3, 1, True),
    # 6
    (512, 3, 1, True),
    # 7    
    (1024, 3, 1, False),
    (256, 3, 1, False),
]


class YOLOv1Backbone(Backbone):
    def __init__(self, config: list[tuple[int, int, int, bool]], in_channels: int = 3):
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
        
class YOLOv1ClassifierBackbone(YOLOv1Backbone):
    def __init__(self, in_channels: int = 3):
        super().__init__(YOLOV1_CFG[:20], in_channels=in_channels)

class YOLOv1PreHeadBackbone(YOLOv1Backbone):
    def __init__(self):
        super().__init__(YOLOV1_CFG[20:], in_channels=YOLOV1_CFG[19][0])
        
        
class YOLOv1TinyClassifierBackbone(YOLOv1Backbone):
    def __init__(self, in_channels: int = 3):
        super().__init__(YOLOV1_TINY_CFG[:7], in_channels=in_channels)

class YOLOv1TinyPreHeadBackbone(YOLOv1Backbone):
    def __init__(self):
        super().__init__(YOLOV1_TINY_CFG[7:], in_channels=YOLOV1_TINY_CFG[6][0])