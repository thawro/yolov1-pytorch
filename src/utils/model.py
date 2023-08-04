import torch
from torch import nn
from src.model.backbones import (
    YOLOv1ClassifierBackbone,
    YOLOv1PreHeadBackbone,
    YOLOv1TinyClassifierBackbone,
    YOLOv1TinyPreHeadBackbone,
)
from typing import Literal, Any

from ..logging.pylogger import get_pylogger
import os

log = get_pylogger(__name__)


def save_checkpoint(ckpt: dict[str, dict[str, Any]], path="ckpt.pt"):
    log.info(f"Saving checkpoint to {path}")
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    log.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def create_backbone(mode: Literal["yolov1", "yolov1-tiny"]):
    if mode == "yolov1":
        return YOLOv1ClassifierBackbone(in_channels=3)
    else:
        return YOLOv1TinyClassifierBackbone(in_channels=3)


def create_pre_head(mode: Literal["yolov1", "yolov1-tiny"]):
    if mode == "yolov1":
        return YOLOv1PreHeadBackbone()
    else:
        return YOLOv1TinyPreHeadBackbone()
