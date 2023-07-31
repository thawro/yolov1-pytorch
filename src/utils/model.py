
import torch
from torch import nn
from .pylogger import get_pylogger

log = get_pylogger(__name__)


def save_checkpoint(ckpt, path="ckpt.pt"):
    print("=> Saving checkpoint")
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path)
    log.info("Loading checkpoint")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])


def load_classifier_checkpoint(path: str, model):
    ckpt = torch.load(path)
    log.info("Loading classifier checkpoint")
    model.backbone.load_state_dict(ckpt["model_state_dict"])