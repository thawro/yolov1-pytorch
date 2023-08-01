from src.model.yolov1 import YOLOv1Classifier, create_backbone
from src.data.dataset import create_classification_dataloaders
from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED
from src.utils.utils import train_loop
from src.utils.model import save_checkpoint, load_checkpoint
from src.utils.pylogger import get_pylogger

from torchmetrics.functional import accuracy
import torch.optim as optim
import torch
from torch import nn
from tqdm.auto import tqdm

log = get_pylogger(__name__)
torch.manual_seed(SEED)

BACKBONE_MODE = "yolov1"

DS_NAME = "HWD+"
DS_PATH = str(DATA_PATH / DS_NAME)
LOAD_MODEL = False
EPOCHS = 1000
CKPT_PATH = ROOT / f"checkpoints/classifier/{DS_NAME}/{BACKBONE_MODE}"
CKPT_PATH.mkdir(parents=True, exist_ok=True)
CKPT_PATH = str(CKPT_PATH / "last.pt")


def val_loop(model, dataloader, loss_fn, device="cpu"):
    preds = []
    targets = []
    model.eval()
    for x, y in tqdm(dataloader, desc="Val"):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            out = model(x)
        preds.append(out)
        targets.append(y)
    preds = torch.stack(preds).flatten(0, 1)
    targets = torch.stack(targets).flatten(0, 1)
    loss = loss_fn(preds, targets)
    acc = accuracy(preds, targets, task="multiclass", num_classes=model.num_classes)
    model.train()
    return {"loss": loss, "acc": acc}


def main():
    log.info(f"Using {DEVICE} device")

    train_dl, val_dl = create_classification_dataloaders(
        dataset_path=DS_PATH, 
        batch_size=128, 
        num_workers=8, 
        drop_last=True, 
        pin_memory=True
    )
    num_classes = len(train_dl.dataset.labels)
    backbone = create_backbone(BACKBONE_MODE)
    model = YOLOv1Classifier(num_classes=num_classes, backbone=backbone).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = torch.inf

    if LOAD_MODEL:
        load_checkpoint(CKPT_PATH, model, optimizer)

    for epoch in range(EPOCHS):
        train_loss = train_loop(model, train_dl, optimizer, loss_fn, device=DEVICE)
        val_metrics = val_loop(model, val_dl, loss_fn, device=DEVICE)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        print(
            f"Epoch {epoch}: train/loss: {train_loss:.4f},  val/loss: {val_loss:.4f},  val/acc: {val_acc:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {"model_state_dict": model.backbone.state_dict()}
            save_checkpoint(checkpoint, path=CKPT_PATH)
            log.info("Saved checkpoint")


if __name__ == "__main__":
    main()
