from src.model.classifier import YOLOv1Classifier
from src.data.dataset import create_classification_datasets, parse_datasets_to_dataloaders
from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED
from src.utils.utils import train_loop, merge_dicts, display_metrics
from src.utils.model import save_checkpoint, load_checkpoint, create_backbone
from src.logging.pylogger import get_pylogger

from torchmetrics.functional import accuracy
import torch.optim as optim
import torch.utils.data
import torch
from torch import nn
from tqdm.auto import tqdm

log = get_pylogger(__name__)
torch.manual_seed(SEED)

MODE = "yolov1-tiny"

DS_NAME = "HWD+"
DS_PATH = str(DATA_PATH / DS_NAME)
LOAD_MODEL = True
EPOCHS = 100
CKPT_PATH = ROOT / f"checkpoints/classifier/{DS_NAME}/{MODE}"
CKPT_PATH.mkdir(parents=True, exist_ok=True)
BACKBONE_CKPT_PATH = str(CKPT_PATH / "backbone_best.pt")
CLASSIFIER_CKPT_PATH = str(CKPT_PATH / "classifier_best.pt")

DL_PARAMS = dict(batch_size=32, num_workers=8, drop_last=True, pin_memory=True)
LIMIT_BATCHES = 1

LOOP_PARAMS = dict(device=DEVICE, limit_batches=LIMIT_BATCHES)


def val_loop(
    model: YOLOv1Classifier,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss._Loss,
    device="cpu",
    limit_batches: int = -1,
) -> dict[str, float]:
    preds = []
    targets = []
    model.eval()
    for x, y in tqdm(dataloader, desc="Val"):
        if limit_batches == 0:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)
        preds.append(out)
        targets.append(y)
        limit_batches -= 1
    preds = torch.stack(preds).flatten(0, 1)
    targets = torch.stack(targets).flatten(0, 1)
    loss = loss_fn(preds, targets)
    acc = accuracy(preds, targets, task="multiclass", num_classes=model.num_classes)
    model.train()
    return {"loss": loss, "acc": acc}


def main():
    log.info(f"Training classifier backbone ({MODE})")
    log.info(f"Using {DEVICE} device")

    train_ds, val_ds = create_classification_datasets(dataset_path=DS_PATH)
    train_dl, val_dl = parse_datasets_to_dataloaders(train_ds, val_ds, test_ds=None, **DL_PARAMS)
    num_classes = len(train_ds.labels)
    backbone = create_backbone(MODE)
    model = YOLOv1Classifier(num_classes=num_classes, backbone=backbone).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = torch.inf

    if LOAD_MODEL:
        load_checkpoint(CLASSIFIER_CKPT_PATH, model, optimizer)

    for epoch in range(EPOCHS):
        train_metrics = train_loop(model, train_dl, optimizer, loss_fn, **LOOP_PARAMS)
        val_metrics = val_loop(model, val_dl, loss_fn, **LOOP_PARAMS)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            backbone_ckpt = {"model": model.backbone.state_dict()}
            save_checkpoint(backbone_ckpt, path=BACKBONE_CKPT_PATH)

            classifier_ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(classifier_ckpt, path=CLASSIFIER_CKPT_PATH)

        metrics = merge_dicts(sep="/", train=train_metrics, val=val_metrics)
        lr = optimizer.param_groups[0]["lr"]
        prefix = f"Epoch: {epoch}, LR = {lr:.5f}  |  "
        display_metrics(prefix=prefix, metrics=metrics)


if __name__ == "__main__":
    main()
