from src.model.detector import YOLOv1Detector
from src.model.loss import YoloV1Loss
from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED
from src.data.dataset import create_detection_datasets, parse_datasets_to_dataloaders
from src.utils.utils import train_loop, merge_dicts, display_metrics
from src.utils.ops import MAP, cellboxes_to_boxes
from src.utils.model import (
    save_checkpoint,
    load_checkpoint,
    create_backbone,
    create_pre_head,
)
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
import torch.utils.data
import torch
from torch import nn
from tqdm.auto import tqdm


torch.manual_seed(SEED)

MODE = "yolov1-tiny"
DS_NAME = "yolo_HWD+"
CLASSIFIER_DS_NAME = "HWD+"
# DS_NAME = "VOC"
DS_PATH = str(DATA_PATH / DS_NAME)

LOAD_CLASSIFIER_BACKBONE_WEIGHTS = False
LOAD_MODEL = True

EPOCHS = 100
DETECTOR_CKPT_PATH = ROOT / f"checkpoints/detector/{DS_NAME}/{MODE}"
DETECTOR_CKPT_PATH.mkdir(parents=True, exist_ok=True)
DETECTOR_CKPT_PATH = str(DETECTOR_CKPT_PATH / "best.pt")
CLASSIFIER_BACKBONE_CKPT_PATH = str(
    ROOT / f"checkpoints/classifier/{CLASSIFIER_DS_NAME}/{MODE}/backbone_best.pt"
)

S = 7
B = 2
IOU_THR = 0.5
OBJ_THR = 0.4
DL_PARAMS = dict(batch_size=16, num_workers=2, drop_last=True, pin_memory=True)

LIMIT_BATCHES = -1

LOOP_PARAMS = dict(device=DEVICE, limit_batches=LIMIT_BATCHES)


def val_loop(
    model: YOLOv1Detector,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.modules.loss._Loss,
    iou_threshold: float,
    obj_threshold: float,
    device: str = "cpu",
    limit_batches: int = -1,
) -> dict[str, float]:
    loop = tqdm(dataloader, leave=True, desc="Val")
    all_pred_boxes = []
    all_true_boxes = []
    loss_values = []
    model.eval()
    train_idx = 0
    for x, y in loop:
        if limit_batches == 0:
            break
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]

        with torch.no_grad():
            out = model(x)

        loss = loss_fn(out, y).item()
        loss_values.append(loss)
        pred_boxes = cellboxes_to_boxes(out, S=model.S, C=model.C, B=model.B)
        pred_boxes = model.perform_nms(pred_boxes, iou_threshold, obj_threshold)
        true_boxes = cellboxes_to_boxes(y, S=model.S, C=model.C, B=model.B)

        for idx in range(batch_size):
            for nms_box in pred_boxes[idx]:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_boxes[idx]:
                if box[1] > obj_threshold:
                    all_true_boxes.append([train_idx] + box.tolist())
            train_idx += 1
        loop.set_postfix(loss=loss)
        limit_batches -= 1

    mean_avg_prec = MAP(model.C, all_pred_boxes, all_true_boxes, iou_threshold=iou_threshold)
    mean_loss = sum(loss_values) / len(loss_values)
    model.train()
    return {"loss": mean_loss, "MAP": mean_avg_prec}


def main():
    train_ds, val_ds, _ = create_detection_datasets(S=S, B=B, dataset_path=DS_PATH)
    train_dl, val_dl = parse_datasets_to_dataloaders(train_ds, val_ds, test_ds=None, **DL_PARAMS)

    C = train_ds.C
    backbone = create_backbone(MODE)
    pre_head = create_pre_head(MODE)
    model = YOLOv1Detector(S=S, B=B, C=C, backbone=backbone, pre_head=pre_head).to(DEVICE)
    # optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # scheduler = ChainedScheduler(
    #     [
    #         LinearLR(optimizer, start_factor=0.1, total_iters=10),
    #         MultiStepLR(optimizer, milestones=[85, 115, 145], gamma=0.1),
    #     ]
    # )
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    loss_fn = YoloV1Loss(C=C, B=B)
    best_loss = torch.inf

    if LOAD_CLASSIFIER_BACKBONE_WEIGHTS:
        load_checkpoint(CLASSIFIER_BACKBONE_CKPT_PATH, model.backbone)

    if LOAD_MODEL:
        load_checkpoint(DETECTOR_CKPT_PATH, model, optimizer)

    for epoch in range(EPOCHS):
        train_metrics = train_loop(model, train_dl, optimizer, loss_fn, **LOOP_PARAMS)
        scheduler.step()
        val_metrics = val_loop(model, val_dl, loss_fn, IOU_THR, OBJ_THR, **LOOP_PARAMS)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(ckpt, path=DETECTOR_CKPT_PATH)
        metrics = merge_dicts(sep="/", train=train_metrics, val=val_metrics)
        lr = optimizer.param_groups[0]["lr"]
        prefix = f"Epoch: {epoch}, LR = {lr:.5f}  |  "
        display_metrics(prefix=prefix, metrics=metrics)


if __name__ == "__main__":
    main()
