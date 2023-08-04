from src.model.detector import YOLOv1Detector
from src.model.loss import YoloV1Loss
from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED

from src.data.dataset import create_detection_datasets, DataModule
from src.model.module import DetectionModelModule

from src.utils.model import (
    load_checkpoint,
    create_backbone,
    create_pre_head,
)
from src.logging.loggers import BaseLogger, MLFlowLogger
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
import torch.utils.data
import torch


torch.manual_seed(SEED)

MODE = "yolov1-tiny"
DS_NAME = "yolo_HWD+"
CLASSIFIER_DS_NAME = "HWD+"
# DS_NAME = "VOC"
DS_PATH = str(DATA_PATH / DS_NAME)

LOAD_CLASSIFIER_BACKBONE_WEIGHTS = False
LOAD_WEIGHTS = "last"

EXPERIMENT_NAME = "detector"
RUN_NAME = f"{DS_NAME}__{MODE}"
LOGS_PATH = ROOT / "logs" / EXPERIMENT_NAME / RUN_NAME

CLASSIFIER_BACKBONE_CKPT_PATH = str(
    ROOT / f"logs/classifier/{CLASSIFIER_DS_NAME}__{MODE}/checkpoints/backbone_best.pt"
)

S = 7
B = 2
IOU_THR = 0.5
OBJ_THR = 0.4
DL_PARAMS = dict(batch_size=16, num_workers=2, drop_last=True, pin_memory=True)

EPOCHS = 100
LIMIT_BATCHES = -1


def main():
    train_ds, val_ds, test_ds = create_detection_datasets(S=S, B=B, dataset_path=DS_PATH)
    datamodule = DataModule(train_ds, val_ds, test_ds, **DL_PARAMS)

    C = train_ds.C
    backbone = create_backbone(MODE)
    pre_head = create_pre_head(MODE)
    model = YOLOv1Detector(
        S=S,
        B=B,
        C=C,
        backbone=backbone,
        pre_head=pre_head,
        iou_threshold=IOU_THR,
        obj_threshold=OBJ_THR,
    ).to(DEVICE)
    if LOAD_CLASSIFIER_BACKBONE_WEIGHTS:
        load_checkpoint(CLASSIFIER_BACKBONE_CKPT_PATH, model.backbone)

    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    loss_fn = YoloV1Loss(C=C, B=B)
    logger = BaseLogger(str(LOGS_PATH))

    module = DetectionModelModule(
        model=model,
        datamodule=datamodule,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        limit_batches=LIMIT_BATCHES,
        device=DEVICE,
    )

    module.fit(epochs=EPOCHS, load_weights=LOAD_WEIGHTS)


if __name__ == "__main__":
    main()
