from src.model.detector import YOLOv1Detector
from src.model.loss import YoloV1Loss
from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED, NOW

from src.data.dataset import create_detection_datasets, DataModule
from src.model.module import DetectionModelModule

from src.utils.model import (
    load_checkpoint,
    create_backbone,
    create_pre_head,
)
from src.logging.loggers import MLFlowLogger
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
import torch.utils.data
import torch


torch.manual_seed(SEED)

CONFIG = {
    "dataset": "yolo_HWD+",
    "backbone": "yolov1-tiny",
    "classifier_run_id": "d29a4f26ae9c48d19a642292d8f32c1b",
    "run_id": None,
    "experiment_name": "detector",
    "tracking_uri": "http://0.0.0.0:5000",
    "epochs": 130,
    "limit_batches": -1,
    "seed": SEED,
    "device": DEVICE,
    "model": {"S": 7, "B": 2, "iou_threshold": 0.5, "objectness_threshold": 0.4},
    "dataloader": {"batch_size": 16, "num_workers": 2, "drop_last": True, "pin_memory": True},
    "optimizer": {"lr": 5e-4, "weight_decay": 5e-4},
    "scheduler": {"milestones": [10, 20, 30], "gamma": 0.5},
}

CONFIG["run_name"] = f"{CONFIG['dataset']}__{CONFIG['backbone']}"

DS_PATH = str(DATA_PATH / CONFIG["dataset"])

LOGS_PATH = ROOT / "results" / CONFIG["experiment_name"] / f"{CONFIG['run_name']}____{NOW}"

CLASSIFIER_RUN_NAME = "HWD+__yolov1-tiny__06-08-2023_11:22:32"
CLASSIFIER_BACKBONE_CKPT_PATH = str(
    ROOT / f"results/classifier/{CLASSIFIER_RUN_NAME}/checkpoints/backbone_best.pt"
)


def main():
    train_ds, val_ds, test_ds = create_detection_datasets(
        S=CONFIG["model"]["S"], B=CONFIG["model"]["B"], dataset_path=DS_PATH
    )
    datamodule = DataModule(train_ds, val_ds, test_ds, **CONFIG["dataloader"])
    CONFIG["model"]["C"] = train_ds.C

    backbone = create_backbone(CONFIG["backbone"])
    pre_head = create_pre_head(CONFIG["backbone"])

    model = YOLOv1Detector(
        S=CONFIG["model"]["S"],
        B=CONFIG["model"]["B"],
        C=CONFIG["model"]["C"],
        backbone=backbone,
        pre_head=pre_head,
        iou_threshold=CONFIG["model"]["iou_threshold"],
        obj_threshold=CONFIG["model"]["objectness_threshold"],
    )

    optimizer = Adam(model.parameters(), **CONFIG["optimizer"])
    scheduler = MultiStepLR(optimizer, **CONFIG["scheduler"])
    loss_fn = YoloV1Loss(C=CONFIG["model"]["C"], B=CONFIG["model"]["B"])

    logger = MLFlowLogger(
        log_path=LOGS_PATH,
        config=CONFIG,
        experiment_name=CONFIG["experiment_name"],
        tracking_uri=CONFIG["tracking_uri"],
        run_id=CONFIG["run_id"],
        run_name=CONFIG["run_name"],
    )

    if CONFIG["classifier_run_id"] is not None:
        ckpt_path = logger.download_artifact(
            artifact_path="checkpoints/backbone_best.pt",
            run_id=CONFIG["classifier_run_id"],
        )
        load_checkpoint(ckpt_path, model.backbone)

    module = DetectionModelModule(
        model=model,
        datamodule=datamodule,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        limit_batches=CONFIG["limit_batches"],
        device=DEVICE,
    )
    if CONFIG["run_id"] is not None:  # resuming training
        CONFIG["ckpt"] = logger.download_artifact(artifact_path="checkpoints/last.pt")
    else:
        CONFIG["ckpt"] = None

    module.fit(epochs=CONFIG["epochs"], ckpt_path=CONFIG["ckpt"])


if __name__ == "__main__":
    main()
