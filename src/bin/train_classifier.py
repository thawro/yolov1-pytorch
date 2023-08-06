from src.model.classifier import YOLOv1Classifier
from src.data.dataset import create_classification_datasets, DataModule
from src.model.module import ClassificationModelModule

from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED, NOW
from src.utils.model import create_backbone


from src.logging.pylogger import get_pylogger
from src.logging.loggers import MLFlowLogger

import torch.optim as optim
import torch.utils.data
import torch
from torch import nn

log = get_pylogger(__name__)
torch.manual_seed(SEED)

CONFIG = {
    "dataset": "HWD+",
    "backbone": "yolov1-tiny",
    "run_id": None,  # run_id used to resume training
    "experiment_name": "classifier",
    "tracking_uri": "http://0.0.0.0:5000",
    "epochs": 20,
    "limit_batches": -1,
    "seed": SEED,
    "device": DEVICE,
    "dataloader": {"batch_size": 32, "num_workers": 8, "drop_last": True, "pin_memory": True},
    "optimizer": {"lr": 2e-5, "weight_decay": 1e-4},
}

CONFIG["run_name"] = f"{CONFIG['dataset']}__{CONFIG['backbone']}"

DS_PATH = str(DATA_PATH / CONFIG["dataset"])

LOGS_PATH = ROOT / "results" / CONFIG["experiment_name"] / f"{CONFIG['run_name']}__{NOW}"


def main():
    log.info(f"Training classifier backbone ({CONFIG['backbone']})")
    log.info(f"Using {DEVICE} device")

    train_ds, val_ds = create_classification_datasets(dataset_path=DS_PATH)
    datamodule = DataModule(train_ds, val_ds, None, **CONFIG["dataloader"])
    num_classes = len(train_ds.labels)
    backbone = create_backbone(CONFIG["backbone"])
    model = YOLOv1Classifier(num_classes=num_classes, backbone=backbone)
    optimizer = optim.Adam(model.parameters(), **CONFIG["optimizer"])
    loss_fn = nn.CrossEntropyLoss()

    logger = MLFlowLogger(
        log_path=LOGS_PATH,
        config=CONFIG,
        experiment_name=CONFIG["experiment_name"],
        tracking_uri=CONFIG["tracking_uri"],
        run_id=CONFIG["run_id"],
        run_name=CONFIG["run_name"],
    )

    module = ClassificationModelModule(
        model=model,
        datamodule=datamodule,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        logger=logger,
        limit_batches=CONFIG["limit_batches"],
        device=DEVICE,
    )

    if CONFIG["run_id"] is not None:  # resuming training
        CONFIG["ckpt"] = logger.download_artifact(
            artifact_path="checkpoints/last.pt",
        )
    else:
        CONFIG["ckpt"] = None

    module.fit(epochs=CONFIG["epochs"], ckpt_path=CONFIG["ckpt"])


if __name__ == "__main__":
    main()
