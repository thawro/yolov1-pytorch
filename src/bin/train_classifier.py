from src.model.classifier import YOLOv1Classifier
from src.data.dataset import create_classification_datasets, DataModule
from src.model.module import ClassificationModelModule

from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED
from src.utils.model import create_backbone
from src.logging.pylogger import get_pylogger
from src.logging.loggers import TerminalLogger, MLFlowLogger, BaseLogger

import torch.optim as optim
import torch.utils.data
import torch
from torch import nn

log = get_pylogger(__name__)
torch.manual_seed(SEED)

MODE = "yolov1-tiny"

DS_NAME = "HWD+"
DS_PATH = str(DATA_PATH / DS_NAME)
LOAD_WEIGHTS = False

EXPERIMENT_NAME = "classifier"
RUN_NAME = f"{DS_NAME}__{MODE}"
LOGS_PATH = ROOT / "logs" / EXPERIMENT_NAME / RUN_NAME

DL_PARAMS = dict(batch_size=32, num_workers=8, drop_last=True, pin_memory=True)

EPOCHS = 20
LIMIT_BATCHES = -1


def main():
    log.info(f"Training classifier backbone ({MODE})")
    log.info(f"Using {DEVICE} device")

    train_ds, val_ds = create_classification_datasets(dataset_path=DS_PATH)
    datamodule = DataModule(train_ds, val_ds, None, **DL_PARAMS)
    num_classes = len(train_ds.labels)
    backbone = create_backbone(MODE)
    model = YOLOv1Classifier(num_classes=num_classes, backbone=backbone).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    logger = BaseLogger(LOGS_PATH)
    module = ClassificationModelModule(
        model=model,
        datamodule=datamodule,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        logger=logger,
        limit_batches=LIMIT_BATCHES,
        device=DEVICE,
    )

    module.fit(epochs=EPOCHS, load_weights=LOAD_WEIGHTS)


if __name__ == "__main__":
    main()
