from src.model.loss import YoloV1Loss
from src.model.yolov1 import YOLOv1Detector, create_backbone, create_detection_head
from src.utils.config import DEVICE, SEED
from src.data.dataset import create_detection_dataloaders
from src.utils.model import load_checkpoint
from src.bin.train_detector import (
    val_loop,
    S,
    B,
    DS_PATH,
    IOU_THR,
    OBJ_THR,
    DETECTOR_CKPT_PATH,
    BACKBONE_MODE
)
import torch


torch.manual_seed(SEED)


def main():
    train_dl, val_dl, test_dl = create_detection_dataloaders(
        S=S, 
        B=B, 
        dataset_path=DS_PATH, 
        batch_size=128, 
        num_workers=2, 
        drop_last=True, 
        pin_memory=True
    )
    C = train_dl.dataset.C
    backbone = create_backbone(BACKBONE_MODE)
    head = create_detection_head(S, C, B, BACKBONE_MODE, backbone.out_channels)
    model = YOLOv1Detector(S=S, B=B, C=C, backbone=backbone, head=head).to(DEVICE)
    loss_fn = YoloV1Loss(C=C, B=B)
    load_checkpoint(DETECTOR_CKPT_PATH, model)


    metrics = val_loop(
        model,
        test_dl,
        loss_fn,
        iou_threshold=IOU_THR,
        objectness_threshold=OBJ_THR,
        n_plot=1,
        device=DEVICE
    )
    print(f"test/loss: {metrics['loss']:.2f}, test/MAP: {metrics['MAP']:.2f}")


if __name__ == "__main__":
    main()
