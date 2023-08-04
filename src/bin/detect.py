import numpy as np
import torch

from src.bin.train_detector import (
    DS_PATH,
    IOU_THR,
    MODE,
    OBJ_THR,
    B,
    S,
)
from src.data.dataset import create_detection_datasets
from src.model.detector import YOLOv1Detector
from src.utils.config import DEVICE, SEED, ROOT
from src.utils.model import create_backbone, create_pre_head, load_checkpoint
from src.visualization import plot_yolo_labels
import random

DETECTOR_CKPT_PATH = str(ROOT / "logs/detector/yolo_HWD+__yolov1-tiny/checkpoints/best.pt")
torch.manual_seed(SEED)
random.seed(SEED)


def detect(
    model: YOLOv1Detector,
    images: list[torch.Tensor],
    iou_threshold: float,
    obj_threshold: float,
    id2label: dict[int, str],
    device: str = "cpu",
):
    model.eval()
    images_tensor = torch.stack(images).to(device)
    all_pred_boxes = model.inference(images_tensor)
    all_pred_boxes = model.perform_nms(all_pred_boxes, iou_threshold, obj_threshold)
    for img, pred_boxes in zip(images, all_pred_boxes):
        nms_boxes = torch.tensor(pred_boxes).numpy()
        img = (img.permute(1, 2, 0).to("cpu").numpy() * 255).astype(np.uint8)
        class_ids = nms_boxes[:, 0].astype(np.int64)
        obj_scores = nms_boxes[:, 1]
        obj_scores = np.clip(obj_scores, 0, 1)
        boxes_xywhn = nms_boxes[:, 2:]
        plot_yolo_labels(img, boxes_xywhn, class_ids, obj_scores, plot=True, id2label=id2label)


def main():
    _, _, test_ds = create_detection_datasets(dataset_path=DS_PATH, S=S, B=B)
    C = test_ds.C
    backbone = create_backbone(MODE)
    pre_head = create_pre_head(MODE)
    model = YOLOv1Detector(S=S, B=B, C=C, backbone=backbone, pre_head=pre_head).to(DEVICE)
    load_checkpoint(DETECTOR_CKPT_PATH, model)
    images = []
    for i in range(8):
        idx = random.randint(0, len(test_ds))
        img, annot = test_ds[idx]
        images.append(img)
    detect(
        model,
        images=images,
        iou_threshold=IOU_THR,
        obj_threshold=OBJ_THR,
        id2label=test_ds.id2label,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
