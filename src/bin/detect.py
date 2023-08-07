import numpy as np
import torch

from src.bin.train_detector import DS_PATH, CONFIG
from src.data.dataset import create_detection_datasets
from src.model.detector import YOLOv1Detector
from src.utils.config import DEVICE, SEED, ROOT
from src.utils.model import create_backbone, create_pre_head, load_checkpoint
from src.visualization import plot_yolo_labels
import random
import matplotlib.pyplot as plt
import mlflow

torch.manual_seed(SEED)
random.seed(SEED)

RUN_ID = "13177606a5164da9ba90e661234bc1de"
CKPT_PATH = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID, artifact_path="checkpoints/best.pt", dst_path=str(ROOT / "tmp")
)


def detect(
    model: YOLOv1Detector,
    images: list[torch.Tensor],
    iou_threshold: float,
    obj_threshold: float,
    id2label: dict[int, str],
    device: str = "cpu",
):
    preds_path = ROOT / "img"
    preds_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    images_tensor = torch.stack(images).to(device)
    all_pred_boxes = model.inference(images_tensor)
    all_pred_boxes = model.perform_nms(all_pred_boxes, iou_threshold, obj_threshold)
    for i, (img, pred_boxes) in enumerate(zip(images, all_pred_boxes)):
        fig, ax = plt.subplots(figsize=(6, 6))
        img = (img.permute(1, 2, 0).to("cpu").numpy() * 255).astype(np.uint8)
        nms_boxes = torch.tensor(pred_boxes).numpy()
        if len(nms_boxes) == 0:
            ax.imshow(img)
            ax.axis("off")
        else:
            class_ids = nms_boxes[:, 0].astype(np.int64)
            obj_scores = nms_boxes[:, 1]
            obj_scores = np.clip(obj_scores, 0, 1)
            boxes_xywhn = nms_boxes[:, 2:]

            plot_yolo_labels(
                img, boxes_xywhn, class_ids, obj_scores, plot=True, id2label=id2label, ax=ax
            )
        filename = str(preds_path / f"{i}.jpg")
        fig.savefig(filename, bbox_inches="tight")


def main():
    _, _, test_ds = create_detection_datasets(
        dataset_path=DS_PATH, S=CONFIG["model"]["S"], B=CONFIG["model"]["B"]
    )
    C = test_ds.C
    backbone = create_backbone(CONFIG["backbone"])
    pre_head = create_pre_head(CONFIG["backbone"])
    model = YOLOv1Detector(
        S=CONFIG["model"]["S"], B=CONFIG["model"]["B"], C=C, backbone=backbone, pre_head=pre_head
    ).to(DEVICE)
    load_checkpoint(CKPT_PATH, model)
    images = []
    for i in range(8):
        idx = random.randint(0, len(test_ds))
        img, annot = test_ds[idx]
        images.append(img)
    detect(
        model,
        images=images,
        iou_threshold=CONFIG["model"]["iou_threshold"],
        obj_threshold=CONFIG["model"]["objectness_threshold"],
        id2label=test_ds.id2label,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
