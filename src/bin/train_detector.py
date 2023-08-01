from src.model.yolov1 import YOLOv1Detector, create_backbone, create_detection_head
from src.model.loss import YoloV1Loss
from src.utils.config import DATA_PATH, ROOT, DEVICE, SEED
from src.data.dataset import create_detection_dataloaders
from src.utils.utils import train_loop
from src.utils.ops import MAP, cellboxes_to_boxes
from src.utils.model import save_checkpoint, load_checkpoint, load_classifier_checkpoint
from src.visualization import plot_yolo_labels
import torch.optim as optim
import torch
import numpy as np

torch.manual_seed(SEED)

BACKBONE_MODE = "yolov1"
DS_NAME = "yolo_HWD+"
# DS_NAME = "VOC"
DS_PATH = str(DATA_PATH / DS_NAME)

LOAD_CLASSIFIER_WEIGHTS = False
LOAD_MODEL = True

EPOCHS = 100
DETECTOR_CKPT_PATH = ROOT / f"checkpoints/detector/{DS_NAME}/{BACKBONE_MODE}"
DETECTOR_CKPT_PATH.mkdir(parents=True, exist_ok=True)
DETECTOR_CKPT_PATH = str(DETECTOR_CKPT_PATH / "last.pt")
CLASSIFIER_CKPT_PATH = str(ROOT / f"checkpoints/classifier/{DS_NAME}/{BACKBONE_MODE}/last.pt")

S = 7
B = 2
IOU_THR = 0.5
OBJ_THR = 0.4


def val_loop(model, dataloader, loss_fn, iou_threshold, objectness_threshold, n_plot=0, device='cpu'):
    all_pred_boxes = []
    all_true_boxes = []
    loss_values = []
    model.eval()
    train_idx = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]

        with torch.no_grad():
            out = model(x)

        loss = loss_fn(out, y)
        loss_values.append(loss.item())
        pred_boxes = cellboxes_to_boxes(out, S=model.S, C=model.C, B=model.B)
        pred_boxes = model.perform_nms(pred_boxes, iou_threshold, objectness_threshold)
        true_boxes = cellboxes_to_boxes(y, S=model.S, C=model.C, B=model.B)

        for idx in range(batch_size):
            for nms_box in pred_boxes[idx]:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_boxes[idx]:
                if box[1] > objectness_threshold:
                    all_true_boxes.append([train_idx] + box.tolist())

            train_idx += 1

        for i in range(n_plot):
            nms_boxes = torch.tensor(pred_boxes[i]).numpy()
            img = (x[i].permute(1, 2, 0).to("cpu").numpy() * 255).astype(np.uint8)
            class_ids = nms_boxes[:, 0].astype(np.int64)
            obj_scores = nms_boxes[:, 1]
            obj_scores = np.clip(obj_scores, 0, 1)
            boxes_xywhn = nms_boxes[:, 2:]
            plot_yolo_labels(
                img, boxes_xywhn, class_ids, obj_scores, plot=True, id2label=dataloader.dataset.id2label
            )

        

    mean_avg_prec = MAP(model.C, all_pred_boxes, all_true_boxes, iou_threshold=iou_threshold)
    mean_loss = sum(loss_values) / len(loss_values)
    model.train()
    return {"MAP": mean_avg_prec, "loss": mean_loss}


def main():
    train_dl, val_dl, test_dl = create_detection_dataloaders(
        S=S, 
        B=B, 
        dataset_path=DS_PATH, 
        batch_size=16, 
        num_workers=2, 
        drop_last=True, 
        pin_memory=True
    )
    C = train_dl.dataset.C
    backbone = create_backbone(BACKBONE_MODE)
    head = create_detection_head(S, C, B, BACKBONE_MODE, backbone.out_channels)
    model = YOLOv1Detector(S=S, B=B, C=C, backbone=backbone, head=head).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)
    loss_fn = YoloV1Loss(C=C, B=B)

    if LOAD_CLASSIFIER_WEIGHTS:
        load_classifier_checkpoint(CLASSIFIER_CKPT_PATH, model)

    if LOAD_MODEL:
        load_checkpoint(DETECTOR_CKPT_PATH, model, optimizer)

    for epoch in range(EPOCHS):
        val_metrics = val_loop(
            model, val_dl, loss_fn, iou_threshold=IOU_THR, objectness_threshold=OBJ_THR, plot=False, device=DEVICE
        )
        val_MAP, val_loss = val_metrics["MAP"], val_metrics["loss"]
        train_loss = train_loop(model, train_dl, optimizer, loss_fn, device=DEVICE)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_checkpoint(ckpt, path=DETECTOR_CKPT_PATH)

        print(
            f"Epoch {epoch}: train/loss: {train_loss:.2f},  val/loss: {val_loss:.2f},  val/MAP: {val_MAP:.2f}"
        )


if __name__ == "__main__":
    main()
