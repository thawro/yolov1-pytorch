import torch
from src.logging.pylogger import get_pylogger
import numpy as np

log = get_pylogger(__name__)


def xywh2xyxy(boxes_xywh: np.ndarray):
    """Parse boxes format from xywh to xyxy"""
    x, y, w, h = boxes_xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes_xyxy = np.column_stack((x1, y1, x2, y2))
    return boxes_xyxy.astype(boxes_xywh.dtype)


def xywhn2xywh(boxes_xywhn: np.ndarray, h: int, w: int):
    """Parse boxes format from xywhn to xywh using image height (`h`) and width (`w`)"""
    xn, yn, wn, hn = boxes_xywhn.T
    x = xn * w
    y = yn * h
    w = wn * w
    h = hn * h
    boxes_xywh = np.column_stack((x, y, w, h))
    return boxes_xywh.astype(np.int16)


def calculate_boxes_iou(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)

    Returns:
        tensor: Intersection over union for all examples
    """

    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def NMS(bboxes: torch.Tensor | list, iou_threshold: float, objectness_threshold: float):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (torch.Tensor): Tensor of shape [S*S, 6]
            columns: [best_class, objectness, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        objectness_threshold (float): threshold to remove predicted bboxes (independent of IoU)

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.tolist()

    bboxes = [box for box in bboxes if box[1] > objectness_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or calculate_boxes_iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]))
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def cellboxes_to_boxes(cell_boxes: torch.Tensor, S: int, C: int, B: int) -> torch.Tensor:
    """Convert boxes wrt cell to boxes wrt image

    Args:
        cell_boxes (torch.Tensor): Boxes predicted by the model.
            Shape: [batch_size, S, S, C + B * 5]
        S (int): grid size
        C (int): number of classes
        B (int): number of boxes

    Returns:
        torch.Tensor: boxes wrt image
    """
    batch_size = cell_boxes.shape[0]

    scores_out = cell_boxes[..., :C]
    scores = scores_out.flatten(start_dim=1, end_dim=2)

    # [batch_size, S, S, C + B * 5] -> [batch_size, S, S, B, 5]
    boxes_out = cell_boxes[..., C:].reshape(batch_size, S, S, B, -1)
    best_boxes_idxs = boxes_out[..., 0].argmax(-1)

    # pick best box per cell by objectness
    index = best_boxes_idxs[..., None, None].expand(-1, -1, -1, 1, boxes_out.size(4))
    best_boxes = torch.gather(boxes_out, dim=3, index=index).squeeze(
        3
    )  # S x S x B x 5 -> S x S x 5
    best_boxes_xywh = best_boxes[..., 1:]

    objectness = best_boxes[..., 0].flatten(start_dim=1, end_dim=-1).unsqueeze(-1)
    best_class = scores.argmax(dim=-1).unsqueeze(-1)

    # convert boxes with cell coords to boxes with img coords
    boxes_xy_cell = best_boxes_xywh[..., :2]
    boxes_wh_cell = best_boxes_xywh[..., 2:]

    ij = torch.arange(7).repeat(7, 1).unsqueeze(-1).unsqueeze(0).to(cell_boxes.device)
    x = (boxes_xy_cell[..., 0:1] + ij) / S
    y = 1 / S * (boxes_xy_cell[..., 1:2] + ij.permute(0, 2, 1, 3))
    wh = boxes_wh_cell / S
    boxes_xywh = torch.cat((x, y, wh), dim=-1)
    boxes_xywh = boxes_xywh.flatten(start_dim=1, end_dim=-2)

    boxes_preds = torch.cat((best_class, objectness, boxes_xywh), dim=-1)
    return boxes_preds
