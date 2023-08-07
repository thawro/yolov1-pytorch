from typing import Literal, Optional, Callable, Any
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from src.utils.utils import read_text_file
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.datasets
import albumentations as A
from .transforms import TRAIN_TRANSFORMS, INFERENCE_TRANSFORMS


class BaseDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "val"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self._init_dirpaths()
        self._init_filepaths()
        self.labels = read_text_file(f"{root}/labels.txt")

    def _init_dirpaths(self):
        self.root = Path(self.root)
        self.labels_path = self.root / "labels" / self.split
        self.images_path = self.root / "images" / self.split

    def _init_filepaths(self):
        self._image_files = glob.glob(f"{self.images_path}/*")
        ext = self._image_files[0].split(".")[-1]
        self._label_files = [
            img_file.replace("images/", "labels/").replace(f".{ext}", ".txt")
            for img_file in self._image_files
        ]

    def get_raw_data(self, idx: int):
        image_filepath = self._image_files[idx]
        labels_filepath = self._label_files[idx]

        image = np.array(Image.open(image_filepath).convert("RGB"))
        target = read_text_file(labels_filepath)
        return image, target

    def __len__(self):
        return len(self._label_files)


class DetectionDataset(BaseDataset):
    def __init__(
        self,
        S: int,
        B: int,
        root: str,
        split: Literal["train", "test", "val"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.S = S  # grid size
        self.B = B  # num boxes
        super().__init__(root, split, transform, target_transform)
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.C = len(self.labels)

    def _transform(
        self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray
    ) -> tuple[torch.Tensor, ...]:
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            return (
                transformed["image"],
                torch.Tensor(transformed["bboxes"]),
                torch.Tensor(transformed["labels"]),
            )
        else:
            return (torch.from_numpy(image), torch.from_numpy(bboxes), torch.from_numpy(labels))

    def __getitem__(self, idx: int) -> Any:
        image, annots = self.get_raw_data(idx)
        # annotations cols:
        # class x_center y_center width height
        annots = np.array([[float(x) for x in label.split(" ")] for label in annots])

        bboxes = annots[:, 1:]
        labels = annots[:, 0]
        image, bboxes, labels = self._transform(image, bboxes, labels)

        annots = torch.cat((labels.unsqueeze(1), bboxes), dim=1)

        annots_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))
        if len(bboxes) == 0:
            return image, annots_matrix

        xy = bboxes[:, :2]
        wh = bboxes[:, 2:]

        boxes_ji = (xy * self.S).int()
        boxes_xy_cell = xy * self.S - boxes_ji
        boxes_wh_cell = wh * self.S

        boxes_xywh_cell = torch.cat((boxes_xy_cell, boxes_wh_cell), dim=1)

        for (j, i), xywh, label in zip(boxes_ji.tolist(), boxes_xywh_cell, labels):
            if annots_matrix[i, j, self.C] == 0:  # first object
                annots_matrix[i, j, self.C] = 1
                # box coords
                annots_matrix[i, j, self.C + 1 : self.C + 5] = xywh
                # one hot class label
                annots_matrix[i, j, int(label)] = 1
        return image, annots_matrix


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, split, transform, target_transform)

    def __getitem__(self, idx: int) -> Any:
        image, label = self.get_raw_data(idx)
        label = int(label[0])  # labels are read as lines from txt file
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


def create_detection_datasets(S: int, B: int, dataset_path: str) -> tuple[DetectionDataset, ...]:
    bbox_params = A.BboxParams(format="yolo", min_visibility=0.0, label_fields=["labels"])

    train_transform = A.Compose(TRAIN_TRANSFORMS, bbox_params=bbox_params)
    inference_transform = A.Compose(INFERENCE_TRANSFORMS, bbox_params=bbox_params)

    train_ds = DetectionDataset(S, B, dataset_path, "train", train_transform)
    val_ds = DetectionDataset(S, B, dataset_path, "val", inference_transform)
    test_ds = DetectionDataset(S, B, dataset_path, "test", inference_transform)
    return train_ds, val_ds, test_ds


def create_classification_datasets(dataset_path: str) -> tuple[ClassificationDataset, ...]:
    train_transform = A.Compose(TRAIN_TRANSFORMS)
    inference_transform = A.Compose(INFERENCE_TRANSFORMS)

    train_ds = ClassificationDataset(dataset_path, "train", train_transform)
    val_ds = ClassificationDataset(dataset_path, "val", inference_transform)
    return train_ds, val_ds


class DataModule:
    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset | None,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
