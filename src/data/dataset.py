from typing import Literal, Optional, Callable, Any
import torch
import torch.utils.data
import glob
from src.utils.utils import read_text_file
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.datasets
import albumentations as A
from torch.utils.data import DataLoader


class BaseDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "val"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self._init_paths()
        if download:
            self.download()

        self.labels = read_text_file(f"{root}/labels.txt")

        self._image_files = glob.glob(f"{self.images_path}/*")
        ext = self._image_files[0].split(".")[-1]
        self._label_files = [
            img_file.replace("images/", "labels/").replace(f".{ext}", ".txt")
            for img_file in self._image_files
        ]

    def _init_paths(self):
        self.root = Path(self.root)
        self.labels_path = self.root / "labels" / self.split
        self.images_path = self.root / "images" / self.split

    def get_raw_data(self, idx: int):
        image_filepath = self._image_files[idx]
        labels_filepath = self._label_files[idx]

        image = np.array(Image.open(image_filepath).convert("RGB"))
        annotations = read_text_file(labels_filepath)
        return image, annotations

    def __len__(self):
        return len(self._label_files)

    def download(self):
        pass



class DetectionDataset(BaseDataset):
    def __init__(
        self,
        S: int,
        B: int,
        root: str,
        split: Literal["train", "test", "val"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.S = S  # grid size
        self.B = B  # num boxes
        super().__init__(root, split, transform, target_transform, download)
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.C = len(self.labels)

    def __getitem__(self, idx: int) -> Any:
        image, annots = self.get_raw_data(idx)
        # annotations cols:
        # class x_center y_center width height
        annots = torch.Tensor([[float(x) for x in label.split(" ")] for label in annots])

        bboxes = annots[:, 1:]
        labels = annots[:, 0]
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]
        bboxes, labels = torch.Tensor(bboxes), torch.Tensor(labels)
        annots = torch.cat((labels.unsqueeze(1), bboxes), dim=1)

        xy = bboxes[:, :2]
        wh = bboxes[:, 2:]

        boxes_ji = (xy * self.S).int()
        boxes_xy_cell = xy * self.S - boxes_ji
        boxes_wh_cell = wh * self.S

        boxes_xywh_cell = torch.cat((boxes_xy_cell, boxes_wh_cell), dim=1)

        annots_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))

        for (j, i), xywh, label in zip(boxes_ji.tolist(), boxes_xywh_cell, labels):
            if annots_matrix[i, j, self.C] == 0:  # first object
                annots_matrix[i, j, self.C] = 1

                # box coords
                annots_matrix[i, j, self.C + 1 : self.C + 5] = xywh
                # one hot class label
                annots_matrix[i, j, int(label)] = 1

        image = torch.Tensor(image).permute(2, 0, 1) / 255
        return image, annots_matrix


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self._init_paths()
        if download:
            self.download()

        self.labels = read_text_file(f"{root}/labels.txt")
        self._image_files = glob.glob(f"{self.images_path}/*")
        ext = self._image_files[0].split(".")[-1]
        self._label_files = [
            img_file.replace("images/", "labels/").replace(f".{ext}", ".txt")
            for img_file in self._image_files
        ]

    def _init_paths(self):
        self.root = Path(self.root)
        self.labels_path = self.root / "labels" / self.split
        self.images_path = self.root / "images" / self.split

    def get_raw_data(self, idx: int):
        image_filepath = self._image_files[idx]
        labels_filepath = self._label_files[idx]

        image = np.array(Image.open(image_filepath).convert("RGB"))
        annotations = read_text_file(labels_filepath)
        return image, annotations

    def __getitem__(self, idx: int) -> Any:
        image, label = self.get_raw_data(idx)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        image = torch.Tensor(image).permute(2, 0, 1) / 255
        return image, int(label[0])

    def __len__(self):
        return len(self._label_files)

    def download(self):
        pass


def create_detection_dataloaders(S: int, B: int, dataset_path: str, **dataloader_kwargs):
    transform = A.Compose(
        [A.Resize(448, 448)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
    )
    train_ds = DetectionDataset(S, B, dataset_path, "train", transform)
    val_ds = DetectionDataset(S, B, dataset_path, "val", transform)
    test_ds = DetectionDataset(S, B, dataset_path, "test", transform)

    train_dl = DataLoader(dataset=train_ds, shuffle=True, **dataloader_kwargs)
    val_dl = DataLoader(dataset=val_ds, shuffle=True, **dataloader_kwargs)
    test_dl = DataLoader(dataset=test_ds, shuffle=True, **dataloader_kwargs)

    return train_dl, val_dl, test_dl


def create_classification_dataloaders(dataset_path: str, **dataloader_kwargs):
    transform = A.Compose(
        [A.Resize(112, 112)],
    )
    train_ds = ClassificationDataset(dataset_path, "train", transform)
    val_ds = ClassificationDataset(dataset_path, "val", transform)

    train_dl = DataLoader(dataset=train_ds, shuffle=True, **dataloader_kwargs)
    val_dl = DataLoader(dataset=val_ds, shuffle=True, **dataloader_kwargs)

    return train_dl, val_dl
