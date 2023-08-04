import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMGSZ = 448
SCALE = 1.1

TRAIN_TRANSFORMS = [
    A.LongestMaxSize(max_size=int(IMGSZ * SCALE)),
    A.PadIfNeeded(
        min_height=int(IMGSZ * SCALE),
        min_width=int(IMGSZ * SCALE),
        border_mode=cv2.BORDER_CONSTANT,
    ),
    A.RandomCrop(width=IMGSZ, height=IMGSZ),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
    A.OneOf(
        [
            A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT),
        ],
        p=1.0,
    ),
    # A.HorizontalFlip(p=0.5),
    A.Blur(p=0.1),
    A.CLAHE(p=0.1),
    A.Posterize(p=0.1),
    A.ToGray(p=0.1),
    A.ChannelShuffle(p=0.05),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255,
    ),
    ToTensorV2(),
]


INFERENCE_TRANSFORMS = [
    A.LongestMaxSize(max_size=IMGSZ),
    A.PadIfNeeded(min_height=IMGSZ, min_width=IMGSZ, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255,
    ),
    ToTensorV2(),
]
