"""Parse VOC dataset annotation format from VOC format (xyxy) to YOLO format (xywh)"""

import xml.etree.ElementTree as ET
from src.utils.config import DATA_PATH
from src.utils.utils import read_text_file, save_txt_to_file
import glob
from tqdm.auto import tqdm
import shutil


def xyxy2xywh(box: tuple[float, float, float, float], img_w: int, img_h: int):
    """Convert x_min, y_min, x_max, y_max to x_c, y_c, w, h"""
    x_c = (box[0] + box[2])/2.0 - 1
    y_c = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x_c /= img_w
    w /= img_w
    y_c /= img_h
    h /= img_h
    return x_c, y_c, w, h

def parse_annotation_voc_to_yolo(filepath: str, labels: list[str]):
    """Parse annotation from VOC format (.xml) to YOLO format (.txt)
    Each object (bounding box) is annotated with `class_id, x_c, y_c, w, h` where
    x_c - center x coordinate
    y_c - center y coordinate
    w - bbox width
    h - bbox height
    The values are normalized wrt the image size.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    annots = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = obj.find('name').text
        if label not in labels or int(difficult) == 1:
            continue
        label_id = labels.index(label)
        xmlbox = obj.find('bndbox')
        
        xyxy_box = tuple([float(xmlbox.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']])
        xywh_box = xyxy2xywh(xyxy_box, img_w, img_h)
        annot = str(label_id) + " " + " ".join([str(val) for val in xywh_box])
        annots.append(annot)
    return "\n".join(annots)


def main():
    VOC_PATH = DATA_PATH / "VOC"
    RAW_VOC_PATH = VOC_PATH / "VOC2012"

    filepaths = sorted(glob.glob(str(RAW_VOC_PATH / "Annotations/*xml")))
    labels = read_text_file(VOC_PATH / "labels.txt")

    all_ids = [path.split("/")[-1].split(".")[0] for path in filepaths]
    train_ids = read_text_file(RAW_VOC_PATH / "ImageSets/Main/train.txt")
    val_ids = read_text_file(RAW_VOC_PATH / "ImageSets/Main/val.txt")
    test_ids = set(all_ids).difference(set(train_ids + val_ids))

    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    # create directories for splits images and labels
    for split in split_ids:
        split_labels_dir = VOC_PATH / "labels" / split
        split_images_dir = VOC_PATH / "images" / split
        split_labels_dir.mkdir(exist_ok=True, parents=True)
        split_images_dir.mkdir(exist_ok=True, parents=True)

    for split, ids in tqdm(split_ids.items(), desc="Splits"):
        for id in tqdm(ids, desc=f"{split}"):
            xml_annot_filepath = str(RAW_VOC_PATH / f"Annotations/{id}.xml")
            yolo_annot_filepath = str(VOC_PATH / "labels" / split / f"{id}.txt")

            src_img_filepath = str(RAW_VOC_PATH / f"JPEGImages/{id}.jpg")
            dst_img_filepath = str(VOC_PATH / "images" / split / f"{id}.jpg")

            yolo_annot = parse_annotation_voc_to_yolo(xml_annot_filepath, labels)

            save_txt_to_file(yolo_annot, yolo_annot_filepath)
            shutil.copy2(src_img_filepath, dst_img_filepath)


if __name__ == "__main__":
    main()