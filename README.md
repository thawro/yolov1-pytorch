# About
PyTorch implementation of [YOLOv1](https://arxiv.org/pdf/1506.02640.pdf) architecture

# Details
Model training is split into two parts:
1. Train classifier using first conv layers as backbone (`yolov1` or `yolov1-tiny`)
2. Train object detector using classifier backbone

# How to train the model?
1. Place your data inside datasets directory with the following structure:
```
├── datasets                                            # directory with all datasets
│   ├── {dataset_name}                                  # directory for specific dataset
|   |   ├── images                                      # directory with all splits data
|   |   |   ├── train                                   # directory with train split images files
|   |   |   |   ├── {filename_0}.{png/jpg/jpeg}        
|   |   |   |   ├── ...
|   |   |   |   └── {filename_K}.{png/jpg/jpeg}
|   |   |   ├── val
|   |   |   |   ├── {filename_0}.{png/jpg/jpeg}
|   |   |   |   ├── ...
|   |   |   |   └── {filename_L}.{png/jpg/jpeg}
|   |   |   ├── test
|   |   |   |   ├── {filename_0}.{png/jpg/jpeg}
|   |   |   |   ├── ...
|   |   |   |   └── {filename_M}.{png/jpg/jpeg}
|   |   ├── labels                                      # directory with all splits labels
|   |   |   ├── train                                   # directory with train split labels files (YOLO format)
|   |   |   |   ├── {filename_0}.txt
|   |   |   |   ├── ...
|   |   |   |   └── {filename_K}.txt
|   |   |   ├── val
|   |   |   |   ├── {filename_0}.txt
|   |   |   |   ├── ...
|   |   |   |   └── {filename_L}.txt
|   |   |   ├── test
|   |   |   |   ├── {filename_0}.txt
|   |   |   |   ├── ...
|   |   |   |   └── {filename_M}.txt
|   |   ├── labels.txt                                  # txt file with class names for each index
├── src
│   ├── ...
│
├── ...
```

2. Run mlflow server locally
```
mlflow server --default-artifact-root ./mlruns/ --backend-store-uri ./mlruns/ --host 0.0.0.0       
```
3. Set `CONFIG` parameters inside `src/bin/train_classifier.py` and train the classifier
```
python src/bin/train_classifier.py
```
3. Set `CONFIG` parameters inside `src/bin/train_detector.py` (especially the `classifier_run_id` field to load classifiers backbone weight from mlflow run) and train the detector
4. Set `RUN_ID` field (mlflow run_id of detectors run) inside `src/bin/detect.py` to load detector weights and run detection
```
python src/bin/detect.py
```

# Example results for the handwritten digits detection:
## Dataset
[HWD+](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9702948/) dataset was used to create the data for classifier and detector training (more about dataset creation in [my YOLOv8 project](https://github.com/thawro/yolov8-digits-detection#yolo_hwd)).

## Training curves
Train and validation losses
![train_val_loss](img/train_val_loss.png)

Validation mAP (~0.83)
![val_map](img/val_map.png)

## Predictions on images from the test set
![pred_0](img/0.jpg)
![pred_1](img/1.jpg)
![pred_2](img/2.jpg)
![pred_3](img/3.jpg)
![pred_4](img/4.jpg)
![pred_5](img/5.jpg)
![pred_6](img/6.jpg)
![pred_7](img/7.jpg)

