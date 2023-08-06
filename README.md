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
