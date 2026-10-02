# Drowning Detection Dataset

## Structure
```
datasets/
├── images/          # Contains image data
│   ├── train/       # Training images
│   └── val/         # Validation images
└── labels/          # Contains annotation files
    ├── train/       # Training labels
    └── val/         # Validation labels
```

## Dataset Details
- Total Classes: 3 (Swimming, Struggling, Drowning)
- Format: YOLO format
- Resolution: 1920x1080
- Image Format: .jpg
- Label Format: .txt

## Label Format
Each .txt file corresponds to an image and contains annotations in YOLO format:
```
<class> <x_center> <y_center> <width> <height>
```

## Classes
0. Swimming
1. Struggling
2. Drowning

## Files
- `train.cache`: Cache file for training data
- `val.cache`: Cache file for validation data
- `classes.txt`: Class definitions

## Usage
Used for training YOLOv8 model for drowning detection system.