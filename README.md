# Real-Time Multi-Sensor Drone Detection and Tracking System

## Libraries & Tools Used

[![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-red.svg)](https://pytorch.org/vision/stable/)
[![Torch](https://img.shields.io/badge/Torch-orange.svg)](https://pytorch.org/)
[![YOLOv8 - Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)
[![Librosa](https://img.shields.io/badge/Librosa-blue.svg)](https://librosa.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-blueviolet.svg)](https://pytorch.org/audio/stable/)
[![AugLy](https://img.shields.io/badge/AugLy-purple.svg)](https://github.com/facebookresearch/AugLy)
[![Pillow](https://img.shields.io/badge/Pillow-yellow.svg)](https://python-pillow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-lightgrey.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-teal.svg)](https://seaborn.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-darkblue.svg)](https://pandas.pydata.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-green.svg)](https://opencv.org/)
[![Supervision](https://img.shields.io/badge/Supervision-black.svg)](https://github.com/roboflow/supervision)
[![Streamlit](https://img.shields.io/badge/Streamlit-red.svg)](https://streamlit.io/)


## Overview

This project is a **Real-Time Multi-Sensor Drone Detection and Tracking System** designed to detect and track drones in real time using deep learning and sensor fusion techniques.

The system integrates three main data sources:

- **Radio-Frequency (RF)** signals
- **Audio** signals
- **Video** streams

These modalities are processed independently and then fused together to improve detection accuracy, robustness, and real-time performance.



## Repository Structure

```
Final-year-project-2026/
│
├── data_preparation/
│   ├── dataset_loader2.py        # Synchronized multi-sensor data reading and loading
│   └── augmentation.py           # Data augmentation strategies for each sensor modality
│
├── models/
│   └── all_models.py             # All 6 model architectures in one file
│                                 # (ResNet-18, CNN, VGG, MobileNet —
│                                 #  2 architectures per sensor modality)
│
├── uni_model_results2/
│   └── ...                       # Training results for each sensor (6 experiments: 2 per sensor)
│
├── fusion_Results/
│   └── ...                       # Late fusion results combining all 3 sensors
│                                 # (6 combination experiments using trained unimodal models)
│
├── tracking/
│   └── ...                       # YOLOv8-based drone detection and tracking pipeline
│
├── deployment/
│   └── app.py                    # Streamlit web application displaying predictions
│                                 # (uses the best-performing late fusion experiment)
│
└── README.md
```

> **Note on imports:** If you run the unimodal model scripts directly, you may need to update import paths for `augmentation` and `dataset_loader2` to use their full module paths:
> ```python
> from data_preparation.dataset_loader2 import ...
> from data_preparation.augmentation import ...
> ```


## System Architecture

The system follows a **three-stage pipeline**:

1. **Unimodal Feature Extraction** — Each sensor modality (RF, Audio, Video) is processed independently using dedicated deep learning models.
2. **Late Fusion** — The outputs of the trained unimodal models are combined across all three modalities to produce a final detection decision.
3. **Tracking & Deployment** — Detected drones are tracked using YOLOv8, and results are visualized through an interactive Streamlit web application.



## Datasets

The multi-modal drone dataset is publicly available at:

**[https://github.com/trimodaldataset/multi-modal-drone-data](https://github.com/trimodaldataset/multi-modal-drone-data)**

> **Note:** To run the Streamlit deployment app, you must download the dataset and organize it according to the expected folder structure for each modality (RF, Audio, Video).



## Pre-Trained Model Weights

All experiment weights are hosted on Google Drive:

| Model Type | Download Link |
|---|---|
| **Unimodal Model Weights** (6 experiments) | [Google Drive — Unimodal Weights](https://drive.google.com/drive/folders/1POsoG35KjqZesHAH-NEi2zGWXv6KoEyG?usp=sharing) |
| **Fusion Model Weights** (6 experiments) | [Google Drive — Fusion Weights](https://drive.google.com/drive/folders/1CI0CcZMj0FwdRkXOu2V4ZI4izZVUhV3Z?usp=sharing) |



## Experiments & Results

### Unimodal Experiments (`uni_model_results2/`)

- **6 total experiments** — 2 per sensor modality (RF, Audio, Video)
- Results include training/validation accuracy, loss curves, and classification metrics

### Fusion Experiments (`fusion_Results/`)

- **6 fusion combinations** — using the trained unimodal models via late fusion
- Final predictions are produced by combining the outputs of all three modalities
- The best-performing fusion configuration is used in the deployment application



## Deployment

The system includes a **Streamlit web application** that provides an interactive interface for visualizing real-time drone detection predictions.

To run the app:

```bash
streamlit run deployment/app.py
```

Once the app is running, you can test the model by uploading your sensor data directly through the interface. The app supports uploading **zipped folders** containing the original raw data for each modality (RF, Audio, Video).

## Tracking

The `tracking/` folder contains the full YOLOv8-based drone detection and tracking pipeline, including:

- YOLO model configuration and inference
- Multi-object tracking integration
- Frame-by-frame visualization



## References

- TRIDENT Multi-Sensor Drone Detection System: [https://github.com/TRIDENT-2025/TRIDENT](https://github.com/TRIDENT-2025/TRIDENT)


