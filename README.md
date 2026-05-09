# Final-year-project-2026


# unimodels weights

https://drive.google.com/drive/folders/1POsoG35KjqZesHAH-NEi2zGWXv6KoEyG?usp=sharing

# fusion weights
https://drive.google.com/drive/folders/1CI0CcZMj0FwdRkXOu2V4ZI4izZVUhV3Z?usp=sharing
# Final-year-project-2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the source code and experimental results for the 2026 Final Year Project, focusing on multimodal deep learning and fusion strategies for synchronized data processing.

## Project Structure

The repository is organized into functional modules to ensure scalability and ease of experimentation.

### 1. data_preparation/
This directory handles the end-to-end data pipeline, from raw input processing to model-ready tensors.
* **augmentation.py**: Implementation of various augmentation techniques applied across all modalities (Video, Audio, etc.) to improve model robustness and generalize performance.
* **dataset_loader_2.py**: A synchronized data loading utility. It ensures that video frames and audio signals are perfectly aligned temporally during the training and inference phases.

> **Note on Imports:** The dataset loader has been moved to this directory. When importing from other modules, ensure you use the package prefix to avoid path errors:
> ```python
> from data_preparation.dataset_loader_2 import YourLoaderClass
># Libraries & Tools Used

## Core Frameworks
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)  
[![Torchvision](https://img.shields.io/badge/Library-Torchvision-red.svg)](https://pytorch.org/vision/stable/)  
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-green.svg)](https://ultralytics.com/)  

---

## Audio Processing
[![Librosa](https://img.shields.io/badge/Audio-Librosa-blue.svg)](https://librosa.org/)  
[![Torchaudio](https://img.shields.io/badge/Audio-Torchaudio-blueviolet.svg)](https://pytorch.org/audio/stable/)  

---

## Data Processing
[![NumPy](https://img.shields.io/badge/Library-NumPy-lightgrey.svg)](https://numpy.org/)  
[![Pandas](https://img.shields.io/badge/Library-Pandas-darkblue.svg)](https://pandas.pydata.org/)  

---

## Image & Video Processing
[![OpenCV](https://img.shields.io/badge/Library-OpenCV-green.svg)](https://opencv.org/)  
[![Pillow](https://img.shields.io/badge/Library-Pillow-yellow.svg)](https://python-pillow.org/)  

---

## Visualization
[![Matplotlib](https://img.shields.io/badge/Plot-Matplotlib-blue.svg)](https://matplotlib.org/)  
[![Seaborn](https://img.shields.io/badge/Plot-Seaborn-teal.svg)](https://seaborn.pydata.org/)  

---

## Deep Learning
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange.svg)](https://pytorch.org/)  
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)  

---

## Augmentation & Utilities
[![AugLy](https://img.shields.io/badge/Augmentation-AugLy-purple.svg)](https://github.com/facebookresearch/AugLy)  
[![Supervision](https://img.shields.io/badge/Tool-Supervision-black.svg)](https://github.com/roboflow/supervision)  

---

## Deployment
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red.svg)](https://streamlit.io/)  
