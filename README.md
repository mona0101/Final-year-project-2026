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
>
