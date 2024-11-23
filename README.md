---
license: apache-2.0
---


# Cloud Adapter Models

This repository contains the code and pre-trained model weights for the paper **"Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images"**. The models are specifically designed to perform robust cloud segmentation in remote sensing imagery by leveraging and fine-tuning vision foundation models.

## Features

- Pre-trained model weights for cloud segmentation tasks.
- Code for fine-tuning and evaluation of the models on remote sensing datasets.
- A user-friendly **Gradio Demo** to test the models interactively.

## Installation

To use the code in this repository, clone it locally and install the required dependencies:

```bash
git clone https://huggingface.co/XavierJiezou/cloud-adapter-models
cd cloud-adapter-models
pip install -r requirements.txt
```

## Usage

### 1. Download Pre-trained Models

The pre-trained model weights are available in the repository. Download the weights and place them in the appropriate directory.

```bash
# Example command to download weights
wget <link_to_model_weights>
```

### 2. Run the Gradio Demo

To interactively test the models using Gradio:

```bash
python demo.py
```

This will launch a web interface where you can upload remote sensing images and view the segmentation results.

### 3. Fine-tune the Model

You can fine-tune the models on your own datasets. Refer to the `train.py` script for instructions and configuration options.

```bash
python train.py --config configs/config.yaml
```

### 4. Evaluate the Model

Evaluate the model on your test set using the `evaluate.py` script:

```bash
python evaluate.py --weights <path_to_weights> --data <path_to_test_data>
```

## Gradio Demo

The Gradio demo allows users to upload remote sensing images, run cloud segmentation, and visualize the results. It can be easily modified to suit custom datasets or tasks.

### Example Screenshot:
*Add a screenshot of the demo interface here if available.*

## Citation

If you find this repository helpful, please consider citing the paper:

```latex
@{cloud-adapter,
title={Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images}, 
author={Xuechao Zou and Shun Zhang and Kai Li and Shiying Wang and Junliang Xing and Lei Jin and Congyan Lang and Pin Tao},
year={2024},
eprint={2411.13127},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2411.13127}
}
```

## Acknowledgements

This project builds upon vision foundation models and uses open-source libraries for training and evaluation. Special thanks to the research community for their contributions to remote sensing and computer vision.


