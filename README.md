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



### 2. Run the Gradio Demo

To interactively test the models using Gradio:

```bash
python app.py
```

#### Notes:
- **GPU Requirement**: If using a GPU, ensure it has at least **16GB of VRAM** to run the model efficiently.
- **CPU-Only Mode**: If you wish to run the demo on CPU, set the environment variable `CUDA_VISIBLE_DEVICES` to `-1`:

```bash
CUDA_VISIBLE_DEVICES=-1 python app.py
``` 


This will launch a web interface where you can upload remote sensing images and view the segmentation results.


## Gradio Demo

The Gradio demo allows users to upload remote sensing images, run cloud segmentation, and visualize the results. It can be easily modified to suit custom datasets or tasks.


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


