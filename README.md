---
license: apache-2.0
datasets:
- XavierJiezou/cloud-adapter-datasets
language:
- en
metrics:
- mean_iou
base_model:
- facebook/dinov2-large
---


# Cloud Adapter Models

This repository contains the pre-trained model weights for the [Cloud-Adapter](https://xavierjiezou.github.io/Cloud-Adapter/).

## Installation

```bash
git clone https://huggingface.co/XavierJiezou/cloud-adapter-models
cd cloud-adapter-models
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

## Citation

If you use our code or models in your research, please cite with:

```latex
@misc{zou2024adaptingvisionfoundationmodels,
      title={Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images}, 
      author={Xuechao Zou and Shun Zhang and Kai Li and Shiying Wang and Junliang Xing and Lei Jin and Congyan Lang and Pin Tao},
      year={2024},
      eprint={2411.13127},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13127}, 
}
```