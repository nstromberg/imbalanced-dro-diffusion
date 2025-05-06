# Imbalanced AT_Diff
Fork of [ICLR 2025] official implementation for "Improved Diffusion-based Generative Model with Better Adversarial Robustness"

# Original Paper
Arxiv version:
```
@article{wang2025improved,
  title={Improved Diffusion-based Generative Model with Better Adversarial Robustness},
  author={Wang, Zekun and Yi, Mingyang and Xue, Shuchen and Li, Zhenguo and Liu, Ming and Qin, Bing and Ma, Zhi-Ming},
  journal={arXiv preprint arXiv:2502.17099},
  year={2025}
}
```
# Installation
To help users get started quickly, we provide an installation script `install.sh`.
More details are in [guided-diffusion](https://github.com/openai/guided-diffusion).

# Reimplementation for 2D data
The `points_train.py` script is provided to adapt the existing adversarial training code to 2D pointcloud data. This implementation was unsuccessful, but is provided for future modification.

The `reimplimentation.ipynb` notebook is a simple standalone implementation of adversarial training for the simple 2D pointcloud data. This is the implementation used in the final slides and contains several changes to make the code more readable and stable.
