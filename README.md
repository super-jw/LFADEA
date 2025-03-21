## Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection

Code for ICME 2025 "[Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection](https://arxiv.org/abs/2408.11408)" 

by *Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun*.

Full code and instructions will be completed soon.

## Introduction

This paper is the first to address the ***intellectual property infringement*** issue arising from Multi-View Diffusion Models. Accordingly, we propose a novel ***latent feature and attention dual erasure attack*** to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously.

<img src='assets\pipeline.png' style="zoom:80%;" />

## Environment

The repository is divided into two parts, `wonder3d` folder contains the code attacking Wonder3d, and `zero123++` folder contains the code attacking Zero123++. Please cheak the environment detail in each folder.

## Content

- ```./wonder3d```: code for attacking wonder3d.
- ```./zero123++```: code for attacking zero123++.

## Usage

**Attack Wonder3d**

```bash
cd ./wonder3d
python attack.py
```

**Attack Zero123++**

```bash
cd ./zero123++
python attack.py
```

## Citation

```bib
@inproceedings{sun2025latent,
title={Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection},
author={Jingwei Sun and Xuchong Zhang and Changfeng Sun and Qicheng Bai and Hongbin Sun},
booktitle={IEEE International Conference on Multimedia&Expo},
year={2025}
}
```

## Reference Code

[1] Wonder3d: https://github.com/xxlong0/Wonder3D

[2] Zero123++: https://github.com/SUDO-AI-3D/zero123plus

## Contact

Please contact 310412@stu.xjtu.edu.cn if you have any question on the codes.
