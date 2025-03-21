## Introduction

This folder contains the code for attacking zero123++.

## Environment
The code is based on [Zero123++](https://github.com/SUDO-AI-3D/zero123plus). Please refer to it for setup. 

## Usage
**Preparation**
1. Download [checkpoints](https://connecthkuhk-my.sharepoint.com/personal/xxlong_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxxlong%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fwonder3d%2Ffull%2Dckpts%2Fckpts&ga=1) and put them under `./zero123plus`. The overall file structure is as follows:
```
zero123++
|-- zero123plus
    |-- pipeline
    |-- v1.1
        |-- unet
        |-- vae
        ...
```

2. Download dataset from [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer). We provide some sample images in ``3d_assets_protection`` folder.

**Attack Zero123++**

```bash
python attack.py
```
