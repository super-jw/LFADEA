## Introduction

This folder contains the code for attacking wonder3d.

## Environment
The code is based on [Wonder3D](https://github.com/xxlong0/Wonder3D). Please refer to it for setup. 

## Usage
**Preparation**
1. Download [checkpoints](https://connecthkuhk-my.sharepoint.com/personal/xxlong_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxxlong%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fwonder3d%2Ffull%2Dckpts%2Fckpts&ga=1) and put them under `./ckpts`. The overall file structure is as follows:
```
wonder3d
|-- ckpts
    |-- unet
    |-- scheduler
    |-- vae
    ...
```

2. Download the [SAM](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) model. Put it to the ``sam_pt`` folder.
```
Wonder3D
|-- sam_pt
    |-- sam_vit_h_4b8939.pth
```

3. Download dataset from [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer). We provide some sample images in ``3d_assets_protection`` folder.

**Attack Wonder3d**

```bash
python attack.py
```