import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import cv2
import argparse
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline# only tested on diffusers[torch]==0.19.3, may have conflicts with newer versions of diffusers
import torchvision.transforms.functional as TF
from torch import optim
from attack_tools import Attack, save_img

guidance_scale = 1.0
num_images_per_prompt = 1


def main(args):
    wonder3d = DiffusionPipeline.from_pretrained(
        args.wonder3d_ckpts,
        custom_pipeline=args.wonder3d_pipeline,
        torch_dtype=torch.float16
    )
    wonder3d.unet.enable_xformers_memory_efficient_attention()
    wonder3d.to('cuda:0')

    attack = Attack(wonder3d,
                    args.iteration,
                    args.epsilon,
                    args.num_inference_steps,
                    args.attention_res,

                    args.dataset)
    dataset = os.listdir(os.path.join(args.dataset, 'init_image'))

    for image_name in dataset:
        image = Image.open(os.path.join(args.dataset, 'init_image', image_name)).resize((256, 256))
        mask = Image.open(os.path.join(args.dataset, 'mask_image', image_name)).resize((256, 256))
        
        image, adv, result = attack(image=image, mask=mask, image_name=image_name)
        result = make_grid(result, nrow=6, ncol=2, padding=0, value_range=(0, 1))
        save_image(result, os.path.join(args.dataset, 'result', image_name))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wonder3d_ckpts', type=str, default='ckpts', help='wonder3d_ckpts')
    parser.add_argument('--wonder3d_pipeline', type=str, default='pipeline', help='wonder3d_pipeline')
    parser.add_argument('--dataset', type=str, default='3d_assets_protection/', help='dataset')
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--epsilon", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--attention_res", type=int, default=8)
    args = parser.parse_args()
    main(args)
