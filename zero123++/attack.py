import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.backends
import torch.backends.cudnn
import argparse
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from attack_tools import Attack, save_img
import torch

def main(args):
    pipeline = DiffusionPipeline.from_pretrained(
        args.zero123_ckpt, custom_pipeline=args.zero123_pipeline, torch_dtype=torch.float16
    )

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to('cuda:0')

    attack = Attack(pipeline,
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
        result.save(os.path.join(args.dataset, 'result', image_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args for MultiView attack")
    parser.add_argument("--dataset", type=str, default='3d_assets_protection')
    parser.add_argument("--zero123_pipeline", type=str, default='zeros123plus/pipeline')
    parser.add_argument("--zero123_ckpt", type=str, default='zeros123plus/v1.1')
    parser.add_argument("--iteration", type=int, default=100)
    parser.add_argument("--epsilon", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--attention_res", type=int, default=16)
    args = parser.parse_args()
    print(args)
    main(args)