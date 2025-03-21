import os
import numpy
from PIL import Image
import torchvision.transforms as transforms
import torch
from copy import deepcopy
from typing import Union, Tuple, List
import cv2

def save_img(img, output_dir):
    if len(img.shape) == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)

    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    img = img.astype(numpy.uint8)
    img = Image.fromarray(img)
    
    img.save(output_dir)

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def preprocess(image, mean, std, size, rescale_factor, dtype):
    normalize = transforms.Normalize(mean, std)
    resize = transforms.Resize([size, size])

    result = resize(image)
    result = (result * rescale_factor).cuda()
    result = normalize(result).unsqueeze(0)

    result = result.to(dtype)

    return result

class Attack():
    def __init__(self, model, iteration, epsilon, num_inference_steps, attention_res, output_dir):
        self.model = model
        self.iteration = iteration
        self.epsilon = epsilon
        self.num_inference_steps = num_inference_steps
        self.attention_res = attention_res
        self.output_dir = output_dir


    def aggregate_attention(self, attention_maps: dict, res: int, from_where: List[str], is_cross: bool, select: int) -> torch.Tensor:
        out = []
        original_out = []
        num_pixels = res ** 2
       
        for item in attention_maps:
            if item.split('_')[0] in from_where:
                if attention_maps[item].shape[1] == num_pixels:
                    print(item)
                    out.append(attention_maps[item])
        
        for item in attention_maps:
            if item.split('_')[0] in from_where:
                if attention_maps[item].shape[1] == num_pixels:
                    print(item)
                    original_out.append(attention_maps[item].detach())

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out, original_out

    def attack_result(self, image, noise, mask):
        image_adv = image + noise * mask
        image_adv = torch.clamp(image_adv, 0, 255)
        image_vae = preprocess(image=image_adv,
                                mean=0.5,
                                std=0.8,
                                size=512,
                                rescale_factor=0.00392156862745098,
                                dtype=self.model.vae.dtype)
        image_clip = preprocess(image=image_adv,
                                mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954,0.26130258,0.27577711],
                                size=224,
                                rescale_factor=0.00392156862745098,
                                dtype=self.model.vae.dtype)
        result = self.model(num_inference_steps=self.num_inference_steps, image_1=image_vae, image_2=image_clip).images[0]
        return image_adv, result
    
    def compute_loss(self, attention, mask, cond_lat, negative_lat):
        # loss choice 1
        mask_resize = transforms.Resize((self.attention_res, self.attention_res))
        loss_mask = mask_resize(mask)[0]# * 255
        loss_mask = loss_mask.reshape(1, self.attention_res**2)
        loss_mask = loss_mask.repeat(self.attention_res**2, 1)
        loss_attention = torch.sum(loss_mask * attention)

        # loss choice 2
        loss_latent = -torch.mean((torch.pow(negative_lat - cond_lat[1], 2)))
        loss = loss_attention + 1 * loss_latent
        print(f'loss:{loss}, loss_attention:{loss_attention}, loss_latent:{loss_latent}')
        return loss

    def __call__(self, image, mask, image_name):
        self.model.prepare()
        image = to_rgb_image(image)
        image = numpy.array(image)[:, :, :3]
        image = torch.from_numpy(image.transpose((2, 0, 1))).cuda()

        mask = to_rgb_image(mask)
        mask = numpy.array(mask)[:, :, :3] / 255
        mask = torch.from_numpy(mask.transpose((2, 0, 1))).cuda()
        noise = torch.zeros_like(image, dtype=torch.float32)
        noise.requires_grad = True

        self.model.scheduler.set_timesteps(self.num_inference_steps, device=self.model._execution_device)
        timesteps = self.model.scheduler.timesteps
        for iter in range(self.iteration):
            image_adv = image + noise * mask
            image_adv = torch.clamp(image_adv, 0, 255)
            image_vae = preprocess(image=image_adv,
                                   mean=0.5,
                                   std=0.8,
                                   size=512,
                                   rescale_factor=0.00392156862745098,
                                   dtype=self.model.vae.dtype)
            image_clip = preprocess(image=image_adv,
                                    mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954,0.26130258,0.27577711],
                                    size=224,
                                    rescale_factor=0.00392156862745098,
                                    dtype=self.model.vae.dtype)
            

            image_vae = image_vae.to(device=self.model.vae.device, dtype=self.model.vae.dtype)
            image_clip = image_clip.to(device=self.model.vae.device, dtype=self.model.vae.dtype)
            cond_lat = self.model.encode_condition_image(image_vae)

            with torch.no_grad():
                original_vae = preprocess(image=image,
                                          mean=0.5,
                                          std=0.8,
                                          size=512,
                                          rescale_factor=0.00392156862745098,
                                          dtype=self.model.vae.dtype)
                original_vae = original_vae.to(device=self.model.vae.device, dtype=self.model.vae.dtype)
                original_lat = self.model.encode_condition_image(original_vae)
            
            negative_lat = self.model.encode_condition_image(torch.zeros_like(image_vae, dtype=self.model.vae.dtype))
            cond_lat = torch.cat([negative_lat, cond_lat])
            encoded = self.model.vision_encoder(image_clip, output_hidden_states=False)
            global_embeds = encoded.image_embeds
            global_embeds = global_embeds.unsqueeze(-2)
            
            
            encoder_hidden_states = self.model._encode_prompt(
                prompt="",
                device=self.model.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            ramp = global_embeds.new_tensor(self.model.config.ramping_coefficients).unsqueeze(-1)
            encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
            prompt_embeds = self.model._encode_prompt(
                None,
                device = self.model.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None,
                prompt_embeds=encoder_hidden_states,
                negative_prompt_embeds=None,
                lora_scale=None,
                )

            # encoder_hidden_states = encoder_hidden_states.to(dtype=self.model.vae.dtype, device=self.model.device)
            # bs_embed, seq_len, _ = prompt_embeds.shape
            # # duplicate text embeddings for each generation per prompt, using mps friendly method
            # prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            # prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

            cak = dict(cond_lat=cond_lat)
                       
            latent_timestep = timesteps[torch.randperm(timesteps.shape[0])[0]]
            # latent_timestep = timesteps[iter]
            ref_dict = self.model.unet(
                None,
                latent_timestep,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cak,
                return_dict=False,
                attack_mode=True,
                )

            adv_attention_maps, original_maps = self.aggregate_attention(
                attention_maps=ref_dict,
                res=self.attention_res,
                from_where=("up", "down", "mid"),
                is_cross=False,
                select=0)

            loss = self.compute_loss(attention=adv_attention_maps, mask=mask, cond_lat=cond_lat, negative_lat=original_lat)
            print(f"iter:{iter},loss:{loss}")
            loss.backward()
            grad = noise.grad.sign()

            noise = noise - grad
            noise.data.clamp_(-self.epsilon, self.epsilon)
            noise = noise.clone().detach()
            noise.requires_grad = True

        image_adv, result = self.attack_result(image, noise, mask)
        return image, image_adv, result