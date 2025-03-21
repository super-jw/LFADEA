import os
from typing import Any, Dict, Optional
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from typing import Callable, List, Optional, Union
import PIL
from einops import rearrange, repeat
import xformers
import xformers.ops
from copy import deepcopy
import cv2

num_views = 6
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]
rescale_factor = 0.00392156862745098
guidance_scale = 1.0


def save_img(img, output_dir):
    if len(img.shape) == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)

    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    
    img.save(output_dir)


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def clip_preprocess(image, dtype, save_memory=True):
    normalize = transforms.Normalize(clip_mean, clip_std)
    resize = transforms.Resize([224, 224])

    image = resize(image)
    image = (image * rescale_factor).cuda()
    image = normalize(image).unsqueeze(0)
    if not save_memory:
        image = torch.repeat_interleave(image, num_views * 2, dim=0)
    image = image.to(dtype=dtype)

    return image


def vae_preprocess(image, dtype, save_memory=True):
    if not save_memory:
        image = torch.stack([image] * num_views * 2, dim=0)
    else:
        image = image.unsqueeze(0)
    image = image / 255.0
    image = 2 * image - 1
    image = image.to(dtype=dtype)

    return image


def my_repeat(tensor, num_repeats):
    """
    Repeat a tensor along a given dimension
    """
    if len(tensor.shape) == 3:
        return repeat(tensor,  "b d c -> (b v) d c", v=num_repeats)
    elif len(tensor.shape) == 4:
        return repeat(tensor,  "a b d c -> (a v) b d c", v=num_repeats)


class XFormersMVAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, name):
        self.name = name

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_views=1.,
        multiview_attention=True,
        sparse_mv_attention=False,
        mvcd_attention=False,
        ref_dict={},
        attack_mode=False,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # from yuancheng; here attention_mask is None
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key_raw = attn.to_k(encoder_hidden_states)
        value_raw = attn.to_v(encoder_hidden_states)

        # print('query', query.shape, 'key', key.shape, 'value', value.shape)
        #([bx4, 1024, 320]) key torch.Size([bx4, 1024, 320]) value torch.Size([bx4, 1024, 320])
        # pdb.set_trace()
        # multi-view self-attention
        if multiview_attention:
            if not sparse_mv_attention:
                key = my_repeat(rearrange(key_raw, "(b t) d c -> b (t d) c", t=num_views), num_views)
                value = my_repeat(rearrange(value_raw, "(b t) d c -> b (t d) c", t=num_views), num_views)
            else:
                key_front = my_repeat(rearrange(key_raw, "(b t) d c -> b t d c", t=num_views)[:, 0, :, :], num_views) # [(b t), d, c]
                value_front = my_repeat(rearrange(value_raw, "(b t) d c -> b t d c", t=num_views)[:, 0, :, :], num_views)
                key = torch.cat([key_front, key_raw], dim=1) # shape (b t) (2 d) c
                value = torch.cat([value_front, value_raw], dim=1)

        else:
            # print("don't use multiview attention.")
            key = key_raw
            value = value_raw

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if attack_mode:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            ref_dict[self.name] = attention_probs
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class XFormersJointAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, name):
        self.name = name
    
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_tasks=2,
        ref_dict={},
        attack_mode=False
    ):
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # from yuancheng; here attention_mask is None
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        assert num_tasks == 2  # only support two tasks now

        key_0, key_1 = torch.chunk(key, dim=0, chunks=2)  # keys shape (b t) d c
        value_0, value_1 = torch.chunk(value, dim=0, chunks=2)
        key = torch.cat([key_0, key_1], dim=1)  # (b t) 2d c
        value = torch.cat([value_0, value_1], dim=1)  # (b t) 2d c
        key = torch.cat([key]*2, dim=0)   # ( 2 b t) 2d c
        value = torch.cat([value]*2, dim=0)  # (2 b t) 2d c

        
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if attack_mode:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            ref_dict[self.name] = attention_probs
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class NewXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, name, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        self.name = name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        ref_dict={},
        attack_mode=False
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if attack_mode:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            ref_dict[self.name] = attention_probs
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel) -> None:
        super().__init__()
        self.unet = unet

        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            # if torch.__version__ >= '2.0':
            #     default_attn_proc = AttnProcessor2_0()
            # elif is_xformers_available():
            #     default_attn_proc = XFormersAttnProcessor()
            # else:
            #     default_attn_proc = AttnProcessor()
            if name.endswith("attn1.processor"):
                unet_lora_attn_procs[name] = XFormersMVAttnProcessor(name)
            elif name.endswith("attn2.processor"):
                unet_lora_attn_procs[name] = NewXFormersAttnProcessor(name)
            elif name.endswith("attn_joint_mid.processor"):         
                unet_lora_attn_procs[name] = XFormersJointAttnProcessor(name)
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        attack_mode = False,
        **kwargs
    ):
        weight_dtype = self.unet.dtype
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


class Attack():
    def __init__(self, model, iteration, epsilon, num_inference_steps, attention_res, output_dir):
        self.model = model
        self.iteration = iteration
        self.epsilon = epsilon
        self.num_inference_steps = num_inference_steps
        self.attention_res = attention_res
        self.output_dir = output_dir
        self.prepare()

    def aggregate_attention(self, attention_maps: dict, res: int, from_where: List[str], is_cross: bool, select: int) -> torch.Tensor:
        out_mv_attention = []
        out_cd_attention = []
        original_out_mv_attention = []
        original_out_cd_attention = []
        num_pixels = res ** 2
       
        for item in attention_maps:
            if item.split('_')[0] in from_where:
                if attention_maps[item].shape[1] == num_pixels:
                    if item.endswith("attn1.processor"):
                        print(item)
                        out_mv_attention.append(attention_maps[item])
                    elif item.endswith("attn_joint_mid.processor"): 
                        print(item)
                        out_cd_attention.append(attention_maps[item])
        
        for item in attention_maps:
            if item.split('_')[0] in from_where:
                if attention_maps[item].shape[1] == num_pixels:
                    if item.endswith("attn1.processor"):
                        print(item)
                        original_out_mv_attention.append(attention_maps[item].detach())
                    elif item.endswith("attn_joint_mid.processor"): 
                        print(item)
                        original_out_cd_attention.append(attention_maps[item].detach())
    
        out_mv_attention = torch.cat(out_mv_attention, dim=0)
        out_mv_attention = out_mv_attention.sum(0) / out_mv_attention.shape[0]
        out_cd_attention = torch.cat(out_cd_attention, dim=0)
        out_cd_attention = out_cd_attention.sum(0) / out_cd_attention.shape[0]
        return out_mv_attention, out_cd_attention, original_out_mv_attention, original_out_cd_attention
    
    def compute_loss(self, attention_mv, attention_cd, mask, image_adv_latents, image_latents):
        # loss choice 1
        mask_resize = transforms.Resize((self.attention_res, self.attention_res))
        loss_mask = mask_resize(mask)[0]# * 255
        loss_mask = loss_mask.reshape(1, self.attention_res**2)
        loss_mask = loss_mask.repeat(self.attention_res**2, 1)
        loss_mask_mv = loss_mask.repeat(1, self.model.num_views)
        loss_mask_cd = loss_mask.repeat(1, 2)

        loss_attention_mv = torch.sum(loss_mask_mv * attention_mv) / self.model.num_views
        loss_attention_cd = torch.sum(loss_mask_cd * attention_cd) / 2
        
        # loss choice 2
        loss_latent = -torch.mean(torch.pow(image_adv_latents - image_latents, 2))
        
        loss = loss_attention_mv + 0.3 * loss_attention_cd + 1 * loss_latent
        print(f'loss:{loss}, loss_attention_mv:{loss_attention_mv}, loss_attention_cd:{loss_attention_cd}, loss_latent:{loss_latent}')
        return loss

    def prepare(self):
        self.model.unet = RefOnlyNoisedUNet(self.model.unet).eval()

    def __call__(self, image, mask, image_name): 
        image = to_rgb_image(image)
        image = np.array(image)[:, :, :3]
        image = torch.from_numpy(image.transpose((2, 0, 1))).cuda()

        mask = to_rgb_image(mask)
        mask = np.array(mask)[:, :, :3] / 255
        mask = torch.from_numpy(mask.transpose((2, 0, 1))).cuda()
        noise = torch.zeros_like(image, dtype=torch.float32)
        noise.requires_grad = True

        self.model.scheduler.set_timesteps(self.num_inference_steps, device=self.model._execution_device)
        timesteps = self.model.scheduler.timesteps
        for iter in range(self.iteration):
            image_adv = image + noise * mask
            image_adv = torch.clamp(image_adv, 0, 255)
            # image_adv = torch.repeat_interleave(image_adv.unsqueeze(0), self.model, dim=0)
            image_adv_clip = clip_preprocess(image_adv, dtype=next(self.model.image_encoder.parameters()).dtype, save_memory=False)
            image_adv_vae = vae_preprocess(image_adv, dtype=next(self.model.image_encoder.parameters()).dtype, save_memory=False)
            image_adv_embeddings = self.model.image_encoder(image_adv_clip).image_embeds.unsqueeze(1)
            image_adv_latents = self.model.vae.encode(image_adv_vae).latent_dist.mode() * self.model.vae.config.scaling_factor

            with torch.no_grad():
                image_vae = vae_preprocess(image, dtype=next(self.model.image_encoder.parameters()).dtype, save_memory=False)
                image_latents = self.model.vae.encode(image_vae).latent_dist.mode() * self.model.vae.config.scaling_factor

            latent_timestep = timesteps[torch.randperm(timesteps.shape[0])[0]]
            # latent_timestep = timesteps[iter]
            ref_dict = forward(self.model,
                               image_adv_embeddings,
                               image_adv_latents,
                               latent_timestep,
                               num_inference_steps=20,
                               output_type='pt',
                               attack_mode=True)

            adv_attention_maps_mv, adv_attention_maps_cd, original_out_mv_attention, original_out_cd_attention = self.aggregate_attention(
                                                                attention_maps=ref_dict,
                                                                res=self.attention_res,
                                                                from_where=("up", "down", "mid"),
                                                                is_cross=False,
                                                                select=0)

            loss = self.compute_loss(attention_mv=adv_attention_maps_mv, 
                                     attention_cd=adv_attention_maps_cd, 
                                     mask=mask, 
                                     image_adv_latents=image_adv_latents, 
                                     image_latents=image_latents)
            print(f"iter:{iter},loss:{loss}")
            loss.backward()
            grad = noise.grad.sign()
            noise = noise - grad
            noise.data.clamp_(-self.epsilon, self.epsilon)
            noise = noise.clone().detach()
            noise.requires_grad = True
            ref_dict = {}

        # attack result
        with torch.no_grad():
            image_adv = image + noise * mask
            image_adv = torch.clamp(image_adv, 0, 255)
            image_adv_clip = clip_preprocess(image_adv, dtype=next(self.model.image_encoder.parameters()).dtype, save_memory=False)
            image_adv_vae = vae_preprocess(image_adv, dtype=next(self.model.image_encoder.parameters()).dtype, save_memory=False)
            image_adv_embeddings = self.model.image_encoder(image_adv_clip).image_embeds.unsqueeze(1)
            image_adv_latents = self.model.vae.encode(image_adv_vae).latent_dist.mode() * self.model.vae.config.scaling_factor
            result = forward(self.model,
                            image_adv_embeddings,
                            image_adv_latents,
                            None,
                            num_inference_steps=self.num_inference_steps,
                            output_type='pt',
                            attack_mode=False).images
            return image, image_adv, result




def encode_image(model, image, device, num_images_per_prompt, do_classifier_free_guidance):
    dtype = next(model.image_encoder.parameters()).dtype

    image_pt = clip_preprocess(image, dtype=dtype, save_memory=False)
    image_embeddings = model.image_encoder(image_pt).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)

    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(num_images_per_prompt, 1, 1)

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    image_pt = vae_preprocess(image, dtype=dtype, save_memory=False)
    image_latents = model.vae.encode(image_pt).latent_dist.mode() * model.vae.config.scaling_factor

    image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

    if do_classifier_free_guidance:
        image_latents = torch.cat([torch.zeros_like(image_latents), image_latents])

    return image_embeddings, image_latents


def forward(
        model,
        image_embeddings,
        image_latents,
        latent_timesteps,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        normal_cond: Optional[Union[List[PIL.Image.Image], torch.FloatTensor]] = None,
        replace_latents = None,
        attack_mode = False,
):

    # assert len(elevation_cond) == batch_size and len(elevation) == batch_size and len(azimuth) == batch_size
    # camera_embeddings = self.prepare_camera_condition(elevation_cond, elevation, azimuth, do_classifier_free_guidance=do_classifier_free_guidance, num_images_per_prompt=num_images_per_prompt)

    dtype = model.vae.dtype
    do_classifier_free_guidance = guidance_scale != 1.0
    batch_size = model.num_views * 2
    device = model._execution_device
    height = model.unet.config.sample_size * model.vae_scale_factor
    width = model.unet.config.sample_size * model.vae_scale_factor

    camera_embedding = model.camera_embedding.to(dtype)
    camera_embedding = repeat(camera_embedding, "Nv Nce -> (B Nv) Nce", B=batch_size // len(camera_embedding))
    camera_embeddings = model.prepare_camera_embedding(camera_embedding,
                                                      do_classifier_free_guidance=do_classifier_free_guidance,
                                                      num_images_per_prompt=num_images_per_prompt)

    # 4. Prepare timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = model.unet.config.out_channels
    latents = model.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        image_embeddings.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * model.scheduler.order
    with model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = torch.cat([
                latent_model_input, image_latents
            ], dim=1)
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            if attack_mode:
                ref_dict = {}
                model.unet(latent_model_input,
                           latent_timesteps,
                           encoder_hidden_states=image_embeddings,
                           class_labels=camera_embeddings,
                           cross_attention_kwargs=dict(ref_dict=ref_dict, attack_mode=attack_mode),).sample
                return ref_dict
            else:
                ref_dict = {}
                noise_pred = model.unet(latent_model_input,
                                        t,
                                        encoder_hidden_states=image_embeddings,
                                        class_labels=camera_embeddings,
                                        cross_attention_kwargs=dict(ref_dict=ref_dict, attack_mode=attack_mode),).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if replace_latents != None:
        latents = replace_latents
    if not output_type == "latent":
        if num_channels_latents == 8:
            latents = torch.cat([latents[:, :4], latents[:, 4:]], dim=0)

        image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = model.run_safety_checker(image, device, image_embeddings.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = model.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
