#%% Init
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
import PIL.Image
from torchvision.transforms import ToPILImage
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from typing import List, Optional, Union

from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    DiffusionPipeline,
    ImagePipelineOutput,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.pipelines.unclip import UnCLIPTextProjModel
from diffusers.utils.torch_utils import randn_tensor

from imbalance_data import cifar100Imbanlance

class UnCLIPEncoderDecoderPipeline(DiffusionPipeline):
    
    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    feature_extractor: CLIPImageProcessor
    image_encoder: CLIPVisionModelWithProjection
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline.__init__
    def __init__(
        self,
        decoder: UNet2DConditionModel,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        super_res_first: UNet2DModel,
        super_res_last: UNet2DModel,
        decoder_scheduler: UnCLIPScheduler,
        super_res_scheduler: UnCLIPScheduler,
        only_image_embeddings: bool = False,
    ):
        super().__init__()

        self.register_modules(
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )
        self.only_image_embeddings = only_image_embeddings
    
    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents
        
    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline._encode_prompt
    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)
        text_encoder_output = self.text_encoder(text_input_ids.to(device))

        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

    @torch.no_grad()
    def encode_image(self, image: Union[PIL.Image.Image, torch.Tensor]) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        image = image.to(self.device, dtype=self.image_encoder.dtype)
        return self.image_encoder(image).image_embeds

    @torch.no_grad()
    def decode_features(
        self,
        features: torch.Tensor,
        num_inference_steps: int = 25,
        super_res_steps: int = 7,
        guidance_scale: float = 8.0,
        output_type: str = "pil",
    ):
        return self.call(
            embedding=features,
            steps=1,
            decoder_num_inference_steps=num_inference_steps,
            super_res_num_inference_steps=super_res_steps,
            decoder_guidance_scale=guidance_scale,
            output_type=output_type,
            retain_image_tensor=True,
        )[0]['images'][0]  # Return single image
    
    
    @torch.no_grad()
    def call(
        self,
        image: Optional[Union[List[PIL.Image.Image], torch.Tensor]] = None,
        embedding: torch.Tensor = None,
        steps: int = 1,
        decoder_num_inference_steps: int = 25,
        super_res_num_inference_steps: int = 7,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        decoder_latents: Optional[torch.Tensor] = None,
        super_res_latents: Optional[torch.Tensor] = None,
        decoder_guidance_scale: float = 8.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        mean_val: float = 1.0,
        text_feat = 1.0,
        text_feat2 = 1.0,
        retain_image_tensor=False,
        var: float = 0.0,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`List[PIL.Image.Image]` or `torch.Tensor`):
                The images to use for the image interpolation. Only accepts a list of two PIL Images or If you provide a tensor, it needs to comply with the
                configuration of
                [this](https://huggingface.co/fusing/karlo-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json)
                `CLIPImageProcessor` while still having a shape of two in the 0th dimension. Can be left to `None` only when `image_embeddings` are passed.
            steps (`int`, *optional*, defaults to 5):
                The number of interpolation images to generate.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            image_embeddings (`torch.Tensor`, *optional*):
                Pre-defined image embeddings that can be derived from the image encoder. Pre-defined image embeddings
                can be passed for tasks like image interpolations. `image` can the be left to `None`.
            decoder_latents (`torch.Tensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`torch.Tensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        """

        batch_size = steps

        device = self._execution_device
        if embedding is None:
            if isinstance(image, List):
                if len(image) != 2:
                    raise AssertionError(
                        f"Expected 'image' List to be of size 2, but passed 'image' length is {len(image)}"
                    )
                elif not (isinstance(image[0], PIL.Image.Image) and isinstance(image[0], PIL.Image.Image)):
                    raise AssertionError(
                        f"Expected 'image' List to contain PIL.Image.Image, but passed 'image' contents are {type(image[0])} and {type(image[1])}"
                    )
            elif isinstance(image, torch.Tensor):
                if image.shape[0] != 2:
                    raise AssertionError(
                        f"Expected 'image' to be torch.Tensor of shape 2 in 0th dimension, but passed 'image' size is {image.shape[0]}"
                    )
            elif isinstance(image_embeddings, torch.Tensor):
                if image_embeddings.shape[0] != 2:
                    raise AssertionError(
                        f"Expected 'image_embeddings' to be torch.Tensor of shape 2 in 0th dimension, but passed 'image_embeddings' shape is {image_embeddings.shape[0]}"
                    )
            else:
                raise AssertionError(
                    f"Expected 'image' or 'image_embeddings' to be not None with types List[PIL.Image] or torch.Tensor respectively. Received {type(image)} and {type(image_embeddings)} repsectively"
                )

            original_image_embeddings = self._encode_image(
                image=image, device=device, num_images_per_prompt=1, image_embeddings=image_embeddings, mean_val=mean_val
            )
            image_embeddings = []

            for interp_step in torch.linspace(0, 1, steps):
                # temp_image_embeddings = slerp(
                #     interp_step, original_image_embeddings[0], original_image_embeddings[1], mean_val=mean_val, text_feat=text_feat, text_feat2=text_feat2
                # ).unsqueeze(0)
                temp_image_embeddings = original_image_embeddings[1].unsqueeze(0)
                image_embeddings.append(temp_image_embeddings)
            image_embeddings = torch.cat(image_embeddings).to(device)
        else:
            image_embeddings = embedding.repeat_interleave(steps, dim=0).to(device)
        #------------------
        if self.only_image_embeddings:
            return [],image_embeddings
        #------------------
        do_classifier_free_guidance = decoder_guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt=["" for i in range(steps)],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_encoder_hidden_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        if device.type == "mps":
            # HACK: MPS: There is a panic when padding bool tensors,
            # so cast to int tensor for the pad and back to bool afterwards
            text_mask = text_mask.type(torch.int)
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=1)
            decoder_text_mask = decoder_text_mask.type(torch.bool)
        else:
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=True)

        self.decoder_scheduler.set_timesteps(decoder_num_inference_steps, device=device)
        decoder_timesteps_tensor = self.decoder_scheduler.timesteps

        num_channels_latents = self.decoder.config.in_channels
        height = self.decoder.config.sample_size
        width = self.decoder.config.sample_size
        single_noise = self.prepare_latents(
            (1, num_channels_latents, height, width),
            text_encoder_hidden_states.dtype,
            device,
            generator,
            None,
            self.decoder_scheduler,
        )
        decoder_latents = single_noise.repeat((batch_size, 1, 1, 1))
        
        for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
            latent_model_input = torch.cat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents

            noise_pred = self.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            decoder_latents = self.decoder_scheduler.step(
                noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=generator,
            ).prev_sample

        decoder_latents = decoder_latents.clamp(-1, 1)

        image_small = decoder_latents


        self.super_res_scheduler.set_timesteps(super_res_num_inference_steps, device=device)
        super_res_timesteps_tensor = self.super_res_scheduler.timesteps

        channels = self.super_res_first.config.in_channels // 2
        height = self.super_res_first.config.sample_size
        width = self.super_res_first.config.sample_size

        super_res_latents = self.prepare_latents(
            (batch_size, channels, height, width),
            image_small.dtype,
            device,
            generator,
            super_res_latents,
            self.super_res_scheduler,
        )

        if device.type == "mps":
            image_upscaled = F.interpolate(image_small, size=[height, width])
        else:
            interpolate_antialias = {}
            if "antialias" in inspect.signature(F.interpolate).parameters:
                interpolate_antialias["antialias"] = True

            image_upscaled = F.interpolate(
                image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
            )

        for i, t in enumerate(self.progress_bar(super_res_timesteps_tensor)):

            if i == super_res_timesteps_tensor.shape[0] - 1:
                unet = self.super_res_last
            else:
                unet = self.super_res_first

            latent_model_input = torch.cat([super_res_latents, image_upscaled], dim=1)

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
            ).sample

            if i + 1 == super_res_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = super_res_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            super_res_latents = self.super_res_scheduler.step(
                noise_pred, t, super_res_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

        image = super_res_latents
        image = image * 0.5 + 0.5
        image_tensor = image.clamp(0, 1)
        image = image_tensor.clone().cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)
        if retain_image_tensor:
            #return ImagePipelineOutput(images=image, image_tensor=image_tensor), image_embeddings, image_tensor
            return {"images": image, "image_tensor": image_tensor}, image_embeddings, image_tensor

        return ImagePipelineOutput(images=image), image_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = UnCLIPEncoderDecoderPipeline.from_pretrained(
    "kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16
).to(device)

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
    transforms.Normalize(mean=mean, std=std),
])
train_dataset = cifar100Imbanlance.Cifar100Imbanlance(transform=transform_val,
                                                      imbanlance_rate=0.01,
                                                      train=True,
                                                      file_path=os.path.join('data/','cifar-100-python/')
                                                      )
val_dataset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=0.01,
                                                    train=False,
                                                    transform=transform_val,
                                                    file_path=os.path.join('data/','cifar-100-python/')
                                                    )

num_class = 100

from model import ResNet_cifar
model = ResNet_cifar.resnet32(num_class=num_class)
model = torch.nn.DataParallel(model).cuda()
resume = os.path.join('GLMC-CVPR2023', 'output', 'cifar100_ckpt.best.pth.tar')
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, map_location='cuda:0')
    best_acc1 = checkpoint['best_acc1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(resume))


#%% Create datasets of feature vectors
feature_dim = model.module.fc_cb.in_features

fc = model.module.fc_cb

model.eval()
# Get KZ
with torch.no_grad():
    clip_dataset = torch.zeros(0,768).to(device)
    class_dataset = torch.zeros(0)
    model_features_dataset = torch.zeros(0, feature_dim)

    for i in range(len(train_dataset)):
        data, target = train_dataset[i] 
        
        class_dataset = torch.concat((class_dataset, torch.tensor([target])))

        #clip_data = data.float()
        clip_data = data.float() * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
        clip_dataset = torch.concat((clip_dataset, autoencoder.encode_image(ToPILImage()(clip_data))))
        
        data = data.to(device).unsqueeze(0)
        _, _, _, _, features = model.module(data, train=True, extract_features=True)
        model_features_dataset = torch.concat((model_features_dataset, features.cpu()), axis=0)
    class_dataset = class_dataset.to(torch.uint8)

# Generate kz and pi per class dataset
layer = model.module.fc_cb
outputs = layer(model_features_dataset.to(device))
pi_dataset = torch.matmul(outputs - layer.bias, torch.linalg.pinv(layer.weight).T)
pi_dataset = pi_dataset.cpu()
kz_dataset = model_features_dataset - pi_dataset

pi_per_class = {i: torch.zeros(0, pi_dataset.shape[1]) for i in range(num_class)}
kz_per_class = {i: torch.zeros(0, pi_dataset.shape[1]) for i in range(num_class)}

for i in range(pi_dataset.shape[0]):
    c = class_dataset[i].to(torch.uint8).item()
    pi_per_class[c] = torch.concat((pi_per_class[c], pi_dataset[i].unsqueeze(0)), dim=0)
    kz_per_class[c] = torch.concat((kz_per_class[c], kz_dataset[i].unsqueeze(0)), dim=0)

#%% Train translator

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class RealNVPBlock(torch.nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)  # Important!
        self.mask = mask
        self.s_net = MLP(dim, dim)
        self.t_net = MLP(dim, dim)

    def forward(self, x, reverse=False):
        x_masked = x * self.mask
        s = self.s_net(x_masked) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)
        if reverse:
            x = (x - t) * torch.exp(-s)
        else:
            x = x * torch.exp(s) + t
        return x

class InvertibleNN(torch.nn.Module):
    def __init__(self, dim, num_blocks=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            mask = self._create_mask(dim, even=(i % 2 == 0))
            self.blocks.append(RealNVPBlock(dim, mask))

    def _create_mask(self, dim, even=True):
        mask = torch.zeros(dim)
        mask[::2] = 1 if even else 0
        mask[1::2] = 0 if even else 1
        return mask

    def forward(self, x, reverse=False):
        if reverse:
            for block in reversed(self.blocks):
                x = block(x, reverse=True)
        else:
            for block in self.blocks:
                x = block(x)
        return x

input_dim = 256
output_dim = 768
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare data
x = model_features_dataset.to(device)                          # [N, 256]
y = clip_dataset.to(device)                                    # [N, 768]

dataset = TensorDataset(pi_dataset, kz_dataset, clip_dataset, class_dataset)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

n_epochs = 1000

# Initialize model
translator = InvertibleNN(dim=output_dim).to(device)
optimizer = torch.optim.Adam(translator.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

weights = 1 / torch.tensor([(class_dataset == i).sum().item() for i in range(100)])
weights = weights / weights.sum()

pi_stds = torch.stack([pi_per_class[c].std(0) for c in range(num_class)])
pi_means = torch.stack([pi_per_class[c].mean(0) for c in range(num_class)])
kz_stds = torch.stack([kz_per_class[c].std(0) for c in range(num_class)])
kz_means = torch.stack([kz_per_class[c].mean(0) for c in range(num_class)])

# Training loop
for epoch in range(n_epochs):
    translator.train()
    for pi, kz, y, c in loader:
        c = c.long()
        sample_weights = weights[c].to(device)
        kz = (kz - kz_means[c]) / kz_stds[c]
        pi = (pi - pi_means[c]) / pi_stds[c]
        pi, kz, y = pi.to(device), kz.to(device), y.to(device)
        
        x = pi + kz
        x = x.detach()

        pad = torch.zeros(x.shape[0], output_dim - input_dim).to(device)
        x_padded = torch.cat([x, pad], dim=1)    
        
        optimizer.zero_grad()
        out = translator(x_padded)  # Forward: model_features → CLIP
        per_sample_loss = F.mse_loss(out, y, reduction='none')
        while per_sample_loss.ndim > 1:
            per_sample_loss = per_sample_loss.mean(dim=1)
        loss = (per_sample_loss * sample_weights).sum()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

translator.eval()
#%% Test inversion of a translated features vector

def show_image(image, mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)):
    image = prepare_for_showing(image, mean, std)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def prepare_for_showing(image, mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)):
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)
    image = image.float()
    image = image * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
    image = image.permute(1,2,0)
    return image.cpu().detach().numpy()

sample_idx = 0
c = class_dataset[sample_idx].item()
pi = pi_dataset[sample_idx]
kz = kz_dataset[sample_idx]
kz = (kz - kz_means[c]) / kz_stds[c]
pi = (pi - pi_means[c]) / pi_stds[c]
model_features = (pi + kz).unsqueeze(0)
pad = torch.zeros(1, output_dim - input_dim)
padded_model_features = torch.cat([model_features, pad], dim=1).to(device)
clip_features = translator(padded_model_features).half()
generated_image = autoencoder.decode_features(clip_features)
show_image(transform_val(generated_image))
show_image(train_dataset[sample_idx][0])

# %% Test inversion of a new features vector
pi_index = np.random.randint(model_features_dataset.shape[0])
kz_index = np.random.randint(model_features_dataset.shape[0])
c_pi = class_dataset[pi_index].item()
c_kz = class_dataset[kz_index].item()
with torch.no_grad():
    layer = model.module.fc_cb
    features = model_features_dataset[pi_index, :]
    outputs = layer(features.unsqueeze(0).to(device))
    pi = torch.matmul(outputs - layer.bias, torch.linalg.pinv(layer.weight).T)
    pi = pi.cpu()
    
    features = model_features_dataset[kz_index, :]
    outputs = layer(features.unsqueeze(0).to(device))
    kz = features.to(device) - torch.matmul(outputs - layer.bias, torch.linalg.pinv(layer.weight).T)
    kz = kz.cpu()
    #kz = (kz - kz_dataset[c_kz].mean(0)) / kz_dataset[c_kz].std(0)
    #kz = kz * kz_dataset[c_pi].std(0) + kz_dataset[c_pi].mean(0)

kz = (kz - kz_means[c_kz]) / kz_stds[c_kz]
pi = (pi - pi_means[c_pi]) / pi_stds[c_pi]
new_feature = (pi + kz)
pad = torch.zeros(1, output_dim - input_dim)
padded_new_features = torch.cat([new_feature, pad], dim=1).to(device)    
clip_features = translator(padded_new_features).half()
generated_image = autoencoder.decode_features(clip_features)

pi_image = prepare_for_showing(train_dataset[pi_index][0])
kz_image = prepare_for_showing(train_dataset[kz_index][0])
generated_image = prepare_for_showing(transform_val(generated_image))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot images with titles
axes[0].imshow(pi_image, cmap='gray')
axes[0].set_title('PI Image')
axes[0].axis('off')

axes[1].imshow(kz_image, cmap='gray')
axes[1].set_title('KZ Image')
axes[1].axis('off')

axes[2].imshow(generated_image, cmap='gray')
axes[2].set_title('Generated Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()

#%% Create Distance Table

def _energy_distance_empirical(X1, X2):
    E12 = torch.cdist(X1, X2).mean().item()
    E11 = torch.cdist(X1, X1).mean().item()
    E22 = torch.cdist(X2, X2).mean().item()
    return 2 * E12 - E11 - E22

n_classes = 100
kz_distance_table = np.zeros((n_classes, n_classes))
pi_distance_table = np.zeros((n_classes, n_classes))
for c1 in range(n_classes):
    for c2 in range(c1+1, n_classes):
        print(f'{c1}, {c2}')
        X1, X2 = kz_per_class[c1], kz_per_class[c2]
        X1, X2 = X1.to(device), X2.to(device)
        X1 = (X1 - X1.mean(0)) / X1.std(0)
        X2 = (X2 - X2.mean(0)) / X2.std(0)
        distance = _energy_distance_empirical(X1, X2)
        kz_distance_table[c1,c2] = kz_distance_table[c2,c1] = distance
        
        X1, X2 = pi_per_class[c1], pi_per_class[c2]
        X1, X2 = X1.to(device), X2.to(device)
        X1 = (X1 - X1.mean(0)) / X1.std(0)
        X2 = (X2 - X2.mean(0)) / X2.std(0)
        distance = _energy_distance_empirical(X1, X2)
        pi_distance_table[c1,c2] = pi_distance_table[c2,c1] = distance

#%%
c_pi = 3
c_kz_list = kz_distance_table[c_pi].argsort()[1:5]
c_pi_list = pi_distance_table[c_pi].argsort()[0:5]
for _ in range(5):
    
    pi_list_amounts = np.array([pi_per_class[c].shape[0] for c in c_pi_list])
    chosen_c_pi = np.random.choice(c_pi_list, p=pi_list_amounts/sum(pi_list_amounts))
    pi = pi_per_class[chosen_c_pi][np.random.randint(pi_per_class[chosen_c_pi].shape[0])].clone()
    pi = (pi - pi_per_class[chosen_c_pi].mean(0)) / pi_per_class[chosen_c_pi].std(0)
    pi = pi * pi_per_class[c_pi].std(0) + pi_per_class[c_pi].mean(0)
    
    #pi = pi_per_class[c_pi][np.random.randint(pi_per_class[c_pi].shape[0])]
    
    kz_list_amounts = np.array([kz_per_class[c].shape[0] for c in c_kz_list])
    chosen_c_kz = np.random.choice(c_kz_list, p=kz_list_amounts/sum(kz_list_amounts))
    kz = kz_per_class[chosen_c_kz][np.random.randint(kz_per_class[chosen_c_kz].shape[0])].clone()
    kz = (kz - kz_per_class[chosen_c_kz].mean(0)) / kz_per_class[chosen_c_kz].std(0)
    kz = kz * kz_per_class[c_pi].std(0) + kz_per_class[c_pi].mean(0)
    
    #kz_list_amounts = np.array([kz_dataset[c].shape[0] for c in c_kz_list])
    #chosen_c_kz = np.random.choice(c_kz_list, p=kz_list_amounts/sum(kz_list_amounts))
    #kz = kz_per_class[chosen_c_kz][np.random.randint(kz_per_class[chosen_c_kz].shape[0])]
    #kz = (kz - kz_per_class[chosen_c_kz].mean(0)) / kz_per_class[chosen_c_kz].std(0)
    #kz = kz * kz_per_class[c_pi].std(0) + kz_per_class[c_pi].mean(0)
    
    kz = (kz - kz_means[c_kz]) / kz_stds[c_kz]
    pi = (pi - pi_means[c_pi]) / pi_stds[c_pi]
    new_feature = pi + kz

    pad = torch.zeros(1, output_dim - input_dim)
    padded_new_feature = torch.cat([new_feature.unsqueeze(0), pad], dim=1)    
    clip_features = translator(padded_new_feature.to(device)).half()
    generated_image = autoencoder.decode_features(clip_features)
    show_image(transform_val(generated_image))
