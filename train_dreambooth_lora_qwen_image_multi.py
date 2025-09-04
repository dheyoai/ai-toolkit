#!/usr/bin/env python
# coding=utf-8
# Fixed multi-concept DreamBooth LoRA training for Qwen Image with wandb integration

import math
import argparse
import json
import os
import torch
import random
import itertools
import copy
import shutil
from contextlib import nullcontext
from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
from torchvision import transforms
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset, DataLoader
from diffusers import QwenImagePipeline, AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.training_utils import (
    compute_density_for_timestep_sampling, 
    compute_loss_weighting_for_sd3, 
    free_memory,
    offload_models,
    cast_training_params,
    _collate_lora_metadata
)
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from tqdm import tqdm
import logging
import wandb

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-concept DreamBooth LoRA for Qwen Image with wandb")
    parser.add_argument("--concepts_file", type=str, default="concepts.json", help="Path to concepts JSON file")
    parser.add_argument("--output_dir", type=str, default="/shareddata/dheyo/aakashvarma/src/ai-toolkit/output", help="Output directory")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512], help="Image resolutions")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--max_train_steps", type=int, default=8000, help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--linear_rank", type=int, default=32, help="LoRA rank for linear layers")
    parser.add_argument("--linear_alpha", type=int, default=32, help="LoRA alpha for linear layers")
    parser.add_argument("--checkpointing_steps", type=int, default=800, help="Save checkpoint every X steps")
    parser.add_argument("--validation_epochs", type=int, default=50, help="Run validation every X epochs")
    parser.add_argument("--log_steps", type=int, default=10, help="Log metrics every X steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="aa_and_ab_qwen_image_tilora_spl_tokens", help="wandb project name")
    parser.add_argument("--with_prior_preservation", action="store_true", help="Flag to add prior preservation loss")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Weight of prior preservation loss")
    parser.add_argument("--class_data_dir", type=str, default=None, help="Directory for class images")
    parser.add_argument("--class_prompt", type=str, default=None, help="Prompt for class images")
    parser.add_argument("--num_class_images", type=int, default=100, help="Number of class images for prior preservation")
    parser.add_argument("--offload", action="store_true", help="Whether to offload VAE and text encoder to CPU when not used")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit normal weighting")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Std for logit normal weighting")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale for mode weighting")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat training data")
    return parser.parse_args()

class MultiConceptDreamBoothDataset(Dataset):
    """
    Multi-concept dataset that properly balances training data across all concepts.
    Each concept can have different number of images but training will be balanced.
    """
    def __init__(self, instance_data_dirs, instance_prompts, resolutions, repeats=1, 
                 class_data_dir=None, class_prompt=None, num_class_images=None, center_crop=False):
        self.instance_prompts = instance_prompts
        self.resolutions = resolutions
        self.instance_data_dirs = instance_data_dirs
        self.class_data_dir = class_data_dir
        self.class_prompt = class_prompt
        self.num_class_images = num_class_images
        self.repeats = repeats
        self.center_crop = center_crop
        
        # Store all processed data for each concept
        self.concept_data = []
        self.concept_lengths = []
        
        # Process each concept's data
        for concept_idx, (data_dir, instance_prompt) in enumerate(zip(instance_data_dirs, instance_prompts)):
            concept_pixel_values = []
            concept_captions = []
            
            root = Path(data_dir)
            if not root.exists():
                raise ValueError(f"Directory {data_dir} does not exist")
            
            # Get all image files
            image_paths = [p for p in root.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            if not image_paths:
                raise ValueError(f"No images found in {data_dir}")
            
            logger.info(f"Found {len(image_paths)} images for concept {concept_idx}: {instance_prompt}")
            
            # Process images with transforms
            for img_path in image_paths:
                # Load and preprocess image
                image = Image.open(img_path)
                image = exif_transpose(image)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                
                # Apply transforms - using first resolution for now
                resolution = resolutions[0]
                image = self._apply_transforms(image, resolution)
                
                # Load caption if exists, otherwise use instance prompt
                caption_path = img_path.with_suffix('.txt')
                if caption_path.exists():
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    caption = caption if caption else instance_prompt
                else:
                    caption = instance_prompt
                
                # Apply repeats - extend data multiple times if needed
                for _ in range(repeats):
                    concept_pixel_values.append(image)
                    concept_captions.append(caption)
            
            self.concept_data.append({
                'pixel_values': concept_pixel_values,
                'captions': concept_captions
            })
            self.concept_lengths.append(len(concept_pixel_values))
        
        # Calculate total length - use max to ensure all concepts get equal representation
        self._length = max(self.concept_lengths)
        logger.info(f"Dataset length set to {self._length} (max across {len(self.concept_data)} concepts)")
        
        # Handle class images for prior preservation
        self.class_images = []
        self.class_captions = []
        if class_data_dir is not None and class_prompt is not None:
            self._setup_class_images(class_data_dir, class_prompt, num_class_images)
    
    def _apply_transforms(self, image, resolution):
        """Apply image transforms consistently"""
        # First resize
        train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        image = train_resize(image)
        
        # Then crop
        if self.center_crop:
            train_crop = transforms.CenterCrop(resolution)
            image = train_crop(image)
        else:
            # Random crop
            if image.width > resolution or image.height > resolution:
                y1, x1, h, w = transforms.RandomCrop.get_params(image, (resolution, resolution))
                image = crop(image, y1, x1, h, w)
        
        # Optional random flip
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return transform(image)
    
    def _setup_class_images(self, class_data_dir, class_prompt, num_class_images):
        """Setup class images for prior preservation"""
        class_root = Path(class_data_dir)
        class_root.mkdir(parents=True, exist_ok=True)
        class_image_paths = [p for p in class_root.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if class_image_paths:
            # Process existing class images
            for img_path in class_image_paths[:num_class_images if num_class_images else len(class_image_paths)]:
                image = Image.open(img_path)
                image = exif_transpose(image)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                
                processed_image = self._apply_transforms(image, self.resolutions[0])
                self.class_images.append(processed_image)
                self.class_captions.append(class_prompt)
            
            self.num_class_images = len(self.class_images)
            # Update length to account for class images
            self._length = max(self._length, self.num_class_images)
            logger.info(f"Loaded {self.num_class_images} class images")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        """
        Returns a batch containing one sample from each concept plus class image if enabled.
        This ensures balanced training across all concepts.
        """
        example = {}
        
        # Get one sample from each concept (cycling through if needed)
        for concept_idx, concept_data in enumerate(self.concept_data):
            # Use modulo to cycle through concept data if index exceeds concept length
            data_idx = index % len(concept_data['pixel_values'])
            example[f"instance_images_{concept_idx}"] = concept_data['pixel_values'][data_idx]
            example[f"instance_prompt_{concept_idx}"] = concept_data['captions'][data_idx]
        
        # Add class image if prior preservation is enabled
        if self.class_images:
            class_idx = index % self.num_class_images
            example["class_images"] = self.class_images[class_idx]
            example["class_prompt"] = self.class_captions[class_idx]
        
        return example

def collate_fn(num_concepts, examples, with_prior_preservation=False):
    """
    Collate function that properly handles multi-concept batching.
    Creates separate batches for each concept and optionally class images.
    """
    all_pixel_values = []
    all_prompts = []
    
    # Collect instance data from all concepts
    for concept_idx in range(num_concepts):
        concept_pixels = [example[f"instance_images_{concept_idx}"] for example in examples]
        concept_prompts = [example[f"instance_prompt_{concept_idx}"] for example in examples]
        
        all_pixel_values.extend(concept_pixels)
        all_prompts.extend(concept_prompts)
    
    # Add class images if prior preservation is enabled
    if with_prior_preservation and "class_images" in examples[0]:
        class_pixels = [example["class_images"] for example in examples]
        class_prompts = [example["class_prompt"] for example in examples]
        
        all_pixel_values.extend(class_pixels)
        all_prompts.extend(class_prompts)
    
    # Stack pixel values and add frame dimension for Qwen
    pixel_values = torch.stack(all_pixel_values)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)  # Add num_frames dimension: [bs, c, 1, h, w]
    
    return {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "prompts": all_prompts,
    }

def unwrap_model(model):
    """Utility to unwrap model from accelerator and compilation"""
    # Handle compiled models if torch.compile was used
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model

def save_model_card(output_dir, images, base_model, instance_prompts, validation_prompts):
    """Generate and save a model card with training details"""
    widget_dict = []
    if images:
        for i, image_set in enumerate(images):
            if image_set:  # Check if image_set is not empty
                for j, image in enumerate(image_set):
                    image_name = f"image_{i}_{j}.png"
                    image.save(os.path.join(output_dir, image_name))
                    widget_dict.append({
                        "text": validation_prompts[i] if i < len(validation_prompts) else instance_prompts[i],
                        "output": {"url": image_name}
                    })

    model_description = f"""
# Multi-Concept DreamBooth LoRA - {Path(output_dir).name}

## Model description
These are multi-concept DreamBooth LoRA weights for {base_model}, trained on multiple subjects simultaneously.

## Concepts trained
{chr(10).join([f"- {prompt}" for prompt in instance_prompts])}

## Usage
```python
import torch
from diffusers import QwenImagePipeline

pipe = QwenImagePipeline.from_pretrained("{base_model}", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("{output_dir}")

# Generate images using your trained concepts
image = pipe("{instance_prompts[0] if instance_prompts else 'your prompt here'}").images[0]
image.save("output.png")
```
"""
    
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_description)

def log_validation(pipeline, accelerator, concepts, weight_dtype, output_dir, epoch, is_final=False):
    """
    Run validation by generating images for each concept.
    Properly manages memory and device placement.
    """
    logger.info(f"Running validation at epoch {epoch}")
    
    # Move pipeline to device with proper dtype
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    
    images_sets = []
    
    # Generate images for each concept
    for concept_idx, concept in enumerate(concepts):
        prompt = concept["validation_prompt"]
        num_images = concept.get("validation_number_images", 1)
        
        logger.info(f"Generating {num_images} validation images for concept {concept_idx}: {prompt}")
        
        # Encode prompt once
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                prompt=prompt, 
                max_sequence_length=512
            )
        
        # Generate images
        generator = torch.Generator(device=accelerator.device).manual_seed(42 + concept_idx)
        concept_images = []
        
        for img_idx in range(num_images):
            with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                try:
                    image = pipeline(
                        prompt_embeds=prompt_embeds,
                        prompt_embeds_mask=prompt_embeds_mask,
                        generator=generator,
                        guidance_scale=4.0,
                        num_inference_steps=25
                    ).images[0]
                    concept_images.append(image)
                    
                    # Save validation image
                    if accelerator.is_main_process:
                        phase = "final" if is_final else "validation"
                        image_name = f"{phase}_concept_{concept_idx}_img_{img_idx}_epoch_{epoch}.png"
                        image.save(os.path.join(output_dir, image_name))
                
                except Exception as e:
                    logger.warning(f"Failed to generate validation image {img_idx} for concept {concept_idx}: {e}")
                    # Create a black placeholder image
                    placeholder = Image.new('RGB', (512, 512), color='black')
                    concept_images.append(placeholder)
        
        images_sets.append(concept_images)
    
    # Log to wandb if available
    if accelerator.is_main_process and wandb.run is not None:
        wandb_images = []
        for concept_idx, (concept_images, concept) in enumerate(zip(images_sets, concepts)):
            for img_idx, image in enumerate(concept_images):
                wandb_images.append(
                    wandb.Image(
                        image, 
                        caption=f"Concept {concept_idx}: {concept['validation_prompt']}"
                    )
                )
        
        phase = "final_validation" if is_final else "validation"
        wandb.log({phase: wandb_images}, step=epoch)
    
    # Clean up memory
    del pipeline
    free_memory()
    
    return images_sets

def main(args):
    # Set random seeds for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Initialize accelerator with proper configuration
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if wandb.run else None,
    )
    
    # Initialize wandb on main process only
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Load concepts configuration
    with open(args.concepts_file, 'r', encoding='utf-8') as f:
        concepts = json.load(f)

    # Extract concept data
    instance_data_dirs = [c["instance_data_dir"] for c in concepts]
    instance_prompts = [c["instance_prompt"] for c in concepts]
    validation_prompts = [c["validation_prompt"] for c in concepts]
    
    logger.info(f"Training {len(concepts)} concepts: {instance_prompts}")

    # Validate prior preservation arguments
    if args.with_prior_preservation and (args.class_data_dir is None or args.class_prompt is None):
        raise ValueError("Must specify --class_data_dir and --class_prompt with --with_prior_preservation")

    # Determine weight dtype based on mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load models with proper memory management
    model_id = "Qwen/Qwen-Image"
    logger.info(f"Loading models from {model_id}")
    
    # Initialize pipeline components
    pipeline = QwenImagePipeline.from_pretrained(model_id, torch_dtype=weight_dtype)
    
    # Setup model offloading strategy
    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    
    # Load and setup components
    vae = pipeline.vae.to(**to_kwargs)
    transformer = pipeline.transformer.to(accelerator.device, dtype=weight_dtype)
    scheduler = copy.deepcopy(pipeline.scheduler)  # Keep a copy for training
    
    # Calculate VAE scaling factors
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device, dtype=weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device, dtype=weight_dtype)

    # Freeze non-trainable components
    vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=args.linear_rank,
        lora_alpha=args.linear_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        init_lora_weights="gaussian"
    )
    transformer.add_adapter(lora_config)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Create dataset with proper multi-concept handling
    train_dataset = MultiConceptDreamBoothDataset(
        instance_data_dirs=instance_data_dirs,
        instance_prompts=instance_prompts,
        resolutions=args.resolutions,
        repeats=args.repeats,
        class_data_dir=args.class_data_dir,
        class_prompt=args.class_prompt,
        num_class_images=args.num_class_images,
        center_crop=False  # Use random crop for better augmentation
    )
    
    # Create dataloader with proper collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(len(instance_data_dirs), examples, args.with_prior_preservation),
        num_workers=0,  # Keep at 0 to avoid multiprocessing issues
        pin_memory=True
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0001,
        eps=1e-8
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)

    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Setup checkpoint saving and loading hooks
    def save_model_hook(models, weights, output_dir):
        """Custom save hook for LoRA weights"""
        if accelerator.is_main_process:
            transformer_lora_layers = None
            modules_to_save = {}
            
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                
                # Pop weights to avoid saving again
                if weights:
                    weights.pop()
            
            # Save LoRA weights
            pipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        """Custom load hook for LoRA weights"""
        transformer_ = None
        
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
        
        if transformer_ is not None:
            # Load LoRA state dict
            lora_state_dict = pipeline.lora_state_dict(input_dir)
            transformer_state_dict = {
                f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() 
                if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            
            if incompatible_keys is not None:
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading LoRA: {unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Handle checkpoint resuming
    global_step = 0
    first_epoch = 0
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get most recent checkpoint
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        
        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    # Cast training parameters to appropriate precision
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # Helper function for sigma calculation
    def get_sigmas(timesteps, n_dim=5, dtype=torch.bfloat16):
        """Calculate sigmas for flow matching"""
        sigmas = scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        
        step_indices = []
        for t in timesteps:
            matches = (schedule_timesteps == t).nonzero()
            if matches.numel() > 0:
                step_indices.append(matches[0].item())
            else:
                # Find closest timestep if exact match not found
                closest_idx = (schedule_timesteps - t).abs().argmin().item()
                step_indices.append(closest_idx)
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Training setup
    logger.info("***** Running training *****")
    logger.info(f"  Num concepts = {len(concepts)}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {args.train_batch_size * accelerator.num_processes}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Training loop
    progress_bar = tqdm(total=args.max_train_steps, desc="Steps", disable=not accelerator.is_main_process)
    progress_bar.update(global_step)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(transformer):
                # Get batch data
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                prompts = batch["prompts"]
                
                # Encode text prompts
                prompt_embeds_list = []
                prompt_embeds_mask_list = []
                
                with torch.no_grad():
                    # Use offload context for text encoding if enabled
                    with offload_models(pipeline, device=accelerator.device, offload=args.offload):
                        for prompt in prompts:
                            embeds, mask = pipeline.encode_prompt(prompt=prompt, max_sequence_length=512)
                            prompt_embeds_list.append(embeds)
                            prompt_embeds_mask_list.append(mask)

                # Pad embeddings to same sequence length for batching
                max_seq_len = max(mask.shape[1] for mask in prompt_embeds_mask_list)
                padded_embeds = []
                padded_masks = []
                
                for embeds, mask in zip(prompt_embeds_list, prompt_embeds_mask_list):
                    # Pad embeddings
                    pad_length = max_seq_len - embeds.shape[1]
                    if pad_length > 0:
                        padded_embed = torch.nn.functional.pad(embeds, (0, 0, 0, pad_length), value=0)
                        padded_mask = torch.nn.functional.pad(mask, (0, pad_length), value=0)
                    else:
                        padded_embed = embeds
                        padded_mask = mask
                    
                    padded_embeds.append(padded_embed)
                    padded_masks.append(padded_mask)
                
                # Stack all embeddings and masks
                prompt_embeds = torch.cat(padded_embeds, dim=0).to(accelerator.device, dtype=weight_dtype)
                prompt_embeds_mask = torch.cat(padded_masks, dim=0).to(accelerator.device)

                # Encode pixel values to latents with proper offloading
                with offload_models(vae, device=accelerator.device, offload=args.offload):
                    pixel_values = pixel_values.to(dtype=vae.dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                
                # Normalize latents using pre-computed statistics
                model_input = (latents - latents_mean) * latents_std
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise for flow matching
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample timesteps with proper weighting scheme
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * scheduler.config.num_train_timesteps).long()
                timesteps = scheduler.timesteps[indices].to(device=model_input.device)

                # Apply flow matching noise addition
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Prepare input for transformer (pack latents)
                noisy_model_input_packed = noisy_model_input.permute(0, 2, 1, 3, 4)
                packed_noisy = pipeline._pack_latents(
                    noisy_model_input_packed, 
                    bsz, 
                    model_input.shape[1], 
                    model_input.shape[3], 
                    model_input.shape[4]
                )
                
                # Calculate image shapes for transformer input
                latent_height = model_input.shape[3]
                latent_width = model_input.shape[4]
                img_shapes = [(1, latent_height // 2, latent_width // 2)] * bsz
                
                # Calculate text sequence lengths
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).long().tolist()
                
                # Forward pass through transformer
                model_pred = transformer(
                    hidden_states=packed_noisy,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    timestep=timesteps / 1000,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                
                # Unpack transformer output
                model_pred = pipeline._unpack_latents(
                    model_pred, 
                    args.resolutions[0], 
                    args.resolutions[0], 
                    vae_scale_factor
                )

                # Calculate loss weighting
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, 
                    sigmas=sigmas
                )
                
                # Flow matching target (velocity)
                target = noise - model_input
                
                # Handle prior preservation loss if enabled
                if args.with_prior_preservation:
                    # Calculate how many samples are instance vs class based on actual batch composition
                    # In multi-concept training, we have: batch_size * num_concepts instance samples + batch_size class samples
                    num_concepts = len(instance_data_dirs)
                    batch_size = args.train_batch_size
                    num_instance_samples = batch_size * num_concepts
                    num_class_samples = batch_size
                    
                    # Ensure we have the expected number of samples
                    if model_pred.shape[0] != (num_instance_samples + num_class_samples):
                        logger.warning(f"Expected {num_instance_samples + num_class_samples} samples, got {model_pred.shape[0]}")
                        # Fallback to simple split if counts don't match
                        num_instance_samples = model_pred.shape[0] // 2
                        num_class_samples = model_pred.shape[0] - num_instance_samples
                    
                    # Split predictions and targets for instance vs class
                    model_pred_instance = model_pred[:num_instance_samples]
                    model_pred_prior = model_pred[num_instance_samples:num_instance_samples + num_class_samples]
                    target_instance = target[:num_instance_samples]
                    target_prior = target[num_instance_samples:num_instance_samples + num_class_samples]
                    weighting_instance = weighting[:num_instance_samples]
                    weighting_prior = weighting[num_instance_samples:num_instance_samples + num_class_samples]
                    
                    # Compute instance loss (all concepts combined)
                    instance_loss = torch.mean(
                        (weighting_instance.float() * (model_pred_instance.float() - target_instance.float()) ** 2).reshape(target_instance.shape[0], -1),
                        1
                    ).mean()
                    
                    # Compute prior preservation loss
                    prior_loss = torch.mean(
                        (weighting_prior.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(target_prior.shape[0], -1),
                        1
                    ).mean()
                    
                    # Combine losses
                    loss = instance_loss + args.prior_loss_weight * prior_loss
                else:
                    # Standard loss calculation
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1
                    ).mean()

                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    params_to_clip = [p for p in transformer.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress and logging
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                # Prepare logs
                logs = {
                    "loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step
                }
                progress_bar.set_postfix(**logs)

                # Log to wandb
                if global_step % args.log_steps == 0 and accelerator.is_main_process and wandb.run is not None:
                    wandb.log(logs, step=global_step)

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Clean up old checkpoints if limit is set
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]
                                
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint_path = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint_path)
                                    logger.info(f"Removed old checkpoint: {removing_checkpoint}")
                        
                        # Save new checkpoint
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint at step {global_step}")

            # Break if max steps reached
            if global_step >= args.max_train_steps:
                break
        
        # Run validation at the end of each epoch if specified
        if (epoch % args.validation_epochs == 0 and epoch > 0) and accelerator.is_main_process:
            logger.info(f"Running validation at epoch {epoch}")
            validation_images = log_validation(
                pipeline, accelerator, concepts, weight_dtype, 
                args.output_dir, epoch, is_final=False
            )

    # Final model saving and validation
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save final LoRA weights
        transformer_unwrapped = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer_unwrapped)
        
        pipeline.save_lora_weights(
            args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
        logger.info(f"Saved final LoRA weights to {args.output_dir}")
        
        # Run final validation
        logger.info("Running final validation")
        final_images = log_validation(
            pipeline, accelerator, concepts, weight_dtype,
            args.output_dir, args.num_train_epochs, is_final=True
        )
        
        # Save model card
        save_model_card(
            args.output_dir, 
            final_images, 
            model_id, 
            instance_prompts, 
            validation_prompts
        )
        logger.info("Saved model card")
        
        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()

    # Clean up
    accelerator.end_training()
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
