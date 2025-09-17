import copy
import json
import math
import weakref
import os
import re
import fnmatch
import sys
from collections import OrderedDict
from typing import List, Optional, Dict, Type, Union, Any, Literal
import torch
import peft
from diffusers import UNet2DConditionModel, PixArtTransformer2DModel, AuraFlowTransformer2DModel, WanTransformer3DModel
from transformers import CLIPTextModel

from .config_modules import NetworkConfig
from .lorm import count_parameters
# FIX: Added 'Module' to the import from network_mixins
from .network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin, Network, Module, ExtractMode

from toolkit.kohya_lora import LoRANetwork # Parent class for non-AdaLoRA network types
from toolkit.models.DoRA import DoRAModule
from typing import TYPE_CHECKING

from pathlib import Path
from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType # Import TaskType
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

# NEW: Helper function to wrap QwenImageTransformer2DModel's forward
# This function must be defined here, as it's used in LoRASpecialNetwork.__init__
def _qwen_unet_forward_wrapper(original_forward_bound_to_instance):
    """
    A wrapper for QwenImageTransformer2DModel.forward to filter out unexpected 'input_ids' keyword argument.
    `original_forward_bound_to_instance` is already `self.original_unet_ref.forward.__get__(self.original_unet_ref)`
    """
    def forward_wrapper(instance, *args, **kwargs):
        kwargs.pop("input_ids", None) # Explicitly remove 'input_ids'
        return original_forward_bound_to_instance(*args, **kwargs)
    return forward_wrapper

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
    'QLinear',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv',
    'QConv2d',
]

class LoRAModule(ToolkitModuleMixin, ExtractableModuleMixin, torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
            self,
            lora_name,
            org_module: torch.nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            network: 'LoRASpecialNetwork' = None,
            use_bias: bool = False,
            **kwargs
    ):
        self.can_merge_in = True
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.org_module_ref = weakref.ref(org_module)
        self.scalar = torch.tensor(1.0, device=org_module.weight.device)
        # check if parent has bias. if not force use_bias to False
        if org_module.bias is None:
            use_bias = False

        if org_module.__class__.__name__ in CONV_MODULES:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ in CONV_MODULES:
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=use_bias)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=use_bias)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier: Union[float, List[float]] = multiplier
        # wrap the original module so it doesn't get weights updated
        self.org_module = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward


class LoRASpecialNetwork(ToolkitNetworkMixin, LoRANetwork):
    NUM_OF_BLOCKS = 12

    UNET_TARGET_REPLACE_MODULE = ["UNet2DConditionModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["UNet2DConditionModel"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    PEFT_PREFIX_UNET = "unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
            self,
            text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
            unet,
            multiplier: float = 1.0,
            lora_dim: int = 4,
            alpha: float = 1,
            dropout: Optional[float] = None,
            rank_dropout: Optional[float] = None,
            module_dropout: Optional[float] = None,
            conv_lora_dim: Optional[int] = None,
            conv_alpha: Optional[float] = None,
            block_dims: Optional[List[int]] = None,
            block_alphas: Optional[List[float]] = None,
            conv_block_dims: Optional[List[int]] = None,
            conv_block_alphas: Optional[List[float]] = None,
            modules_dim: Optional[Dict[str, int]] = None,
            modules_alpha: Optional[Dict[str, int]] = None,
            module_class: Type[object] = LoRAModule,
            varbose: Optional[bool] = False,
            train_text_encoder: Optional[bool] = True,
            use_text_encoder_1: bool = True,
            use_text_encoder_2: bool = True,
            train_unet: Optional[bool] = True,
            is_sdxl=False,
            is_v2=False,
            is_v3=False,
            is_pixart: bool = False,
            is_auraflow: bool = False,
            is_flux: bool = False,
            is_lumina2: bool = False,
            use_bias: bool = False,
            is_lorm=False,
            ignore_if_contains = None,
            only_if_contains = None,
            parameter_threshold: float = 0.0,
            attn_only: bool = False,
            target_lin_modules=LoRANetwork.UNET_TARGET_REPLACE_MODULE,
            target_conv_modules=LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
            network_type: str = "lora",
            full_train_in_out: bool = False,
            transformer_only: bool = False,
            peft_format: bool = False,
            is_assistant_adapter: bool = False,
            is_transformer: bool = False,
            base_model: 'StableDiffusion' = None,
            network_config: NetworkConfig = None, # Passed directly by BaseSDTrainProcess
            **kwargs
    ) -> None:
        torch.nn.Module.__init__(self)

        # CRITICAL FIX: Ensure network_config is assigned IMMEDIATELY and unconditionally
        self.network_config = network_config 

        # --- PEFT-specific attribute initializations (always add these) ---
        self.peft_adapted_text_encoders = torch.nn.ModuleList()
        self.peft_adapted_unet = None
        self.peft_adapter_name = "default_adalora_adapter" 
        self.adalora_config = None # Initialize adalora_config as None at class level
        
        # Store original model references (before they get wrapped)
        self.original_unet_ref = unet
        self.original_text_encoder_refs = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # Call ToolkitNetworkMixin init (this handles common attributes and sets internal flags like self.network_type)
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_sdxl=is_sdxl,
            is_v2=is_v2,
            is_lorm=is_lorm,
            is_v3=is_v3,
            is_pixart=is_pixart,
            is_auraflow=is_auraflow,
            is_flux=is_flux,
            is_lumina2=is_lumina2,
            network_type=network_type, # Pass network_type to parent
            base_model_ref=weakref.ref(base_model) if base_model is not None else None,
            network_config=self.network_config, # Pass the now-guaranteed network_config to parent
            **kwargs
        )
        
        # These attributes are now correctly set by ToolkitNetworkMixin.__init__ or from constructor arguments
        self.ignore_if_contains = ignore_if_contains if ignore_if_contains is not None else []
        self.transformer_only = transformer_only
        self.base_model_ref = weakref.ref(base_model) if base_model is not None else None 
        self.only_if_contains: Union[List, str, None] = only_if_contains 
        
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False
        self._multiplier: float = 1.0
        self.is_active: bool = False
        self.torch_multiplier = None
        self.multiplier = multiplier # Triggers state updates
        self.is_sdxl = is_sdxl
        self.is_v2 = is_v2
        self.is_v3 = is_v3
        self.is_pixart = is_pixart
        self.is_auraflow = is_auraflow
        self.is_flux = is_flux
        self.is_lumina2 = is_lumina2
        self.network_type = network_type # Ensure this is explicitly set for the instance
        self.is_assistant_adapter = is_assistant_adapter
        self.full_train_in_out = full_train_in_out 
        self.peft_format = peft_format 
        self.is_transformer = is_transformer 


        if self.network_type.lower() == "dora":
            self.module_class = DoRAModule
            module_class = DoRAModule
        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule          
        # --- START OF ADALORA CONDITIONAL BRANCH ---
        elif self.network_type.lower() == "adalora":
            # For AdaLoRA, we won't be using LoRANetwork's module lists or _lora_map directly.
            self.text_encoder_loras = torch.nn.ModuleList() 
            self.unet_loras = torch.nn.ModuleList()       
            self._lora_map = {} # Clear the internal map for LoRA modules (important for inherited LoRANetwork methods)
            
            # --- Get total_training_steps from network_config ---
            total_training_steps = self.network_config.adalora_total_step
            if total_training_steps is None:
                 print("Warning: adalora_total_step not set in network_config, falling back to sum of tinit/tfinal/deltaT. Ensure total_step is properly set in config.")
                 total_training_steps = (self.network_config.adalora_tinit + self.network_config.adalora_tfinal + (self.network_config.adalora_deltaT * 10)) # Reasonable fallback
                 
            adalora_config_kwargs = {
                "peft_type": "ADALORA",
                "r": network_config.linear, 
                "lora_alpha": network_config.linear_alpha,
                "lora_dropout": network_config.dropout if network_config.dropout is not None else 0.0,
                "bias": "none",
                "init_r": network_config.adalora_init_r,
                "target_r": network_config.adalora_target_r,
                "tinit": network_config.adalora_tinit,
                "tfinal": network_config.adalora_tfinal,
                "deltaT": network_config.adalora_deltaT,
                "beta1": network_config.adalora_beta1,
                "beta2": network_config.adalora_beta2,
                "orth_reg_weight": network_config.adalora_orth_reg_weight,
                "total_step": total_training_steps, # Use the resolved total_training_steps
                "target_modules": kwargs.get('target_modules', None) 
            }
            
            # --- Specific target modules for Qwen's UNet based on your debug output ---
            qwen_unet_target_modules = [
                "attn.to_q", "attn.to_k", "attn.to_v", "attn.add_k_proj", "attn.add_v_proj", "attn.add_q_proj",
                "attn.to_out.0", # Explicitly targeting the Linear layer inside the ModuleList
                "attn.to_add_out",
                "img_mlp.net.0.proj", "img_mlp.net.2",
                "txt_mlp.net.0.proj", "txt_mlp.net.2",
                "proj_out", # Top-level UNet output projection
                "img_mod.1", # Assuming index 1 within Sequential is the Linear layer
                "txt_mod.1", # Assuming index 1 within Sequential is the Linear layer
            ]
            
            # --- Specific target modules for Qwen's Text Encoder based on your debug output ---
            qwen_text_encoder_target_modules = [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ]

            # Assign UNet target modules
            if network_config.transformer_only:
                adalora_config_kwargs['target_modules'] = qwen_unet_target_modules
            else:
                adalora_config_kwargs['target_modules'] = None 

            # CRITICAL: Create and store the AdaLoraConfig instance on `self.adalora_config`
            self.adalora_config = AdaLoraConfig(**adalora_config_kwargs)
            
            # Adapt Text Encoder(s) (This part of __init__ is now just for initial setup, actual PEFT wrapping happens in apply_to)
            # We store references, actual PEFT models are created in apply_to
            for i, te in enumerate(self.original_text_encoder_refs):
                if not train_text_encoder or (not use_text_encoder_1 and i == 0) or \
                   (not use_text_encoder_2 and i == 1):
                    # Store original TE in the list if not training them
                    self.peft_adapted_text_encoders.append(te)
                    continue
                # For training TEs, we don't apply PEFT here, just prepare
                # The actual PEFT-wrapped TE will be added to self.peft_adapted_text_encoders in apply_to
                # For now, just ensure the list is ready.
            
            # Adapt UNet (This part of __init__ is now just for initial setup, actual PEFT wrapping happens in apply_to)
            if not train_unet:
                self.peft_adapted_unet = unet # Keep original if not training
            
            # base_model references are updated in apply_to
            
            # Handle full_train_in_out for AdaLoRA by directly modifying the original base_model parts
            # These attributes need to be stored on self for prepare_optimizer_params
            if self.full_train_in_out:
                base_model_arch = base_model.arch if base_model else None
                if is_pixart or is_auraflow or is_flux or (base_model_arch == "wan21"):
                    transformer_root_module = self.original_unet_ref 
                    self.transformer_pos_embed = copy.deepcopy(transformer_root_module.pos_embed)
                    self.transformer_proj_out = copy.deepcopy(transformer_root_module.proj_out)
                    # Note: These are attributes on `self` for parameter collection.
                else:
                    unet_root_module = self.original_unet_ref 
                    self.unet_conv_in = copy.deepcopy(unet_root_module.conv_in)
                    self.unet_conv_out = copy.deepcopy(unet_root_module.conv_out)
                    # Same note as above.
                    
        # --- END OF ADALORA CONDITIONAL BRANCH ---
        # --- START OF ORIGINAL LORANETWORK INITIALIZATION BRANCH ---
        else: # This 'else' block contains the original LoRANetwork's custom initialization logic.
            super().__init__(
                text_encoder=text_encoder,
                unet=unet,
                multiplier=multiplier,
                lora_dim=lora_dim,
                alpha=alpha,
                dropout=dropout,
                rank_dropout=rank_dropout,
                module_dropout=module_dropout,
                conv_lora_dim=conv_lora_dim,
                conv_alpha=conv_alpha,
                block_dims=block_dims,
                block_alphas=block_alphas,
                conv_block_dims=conv_block_dims,
                conv_block_alphas=conv_block_alphas,
                modules_dim=modules_dim,
                modules_alpha=modules_alpha,
                module_class=module_class,
                varbose=varbose,
                train_text_encoder=train_text_encoder,
                use_text_encoder_1=use_text_encoder_1,
                use_text_encoder_2=use_text_encoder_2,
                train_unet=train_unet,
                is_sdxl=is_sdxl,
                is_v2=is_v2,
                is_v3=is_v3,
                is_pixart=is_pixart,
                is_auraflow=is_auraflow,
                is_flux=is_flux,
                is_lumina2=is_lumina2,
                use_bias=use_bias,
                is_lorm=is_lorm,
                ignore_if_contains=ignore_if_contains,
                only_if_contains=only_if_contains,
                parameter_threshold=parameter_threshold,
                attn_only=attn_only,
                target_lin_modules=target_lin_modules,
                target_conv_modules=target_conv_modules,
                network_type=network_type,
                full_train_in_out=full_train_in_out,
                transformer_only=transformer_only,
                peft_format=peft_format,
                is_assistant_adapter=is_assistant_adapter,
                is_transformer=is_transformer,
                base_model=base_model,
                network_config=network_config,
                **kwargs
            )

            # --- ORIGINAL LORA MODULE CREATION LOGIC (RESTORED AND CONSOLIDATED) ---
            # This block now runs ONLY for non-AdaLoRA network types.

            # create module instances (ORIGINAL create_modules function - moved inside `else`)
            def create_modules(
                    is_unet: bool,
                    text_encoder_idx: Optional[int],
                    root_module: torch.nn.Module,
                    target_replace_modules: List[str], # Changed to List[str]
            ) -> List[LoRAModule]:

                def process_selective_layers ():
                    expanded_only_if_contains = []
                    current_base_model = self.base_model_ref() if self.base_model_ref else None
                    if current_base_model and hasattr(current_base_model, 'get_transformer_block_names'):
                        transformer_block_handles = current_base_model.get_transformer_block_names()
                    else:
                        transformer_block_handles = [] 

                    for layer_pattern in self.only_if_contains: # Renamed `layer` to `layer_pattern` to avoid confusion
                        for handle in transformer_block_handles:
                            module_list = getattr(root_module, handle) 
                            # Iterate through named_modules to find actual module paths
                            for name, _ in module_list.named_modules():
                                try:
                                    if re.search(layer_pattern, name):
                                        # Use the full path for resolution if needed, or just the leaf name
                                        resolved_modules.append(name.split('.')[-1]) # Or full name if needed
                                except re.error:
                                    pass # Pattern was not a valid regex
                            
                    self.only_if_contains = list(set(expanded_only_if_contains))
                    print(self.only_if_contains)

                if isinstance(self.only_if_contains, List) and self.only_if_contains: 
                    process_selective_layers()
                elif isinstance(self.only_if_contains, str) and Path(self.only_if_contains).exists():
                    layer_regex_path = Path(self.only_if_contains)
                    self.only_if_contains = layer_regex_path.read_text(encoding="utf-8").splitlines()
                    process_selective_layers()
                
                unet_prefix = self.LORA_PREFIX_UNET
                if self.peft_format:
                    unet_prefix = self.PEFT_PREFIX_UNET
                if self.is_pixart or self.is_v3 or self.is_auraflow or self.is_flux or self.is_lumina2 or self.is_transformer:
                    unet_prefix = f"lora_transformer"
                    if self.peft_format:
                        unet_prefix = "transformer"

                prefix = (
                    unet_prefix
                    if is_unet
                    else (
                        self.LORA_PREFIX_TEXT_ENCODER
                        if text_encoder_idx is None
                        else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                    )
                )
                loras = []
                skipped = []
                lora_shape_dict = {} 
                
                # Iterate over ALL named modules in the root_module
                for full_name, module_obj in root_module.named_modules():
                    # Check if the module's name matches any of the target_replace_modules patterns
                    # This is more robust than checking module.__class__.__name__ directly for deeply nested modules
                    if not any(re.search(pattern, full_name) for pattern in target_replace_modules):
                        continue # Skip if not a target module based on name pattern

                    is_linear = module_obj.__class__.__name__ in LINEAR_MODULES
                    is_conv2d = module_obj.__class__.__name__ in CONV_MODULES
                    is_conv2d_1x1 = is_conv2d and module_obj.kernel_size == (1, 1)

                    if not (is_linear or is_conv2d): # Only consider actual Linear or Conv2d layers
                        continue

                    # Construct lora_name (toolkit format)
                    clean_name_for_filters = full_name 
                    if self.peft_format:
                        lora_name_toolkit = full_name.replace(".", "$$")
                    else:
                        lora_name_toolkit = full_name.replace(".", "_")

                    skip = False
                    if any([re.search(pattern, clean_name_for_filters) for pattern in self.ignore_if_contains]):
                        skip = True

                    if hasattr(module_obj, 'weight') and count_parameters(module_obj) < parameter_threshold:
                        skip = True
                    
                    if self.transformer_only and is_unet:
                        transformer_keywords = ["transformer_blocks", "single_transformer_blocks", "double_blocks", "layers", "noise_refiner", "context_refiner", "attn", "ff"]
                        if not any(kw in clean_name_for_filters for kw in transformer_keywords):
                            skip = True
                                        
                    if self.only_if_contains and not any(re.search(pattern, clean_name_for_filters) for pattern in self.only_if_contains):
                        skip = True
                    
                    if skip:
                        skipped.append(lora_name_toolkit)
                        continue

                    dim = None
                    alpha = None

                    if modules_dim is not None:
                        if lora_name_toolkit in modules_dim:
                            dim = modules_dim[lora_name_toolkit]
                            alpha = modules_alpha[lora_name_toolkit]
                    else:
                        if is_linear or is_conv2d_1x1:
                            dim = self.lora_dim
                            alpha = self.alpha
                        elif is_conv2d and self.conv_lora_dim is not None and self.conv_lora_dim > 0:
                            dim = self.conv_lora_dim
                            alpha = self.conv_alpha

                    if dim is None or dim == 0:
                        skipped.append(lora_name_toolkit)
                        continue
                    
                    module_kwargs = {}
                    
                    if self.network_type.lower() == "lokr":
                        module_kwargs["factor"] = self.network_config.lokr_factor

                    # Determine parent module correctly for LoRAModule instantiation
                    parent_path_parts = full_name.split('.')
                    current_parent_module = root_module
                    if len(parent_path_parts) > 1:
                        for part in parent_path_parts[:-1]:
                            current_parent_module = getattr(current_parent_module, part)

                    lora = module_class(
                        lora_name_toolkit,
                        module_obj, # Pass the actual Linear/Conv module here as org_module
                        self.multiplier,
                        dim,
                        alpha,
                        dropout=dropout,
                        rank_dropout=rank_dropout,
                        module_dropout=module_dropout,
                        network=self,
                        parent=current_parent_module, # Pass the direct parent module
                        use_bias=use_bias,
                        **module_kwargs
                    )
                    loras.append(lora)
                    if self.network_type.lower() == "lokr":
                        try:
                            lora_shape_dict[lora_name_toolkit] = [list(lora.lokr_w1.weight.shape), list(lora.lokr_w2.weight.shape)]
                        except:
                            pass
                    else:
                        lora_shape_dict[lora_name_toolkit] = [list(lora.lora_down.weight.shape), list(lora.lora_up.weight.shape)]
                return loras, skipped

            text_encoders_list = text_encoder if type(text_encoder) == list else [text_encoder]

            self.text_encoder_loras = []
            skipped_te = []
            if train_text_encoder:
                text_encoder_generic_target_modules = target_lin_modules + target_conv_modules # Use direct arguments
                if self.is_pixart and "T5EncoderModel" not in text_encoder_generic_target_modules:
                    text_encoder_generic_target_modules.append("T5EncoderModel")
                if not text_encoder_generic_target_modules:
                    text_encoder_generic_target_modules = LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE


                for i, te_model_instance in enumerate(text_encoders_list):
                    if not use_text_encoder_1 and i == 0:
                        continue
                    if not use_text_encoder_2 and i == 1:
                        continue
                    if len(text_encoders_list) > 1:
                        index = i + 1
                        print(f"create LoRA for Text Encoder {index}:")
                    else:
                        index = None
                        print(f"create LoRA for Text Encoder:")

                    text_encoder_loras, skipped = create_modules(False, index, te_model_instance, text_encoder_generic_target_modules)
                    self.text_encoder_loras.extend(text_encoder_loras)
                    skipped_te += skipped
            print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

            unet_specific_target_modules = target_lin_modules + target_conv_modules # Use direct arguments
            if is_v3 and "SD3Transformer2DModel" not in unet_specific_target_modules:
                unet_specific_target_modules.append("SD3Transformer2DModel")
            elif is_pixart and "PixArtTransformer2DModel" not in unet_specific_target_modules:
                unet_specific_target_modules.append("PixArtTransformer2DModel")
            elif is_auraflow and "AuraFlowTransformer2DModel" not in unet_specific_target_modules:
                unet_specific_target_modules.append("AuraFlowTransformer2DModel")
            elif is_flux and "FluxTransformer2DModel" not in unet_specific_target_modules:
                unet_specific_target_modules.append("FluxTransformer2DModel")
            elif is_lumina2 and "Lumina2Transformer2DModel" not in unet_specific_target_modules:
                unet_specific_target_modules.append("Lumina2Transformer2DModel")
            elif not unet_specific_target_modules: # Fallback to default LoRANetwork targets if nothing is set
                unet_specific_target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE + LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3


            if train_unet:
                self.unet_loras, skipped_un = create_modules(True, None, unet, unet_specific_target_modules)
            else:
                self.unet_loras = []
                skipped_un = []
            print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

            skipped = skipped_te + skipped_un
            if varbose and len(skipped) > 0:
                print(
                    f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
                )
                for name in skipped:
                    print(f"\t{name}")

            self.up_lr_weight: List[float] = None
            self.down_lr_weight: List[float] = None
            self.mid_lr_weight: float = None
            self.block_lr = False

            names = set()
            for lora in self.text_encoder_loras + self.unet_loras:
                assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
                names.add(lora.lora_name)

            if self.full_train_in_out:
                print("full train in out")
                base_model_ref = self.base_model_ref()
                base_model_arch = base_model_ref.arch if base_model_ref else None

                if is_pixart or is_auraflow or is_flux or (base_model_arch == "wan21"):
                    transformer = unet
                    self.transformer_pos_embed = copy.deepcopy(transformer.pos_embed)
                    self.transformer_proj_out = copy.deepcopy(transformer.proj_out)
                    transformer.pos_embed = self.transformer_pos_embed
                    transformer.proj_out = self.transformer_proj_out
                else:
                    unet_conv_in: torch.nn.Conv2d = unet.conv_in
                    unet_conv_out: torch.nn.Conv2d = unet.conv_out
                    self.unet_conv_in = copy.deepcopy(unet_conv_in)
                    self.unet_conv_out = copy.deepcopy(unet_conv_out)
                    unet.conv_in = self.unet_conv_in
                    unet.conv_out = self.unet_conv_out
        # --- END OF ORIGINAL LORANETWORK INITIALIZATION BRANCH ---

    # --- NEW METHOD: Overrides LoRANetwork.apply_to to make it conditional ---
    def apply_to(self, text_encoder_modules, unet_module, train_text_encoder, train_unet):
        """
        Conditionally applies PEFT AdaLoRA or the original LoRA activation based on network_type.
        This method is called from BaseSDTrainProcess.run()
        """
        base_model_instance = self.base_model_ref()
        if base_model_instance is None:
            raise ValueError("Base model reference is missing during apply_to. This should not happen.")

        if self.network_type.lower() == "adalora":
            # Apply to UNet
            if train_unet:
                print("Applying AdaLoRA to UNet (transformer)")
                # CRITICAL: Apply the forward wrapper to the UNet BEFORE PEFT wraps it
                original_unet_forward_method = unet_module.forward
                unet_module.forward = _qwen_unet_forward_wrapper(original_unet_forward_method.__get__(unet_module))
                
                # Use the stored adalora_config from __init__
                self.peft_adapted_unet = get_peft_model(unet_module, self.adalora_config, adapter_name="adalora_unet")
                self.peft_adapted_unet.train()
                # IMPORTANT: Restore original unet.forward after PEFT has wrapped it
                unet_module.forward = original_unet_forward_method

                # Update the base_model's unet reference to the PEFT-adapted one
                base_model_instance.unet = self.peft_adapted_unet
            else:
                self.peft_adapted_unet = unet_module # Keep original if not training

            # Apply to Text Encoder(s)
            if train_text_encoder and text_encoder_modules:
                print("Applying AdaLoRA to Text Encoder(s)")
                text_encoders_to_adapt = text_encoder_modules if isinstance(text_encoder_modules, list) else [text_encoder_modules]
                
                # Clear self.peft_adapted_text_encoders (which is a ModuleList) before populating
                self.peft_adapted_text_encoders = torch.nn.ModuleList() # FIX: Re-initialize as ModuleList
                
                for i, te_module in enumerate(text_encoders_to_adapt):
                    # For Text Encoder, we also need to ensure `input_ids` filtering if Qwen pipeline passes it
                    original_te_forward_method = te_module.forward
                    te_module.forward = _qwen_unet_forward_wrapper(original_te_forward_method.__get__(te_module)) # Using unet wrapper, assuming it only pops input_ids

                    # Use the stored adalora_config from __init__
                    peft_te = get_peft_model(te_module, self.adalora_config, adapter_name=f"adalora_te_{i}") 
                    peft_te.train()
                    self.peft_adapted_text_encoders.append(peft_te) # Append to the ModuleList
                    te_module.forward = original_te_forward_method # Restore

                # Update the base_model's text_encoder reference(s) to the PEFT-adapted one(s)
                if isinstance(base_model_instance.text_encoder, list):
                    base_model_instance.text_encoder = self.peft_adapted_text_encoders # This is now a ModuleList
                else:
                    base_model_instance.text_encoder = self.peft_adapted_text_encoders[0] # Assuming single TE case, get first element from ModuleList
            else:
                # If not training TE, ensure self.peft_adapted_text_encoders (ModuleList) contains original modules
                self.peft_adapted_text_encoders = torch.nn.ModuleList(text_encoder_modules) # FIX: Wrap in ModuleList

        else:
            # For all other network types (lora, locon, lokr, dora, lorm),
            # call the parent's apply_to method. This will set up the LoRAModule forwards.
            super().apply_to(text_encoder_modules, unet_module, train_text_encoder, train_unet)


    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        if self.network_type.lower() == "adalora": # Use self.network_type, which is already set
            all_params = []
            
            base_model_instance = self.base_model_ref()
            if base_model_instance is None:
                raise ValueError("Base model reference is missing for AdaLoRA parameter preparation.")

            if self.peft_adapted_text_encoders:
                for i, te_model in enumerate(self.peft_adapted_text_encoders):
                    if isinstance(te_model, peft.PeftModel) and \
                       f"{self.peft_adapter_name}_te_{i}" in te_model.peft_config:
                        all_params.append({"lr": text_encoder_lr, "params": list(te_model.parameters())})
                    elif hasattr(te_model, "parameters") and len(list(te_model.parameters(recurse=False))) > 0:
                        all_params.append({"lr": text_encoder_lr, "params": list(te_model.parameters())})

            if self.peft_adapted_unet is not None:
                if isinstance(self.peft_adapted_unet, peft.PeftModel) and \
                   f"{self.peft_adapter_name}_unet" in self.peft_adapted_unet.peft_config:
                    all_params.append({"lr": unet_lr, "params": list(self.peft_adapted_unet.parameters())})
                elif hasattr(self.peft_adapted_unet, "parameters") and len(list(self.peft_adapted_unet.parameters(recurse=False))) > 0:
                    all_params.append({"lr": unet_lr, "params": list(self.peft_adapted_unet.parameters())})
            
            if self.full_train_in_out:
                if hasattr(self, 'transformer_pos_embed'):
                    all_params.append({"lr": unet_lr, "params": list(self.transformer_pos_embed.parameters())})
                if hasattr(self, 'transformer_proj_out'):
                    all_params.append({"lr": unet_lr, "params": list(self.transformer_proj_out.parameters())})
                if hasattr(self, 'unet_conv_in'):
                    all_params.append({"lr": unet_lr, "params": list(self.unet_conv_in.parameters())})
                if hasattr(self, 'unet_conv_out'):
                    all_params.append({"lr": unet_lr, "params": list(self.unet_conv_out.parameters())})

            return all_params
        else:
            all_params = super().prepare_optimizer_params(text_encoder_lr, unet_lr, default_lr)
            if self.full_train_in_out:
                base_model_ref = self.base_model_ref()
                base_model_arch = base_model_ref.arch if base_model_ref else None
                if self.is_pixart or self.is_auraflow or self.is_flux or (base_model_arch == "wan21"):
                    if hasattr(self, 'transformer_pos_embed'):
                        all_params.append({"lr": unet_lr, "params": list(self.transformer_pos_embed.parameters())})
                    if hasattr(self, 'transformer_proj_out'):
                        all_params.append({"lr": unet_lr, "params": list(self.transformer_proj_out.parameters())})
                else:
                    if hasattr(self, 'unet_conv_in'):
                        all_params.append({"lr": unet_lr, "params": list(self.unet_conv_in.parameters())})
                    if hasattr(self, 'unet_conv_out'):
                        all_params.append({"lr": unet_lr, "params": list(self.unet_conv_out.parameters())})
            return all_params

    def save_weights(
        self,
        file_path,
        dtype=torch.float32,
        metadata: Optional[Dict[str, str]] = None,
        extra_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if self.network_type.lower() == "adalora":
            adapters_state_dict = OrderedDict() 
            
            base_model_instance = self.base_model_ref()
            if base_model_instance is None:
                raise ValueError("Base model reference is missing for AdaLoRA saving.")

            if self.peft_adapted_text_encoders: 
                for i, te_model in enumerate(self.peft_adapted_text_encoders):
                    if isinstance(te_model, peft.PeftModel) and \
                       f"{self.peft_adapter_name}_te_{i}" in te_model.peft_config:
                        te_state_dict = get_peft_model_state_dict(te_model, adapter_name=f"{self.peft_adapter_name}_te_{i}")
                        for k, v in te_state_dict.items():
                            adapters_state_dict[f"text_encoder.{i}.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            if self.peft_adapted_unet: 
                if isinstance(self.peft_adapted_unet, peft.PeftModel) and \
                   f"{self.peft_adapter_name}_unet" in self.peft_adapted_unet.peft_config:
                    unet_state_dict = get_peft_model_state_dict(self.peft_adapted_unet, adapter_name=f"{self.peft_adapter_name}_unet")
                    for k,v in unet_state_dict.items():
                        adapters_state_dict[f"transformer.{k}"] = v.detach().clone().to("cpu").to(dtype)

            if self.full_train_in_out:
                if hasattr(self, 'transformer_pos_embed'):
                    adapters_state_dict.update({"transformer_pos_embed." + k: v for k, v in self.transformer_pos_embed.state_dict().items()})
                if hasattr(self, 'transformer_proj_out'):
                    adapters_state_dict.update({"transformer_proj_out." + k: v for k, v in self.transformer_proj_out.state_dict().items()})
                if hasattr(self, 'unet_conv_in'):
                    adapters_state_dict.update({"unet_conv_in." + k: v for k, v in self.unet_conv_in.state_dict().items()})
                if hasattr(self, 'unet_conv_out'):
                    adapters_state_dict.update({"unet_conv_out." + k: v for k, v in self.unet_conv_out.state_dict().items()})

            if extra_state_dict:
                adapters_state_dict.update(extra_state_dict)

            for key, value in adapters_state_dict.items():
                adapters_state_dict[key] = value.clone().to('cpu', dtype=dtype)

            from safetensors.torch import save_file 
            save_file(adapters_state_dict, file_path, metadata)
            return
        else:
            super().save_weights(
                file_path,
                dtype=dtype,
                metadata=metadata,
                extra_state_dict=extra_state_dict,
            )
            if self.full_train_in_out:
                state_dict_to_save = {}
                base_model_ref = self.base_model_ref()
                base_model_arch = base_model_ref.arch if base_model_ref else None

                if self.is_pixart or self.is_auraflow or self.is_flux or (base_model_arch == "wan21"):
                    if hasattr(self, 'transformer_pos_embed'):
                        state_dict_to_save.update({"transformer_pos_embed." + k: v for k, v in self.transformer_pos_embed.state_dict().items()})
                    if hasattr(self, 'transformer_proj_out'):
                        state_dict_to_save.update({"transformer_proj_out." + k: v for k, v in self.transformer_proj_out.state_dict().items()})
                else:
                    if hasattr(self, 'unet_conv_in'):
                        state_dict_to_save.update({"unet_conv_in." + k: v for k, v in self.unet_conv_in.state_dict().items()})
                    if hasattr(self, 'unet_conv_out'):
                        state_dict_to_save.update({"unet_conv_out." + k: v for k, v in self.unet_conv_out.state_dict().items()})
                
                if state_dict_to_save:
                    current_state_dict = load_file(file_path, device="cpu")
                    current_state_dict.update(state_dict_to_save)
                    save_file(current_state_dict, file_path, metadata)

    def load_weights(self: Network, file, force_weight_mapping=False):
        if self.network_type.lower() == "adalora":
            adapter_state_dict = load_file(file, device="cpu")
            base_model_instance = self.base_model_ref()
            if base_model_instance is None:
                raise ValueError("Base model reference is missing for AdaLoRA loading.")

            unet_peft_state_dict = OrderedDict()
            text_encoder_peft_state_dict_map = {} # Map index to its state dict

            for key, value in adapter_state_dict.items():
                if key.startswith("transformer."):
                    unet_peft_state_dict[key.replace("transformer.", "")] = value
                elif key.startswith("text_encoder."):
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[1].isdigit(): # key looks like text_encoder.0.xxx
                        te_idx = int(parts[1])
                        peft_te_key = ".".join(parts[2:]) # Reconstruct PEFT key
                        if te_idx not in text_encoder_peft_state_dict_map:
                            text_encoder_peft_state_dict_map[te_idx] = OrderedDict()
                        text_encoder_peft_state_dict_map[te_idx][peft_te_key] = value
                    else: # Handle single TE case or different key format (e.g., text_encoder.lora_A.weight)
                        if 0 not in text_encoder_peft_state_dict_map:
                            text_encoder_peft_state_dict_map[0] = OrderedDict()
                        text_encoder_peft_state_dict_map[0][key.replace("text_encoder.", "")] = value
                # --- Handle full_train_in_out modules ---
                elif key.startswith("transformer_pos_embed."):
                    if not hasattr(self, 'transformer_pos_embed') or self.transformer_pos_embed is None: 
                        self.transformer_pos_embed = copy.deepcopy(base_model_instance.transformer_pos_embed)
                    self.transformer_pos_embed.load_state_dict({key.replace("transformer_pos_embed.", ""): value}, strict=False)
                elif key.startswith("transformer_proj_out."):
                    if not hasattr(self, 'transformer_proj_out') or self.transformer_proj_out is None:
                        self.transformer_proj_out = copy.deepcopy(base_model_instance.transformer_proj_out)
                    self.transformer_proj_out.load_state_dict({key.replace("transformer_proj_out.", ""): value}, strict=False)
                elif key.startswith("unet_conv_in."):
                    if not hasattr(self, 'unet_conv_in') or self.unet_conv_in is None:
                        self.unet_conv_in = copy.deepcopy(base_model_instance.unet_conv_in)
                    self.unet_conv_in.load_state_dict({key.replace("unet_conv_in.", ""): value}, strict=False)
                elif key.startswith("unet_conv_out."):
                    if not hasattr(self, 'unet_conv_out') or self.unet_conv_out is None:
                        self.unet_conv_out = copy.deepcopy(base_model_instance.unet_conv_out)
                    self.unet_conv_out.load_state_dict({key.replace("unet_conv_out.", ""): value}, strict=False)
            
            if self.peft_adapted_unet:
                set_peft_model_state_dict(self.peft_adapted_unet, unet_peft_state_dict, adapter_name=f"{self.peft_adapter_name}_unet")
            
            if self.peft_adapted_text_encoders:
                for idx, te_model in enumerate(self.peft_adapted_text_encoders):
                    if idx in text_encoder_peft_state_dict_map:
                        set_peft_model_state_dict(te_model, text_encoder_peft_state_dict_map[idx], adapter_name=f"adalora_te_{idx}")
            
            print(f"Loaded AdaLoRA weights from {file}")
            return {} 
        else:
            keymap = self.get_keymap(force_weight_mapping)
            keymap = {} if keymap is None else keymap

            if isinstance(file, str):
                if os.path.splitext(file)[1] == ".safetensors":
                    from safetensors.torch import load_file

                    weights_sd = load_file(file)
                else:
                    weights_sd = torch.load(file, map_location="cpu")
            else:
                weights_sd = file
            
            if self.base_model_ref is not None:
                weights_sd = self.base_model_ref().convert_lora_weights_before_load(weights_sd)

            load_sd = OrderedDict()
            for key, value in weights_sd.items():
                load_key = keymap[key] if key in keymap else key
                if self.is_pixart:
                    load_key = load_key.replace('__', '_')

                if self.peft_format:
                    if load_key.endswith('.alpha'):
                        continue
                    load_key = load_key.replace('lora_A', 'lora_down')
                    load_key = load_key.replace('lora_B', 'lora_up')
                    load_key = load_key.replace('.', '$$')
                    load_key = load_key.replace('$$lora_down$$', '.lora_down.')
                    load_key = load_key.replace('$$lora_up$$', '.lora_up.')
                
                if self.network_type.lower() == "lokr":
                    load_key = load_key.replace('lycoris_', 'lora_transformer_')

                load_sd[load_key] = value

            current_state_dict = self.state_dict()
            extra_dict = OrderedDict()
            to_delete = []
            for key in list(load_sd.keys()):
                if key not in current_state_dict:
                    extra_dict[key] = load_sd[key]
                    to_delete.append(key)
            for key in to_delete:
                del load_sd[key]

            print(f"Missing keys: {to_delete}")
            if len(to_delete) > 0 and self.is_v1 and not force_weight_mapping and not (
                    len(to_delete) == 1 and 'emb_params' in to_delete):
                print(" Attempting to load with forced keymap")
                return self.load_weights(file, force_weight_mapping=True)

            info = self.load_state_dict(load_sd, False)
            if len(extra_dict.keys()) == 0:
                extra_dict = None
            return extra_dict

    @torch.no_grad()
    def _update_torch_multiplier(self: Network):
        if self.network_type.lower() == "adalora":
            device = torch.device("cpu")
            dtype = torch.float32
            # Determine device/dtype from the PEFT-adapted models if they exist
            if hasattr(self, 'peft_adapted_unet') and self.peft_adapted_unet is not None and isinstance(self.peft_adapted_unet, peft.PeftModel):
                device = self.peft_adapted_unet.device
                dtype = self.peft_adapted_unet.dtype
            elif hasattr(self, 'peft_adapted_text_encoders') and self.peft_adapted_text_encoders is not None:
                if isinstance(self.peft_adapted_text_encoders, torch.nn.ModuleList) and len(self.peft_adapted_text_encoders) > 0 and isinstance(self.peft_adapted_text_encoders[0], peft.PeftModel):
                    device = self.peft_adapted_text_encoders[0].device
                    dtype = self.peft_adapted_text_encoders[0].dtype
                elif isinstance(self.peft_adapted_text_encoders, peft.PeftModel): # Handle single PeftModel
                    device = self.peft_adapted_text_encoders.device
                    dtype = self.peft_adapted_text_encoders.dtype

            self.torch_multiplier = torch.tensor(1.0).to(device, dtype=dtype)
            return # Exit for AdaLoRA

        multiplier = self._multiplier
        try:
            first_module = self.get_all_modules()[0]
        except IndexError:
            raise ValueError("There are not any lora modules in this network. Check your config and try again")
        
        if hasattr(first_module, 'lora_down'):
            device = first_module.lora_down.weight.device
            dtype = first_module.lora_down.weight.dtype
        elif hasattr(first_module, 'lokr_w1'):
            device = first_module.lokr_w1.device
            dtype = first_module.lokr_w1.dtype
        elif hasattr(first_module, 'lokr_w1_a'):
            device = first_module.lokr_w1_a.device
            dtype = first_module.lokr_w1_a.dtype
        else:
            raise ValueError("Unknown module type")
        with torch.no_grad():
            tensor_multiplier = None
            if isinstance(multiplier, int) or isinstance(multiplier, float):
                tensor_multiplier = torch.tensor((multiplier,)).to(device, dtype=dtype)
            elif isinstance(multiplier, list):
                tensor_multiplier = torch.tensor(multiplier).to(device, dtype=dtype)
            elif isinstance(multiplier, torch.Tensor):
                tensor_multiplier = multiplier.clone().detach().to(device, dtype=dtype)

            self.torch_multiplier = tensor_multiplier.clone().detach()

    @property
    def multiplier(self) -> Union[float, List[float], List[List[float]]]:
        if self.network_type.lower() == "adalora":
            return 1.0 # AdaLoRA does not use this property in the same way
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float], List[List[float]]]):
        if self.network_type.lower() == "adalora":
            self._multiplier = value # Still store, but _update_torch_multiplier will bypass
            return

        if self._multiplier == value:
            return
        self._multiplier = value
        self._update_torch_multiplier()

    def __enter__(self: Network):
        self.is_active = True

    def __exit__(self: Network, exc_type, exc_value, tb):
        self.is_active = False

    def force_to(self: Network, device, dtype):
        if self.network_type.lower() == "adalora":
            # For AdaLoRA, accelerator handles moving PEFT models.
            return

        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)

    def get_all_modules(self: Network) -> List[Module]:
        if self.network_type.lower() == "adalora":
            return [] # AdaLoRA bypasses this logic, PEFT models are managed differently
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        return loras

    def _update_checkpointing(self: Network):
        if self.network_type.lower() == "adalora":
            # PEFT models handle their own gradient checkpointing internally.
            return
        for module in self.get_all_modules():
            if self.is_checkpointing:
                if hasattr(module, 'enable_gradient_checkpointing'):
                    module.enable_gradient_checkpointing()
                elif hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
            else:
                if hasattr(module, 'disable_gradient_checkpointing'):
                    module.disable_gradient_checkpointing()
                elif hasattr(module, 'gradient_checkpointing_disable'):
                    module.gradient_checkpointing_disable()

    def enable_gradient_checkpointing(self: 'LoRASpecialNetwork'):
        self.is_checkpointing = True
        if self.network_type.lower() == "adalora":
            if self.peft_adapted_unet and hasattr(self.peft_adapted_unet, "gradient_checkpointing_enable"):
                self.peft_adapted_unet.gradient_checkpointing_enable()
            if self.peft_adapted_text_encoders:
                for te_model in self.peft_adapted_text_encoders:
                    if hasattr(te_model, "gradient_checkpointing_enable"):
                        te_model.gradient_checkpointing_enable()
        else:
            super().enable_gradient_checkpointing()
            self._update_checkpointing()
    
    def disable_gradient_checkpointing(self: 'LoRASpecialNetwork'):
        self.is_checkpointing = False
        if self.network_type.lower() == "adalora":
            if self.peft_adapted_unet and hasattr(self.peft_adapted_unet, "gradient_checkpointing_disable"):
                self.peft_adapted_unet.gradient_checkpointing_disable()
            if self.peft_adapted_text_encoders:
                for te_model in self.peft_adapted_text_encoders:
                    if hasattr(te_model, "gradient_checkpointing_disable"):
                        te_model.gradient_checkpointing_disable()
        else:
            super().disable_gradient_checkpointing()
            self._update_checkpointing()

    def merge_in(self: 'LoRASpecialNetwork', merge_weight=1.0):
        if self.network_type.lower() == "adalora":
            print("Warning: merge_in is not directly supported for AdaLoRA networks via this method. PEFT handles merging internally.")
        else:
            if self.network_type.lower() == 'dora':
                return
            self.is_merged_in = True
            for module in self.get_all_modules():
                if hasattr(module, 'merge_in'):
                    module.merge_in(merge_weight)
    
    def merge_out(self: 'LoRASpecialNetwork', merge_weight=1.0):
        if self.network_type.lower() == "adalora":
            print("Warning: merge_out is not directly supported for AdaLoRA networks via this method. PEFT handles merging internally.")
        else:
            if not self.is_merged_in:
                return
            self.is_merged_in = False
            for module in self.get_all_modules():
                if hasattr(module, 'merge_out'):
                    module.merge_out(merge_weight)

    def extract_weight(
            self: 'Network',
            extract_mode: ExtractMode = "existing",
            extract_mode_param: Union[int, float] = None,
    ):
        if self.network_type.lower() == "adalora":
            # This is specific to custom LoRA modules. AdaLoRA does not use this.
            return
        if extract_mode_param is None:
            raise ValueError("extract_mode_param must be set")
        for module in tqdm(self.get_all_modules(), desc="Extracting weights"):
            if hasattr(module, 'extract_weight'):
                module.extract_weight(
                    extract_mode=extract_mode,
                    extract_mode_param=extract_mode_param
                )

    def setup_lorm(self: 'Network', state_dict: Optional[Dict[str, Any]] = None):
        if self.network_type.lower() == "adalora":
            # This is specific to LoRM. AdaLoRA does not use this.
            return
        for module in tqdm(self.get_all_modules(), desc="Extracting LoRM"):
            if hasattr(module, 'setup_lorm'):
                module.setup_lorm(state_dict=state_dict)

    def calculate_lorem_parameter_reduction(self):
        if self.network_type.lower() == "adalora":
            # AdaLoRA dynamically manages ranks, not LoRM-style reduction calculation.
            return 0
        params_reduced = 0
        for module in self.get_all_modules():
            if hasattr(module, 'org_module') and hasattr(module, 'lora_down') and hasattr(module, 'lora_up'):
                num_orig_module_params = count_parameters(module.org_module[0])
                num_lorem_params = count_parameters(module.lora_down) + count_parameters(module.lora_up)
                params_reduced += (num_orig_module_params - num_lorem_params)

        return params_reduced