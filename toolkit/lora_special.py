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
from diffusers import UNet2DConditionModel, PixArtTransformer2DModel, AuraFlowTransformer2DModel, WanTransformer3DModel
from transformers import CLIPTextModel
from dataclasses import field # Added this import
from toolkit.models.lokr import LokrModule

from .config_modules import NetworkConfig
from .lorm import count_parameters
from .network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin, ExtractMode
from toolkit.kohya_lora import LoRANetwork
from toolkit.models.DoRA import DoRAModule

# --- NEW IMPORTS FOR PEFT (start) ---
from peft import LoraConfig, AdaLoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
# --- NEW IMPORTS FOR PEFT (end) ---

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
    'QLinear',
    # 'GroupNorm',
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
        # del self.org_module


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
            is_lorm: bool = False,
            ignore_if_contains = None,
            only_if_contains = None,
            parameter_threshold: float = 0.0,
            attn_only: bool = False,
            target_lin_modules: Optional[List[str]] = None, # No longer default_factory
            target_conv_modules: Optional[List[str]] = None, # No longer default_factory
            network_type: str = "lora",
            full_train_in_out: bool = False,
            transformer_only: bool = False,
            peft_format: bool = False,
            is_assistant_adapter: bool = False,
            is_transformer: bool = False,
            base_model: 'StableDiffusion' = None,
            total_training_steps: Optional[int] = None,
            **kwargs
    ) -> None:
        torch.nn.Module.__init__(self)
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_sdxl=is_sdxl,
            is_v2=is_v2,
            is_lorm=is_lorm,
            network_config=kwargs.get("network_config", None),
            **kwargs
        )
        if ignore_if_contains is None:
            ignore_if_contains = []
        self.ignore_if_contains = ignore_if_contains
        self.transformer_only = transformer_only
        self.base_model_ref = None
        if base_model is not None:
            self.base_model_ref = weakref.ref(base_model)

        self.only_if_contains: Union[List, None] = only_if_contains

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
        self.multiplier = multiplier
        self.is_sdxl = is_sdxl
        self.is_v2 = is_v2
        self.is_v3 = is_v3
        self.is_pixart = is_pixart
        self.is_auraflow = is_auraflow
        self.is_flux = is_flux
        self.is_lumina2 = is_lumina2
        self.network_type = network_type
        self.is_assistant_adapter = is_assistant_adapter
        
        self.is_adalora = (network_type.lower() == "adalora")
        self.adalora_config = None
        self.adalora_unet_model = None
        self.adalora_text_encoder_model = None

        if self.network_type.lower() == "dora":
            self.module_class = DoRAModule
            module_class = DoRAModule
        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule
        self.network_config: NetworkConfig = kwargs.get("network_config", None)
        
        # --- MODIFIED: Explicitly assign target_lin_modules and target_conv_modules to self (start) ---
        self.target_lin_modules: List[str] = kwargs.get('target_lin_modules', [])
        self.target_conv_modules: List[str] = kwargs.get('target_conv_modules', [])
        
        if self.target_lin_modules is None: # Ensure it's always a list
            self.target_lin_modules = []
        if self.target_conv_modules is None: # Ensure it's always a list
            self.target_conv_modules = []
        # --- MODIFIED: Explicitly assign target_lin_modules and target_conv_modules to self (end) ---

        self.peft_format = peft_format
        self.is_transformer = is_transformer
        
        if self.is_flux or self.is_v3 or self.is_lumina2 or is_transformer:
            if self.network_type.lower() != "lokr":
                self.peft_format = True

        if self.peft_format:
            self.alpha = self.lora_dim
            alpha = self.alpha
            self.conv_alpha = self.conv_lora_dim
            conv_alpha = self.conv_alpha

        self.full_train_in_out = full_train_in_out

        if self.is_adalora:
            print(f"Creating AdaLoRA network. Initializing AdaLoraConfig.")
            adalora_rank = lora_dim
            adalora_alpha = alpha
            adalora_dropout = dropout if dropout is not None else 0.0
            
            target_r = kwargs.get('adalora_target_r', adalora_rank // 4 if adalora_rank else 0)
            init_r = kwargs.get('adalora_init_r', adalora_rank)
            tinit = kwargs.get('adalora_tinit', total_training_steps // 4 if total_training_steps else 0)
            tfinal = kwargs.get('adalora_tfinal', 3 * total_training_steps // 4 if total_training_steps else 0)
            deltaT = kwargs.get('adalora_deltaT', 10)
            beta1 = kwargs.get('adalora_beta1', 0.85)
            beta2 = kwargs.get('adalora_beta2', 0.85)
            orth_reg_weight = kwargs.get('adalora_orth_reg_weight', 0.5)

            adalora_target_modules = kwargs.get('target_lin_modules', ["to_k", "to_q", "to_v", "to_out.0"])
            
            if is_v3:
                adalora_target_modules = ["SD3Transformer2DModel"]
            elif is_pixart:
                adalora_target_modules = ["PixArtTransformer2DModel"]
            elif is_auraflow:
                adalora_target_modules = ["AuraFlowTransformer2DModel"]
            elif is_flux:
                adalora_target_modules = ["FluxTransformer2DModel"]
            elif is_lumina2:
                adalora_target_modules = ["Lumina2Transformer2DModel"]
            elif self.is_transformer:
                if not adalora_target_modules:
                    adalora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

            if self.only_if_contains and base_model is not None:
                resolved_modules = []
                if hasattr(unet, 'named_modules'):
                    for module_name_pattern in self.only_if_contains:
                        for name, _ in unet.named_modules():
                            try:
                                if re.search(module_name_pattern, name):
                                    resolved_modules.append(name.split('.')[-1])
                            except re.error:
                                pass
                if resolved_modules:
                    print(f"Warning: `only_if_contains` patterns were found for AdaLoRA, attempting to resolve to PEFT target modules. Result: {list(set(resolved_modules))}")
                    adalora_target_modules.extend(resolved_modules)
                    adalora_target_modules = list(set(adalora_target_modules))

            self.adalora_config = AdaLoraConfig(
                peft_type="ADALORA",
                task_type="FEATURE_EXTRACTION",
                r=adalora_rank,
                lora_alpha=adalora_alpha,
                lora_dropout=adalora_dropout,
                target_modules=adalora_target_modules,
                init_lora_weights="gaussian",
                target_r=target_r,
                init_r=init_r,
                tinit=tinit,
                tfinal=tfinal,
                deltaT=deltaT,
                beta1=beta1,
                beta2=beta2,
                orth_reg_weight=orth_reg_weight,
                total_step=total_training_steps,
            )
            print(f"AdaLoRA configured with max rank {adalora_rank}, target rank {target_r}, "
                  f"tinit {tinit}, tfinal {tfinal}, total_step {total_training_steps}")

            self.text_encoder_loras = []
            self.unet_loras = []

        else: # Existing logic for LoRA, DoRA, LoKr (NON-ADALORA PATH)
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            if self.conv_lora_dim is not None and self.conv_lora_dim > 0:
                print(
                    f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

            # Define create_modules function (refactored for robust naming)
            def create_modules(
                    is_unet: bool,
                    text_encoder_idx: Optional[int],
                    root_module: torch.nn.Module,
                    target_replace_modules_compat: List[str], # Compatibility argument
            ) -> List[LoRAModule]:
                unet_prefix = self.LORA_PREFIX_UNET
                if self.peft_format:
                    unet_prefix = self.PEFT_PREFIX_UNET
                if is_pixart or is_v3 or is_auraflow or is_flux or is_lumina2 or self.is_transformer:
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

                # Iterate over ALL named modules in the root_module to get their full qualified names
                for full_name, module_obj in root_module.named_modules():
                    if full_name == "": # Skip the root module itself
                        continue
                    
                    is_linear = module_obj.__class__.__name__ in LINEAR_MODULES
                    is_conv2d = module_obj.__class__.__name__ in CONV_MODULES
                    is_conv2d_1x1 = is_conv2d and module_obj.kernel_size == (1, 1)

                    if not (is_linear or is_conv2d): # Only consider actual Linear or Conv2d layers
                        continue

                    clean_name_for_filters = full_name # Use full_name for all filtering logic
                    
                    if self.peft_format:
                        lora_name_toolkit = full_name.replace(".", "$$")
                    else:
                        lora_name_toolkit = full_name.replace(".", "_")

                    skip = False
                    # Apply ignore_if_contains filter
                    if any([re.search(pattern, clean_name_for_filters) for pattern in self.ignore_if_contains]):
                        skip = True

                    # Apply parameter_threshold
                    if hasattr(module_obj, 'weight') and count_parameters(module_obj) < parameter_threshold:
                        skip = True
                    
                    # Apply transformer_only filter for UNet
                    if self.transformer_only and is_unet:
                        transformer_keywords = ["transformer_blocks", "single_transformer_blocks", "double_blocks", "layers", "noise_refiner", "context_refiner", "attn", "ff"]
                        if not any(kw in clean_name_for_filters for kw in transformer_keywords):
                            skip = True
                                        
                    # Apply only_if_contains filter last
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

                    parent_path_parts = full_name.split('.')
                    parent_module_obj = root_module
                    if len(parent_path_parts) > 1:
                        for part in parent_path_parts[:-1]:
                            parent_module_obj = getattr(parent_module_obj, part)

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
                        parent=parent_module_obj,
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
            # End of create_modules function definition

            # --- DYNAMIC TARGET MODULE SETTING FOR NON-ADALORA NETWORKS (start) ---
            if self.is_transformer:
                if not self.target_lin_modules: 
                    self.target_lin_modules = ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"]
                if not self.target_conv_modules:
                    self.target_conv_modules = [] 
            else:
                if not self.target_lin_modules:
                    self.target_lin_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
                if not self.target_conv_modules:
                    self.target_conv_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            # --- DYNAMIC TARGET MODULE SETTING FOR NON-ADALORA NETWORKS (end) ---

            text_encoders_list = text_encoder if isinstance(text_encoder, list) else [text_encoder]

            # create LoRA for text encoder
            self.text_encoder_loras = []
            skipped_te = []
            if train_text_encoder:
                # Generic target list for Text Encoder, `only_if_contains` in YAML will filter further.
                text_encoder_generic_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "CLIPAttention", "CLIPMLP", "T5EncoderModel"]
                if self.is_pixart:
                    text_encoder_generic_target_modules = ["T5EncoderModel"]
                
                for i, text_encoder_module in enumerate(text_encoders_list):
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

                    text_encoder_loras, skipped = create_modules(False, index, text_encoder_module, text_encoder_generic_target_modules)
                    self.text_encoder_loras.extend(text_encoder_loras)
                    skipped_te += skipped
            print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

            # create LoRA for U-Net
            unet_generic_target_modules = ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", 
                                        "Transformer2DModel", "ResnetBlock2D", "AttentionBlock", 
                                        "SpatialTransformer", "BasicTransformerBlock", "linear", "conv"]
            if is_v3:
                unet_generic_target_modules = ["SD3Transformer2DModel"]
            elif is_pixart:
                unet_generic_target_modules = ["PixArtTransformer2DModel"]
            elif is_auraflow:
                unet_generic_target_modules = ["AuraFlowTransformer2DModel"]
            elif is_flux:
                unet_generic_target_modules = ["FluxTransformer2DModel"]
            elif is_lumina2:
                unet_generic_target_modules = ["Lumina2Transformer2DModel"]

            if train_unet:
                self.unet_loras, skipped_un = create_modules(True, None, unet, unet_generic_target_modules)
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

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        if self.full_train_in_out and not self.is_adalora:
            print("full train in out")
            if self.is_pixart:
                transformer: PixArtTransformer2DModel = unet
                self.transformer_pos_embed = copy.deepcopy(transformer.pos_embed)
                self.transformer_proj_out = copy.deepcopy(transformer.proj_out)

                transformer.pos_embed = self.transformer_pos_embed
                transformer.proj_out = self.transformer_proj_out

            elif self.is_auraflow:
                transformer: AuraFlowTransformer2DModel = unet
                self.transformer_pos_embed = copy.deepcopy(transformer.pos_embed)
                self.transformer_proj_out = copy.deepcopy(transformer.proj_out)

                transformer.pos_embed = self.transformer_pos_embed
                transformer.proj_out = self.transformer_proj_out
            
            elif base_model is not None and base_model.arch == "wan21":
                transformer: WanTransformer3DModel = unet
                self.transformer_pos_embed = copy.deepcopy(transformer.patch_embedding)
                self.transformer_proj_out = copy.deepcopy(transformer.proj_out)

                transformer.patch_embedding = self.transformer_pos_embed
                transformer.proj_out = self.transformer_proj_out

            else:
                unet: UNet2DConditionModel = unet
                unet_conv_in: torch.nn.Conv2d = unet.conv_in
                unet_conv_out: torch.nn.Conv2d = unet.conv_out

                self.unet_conv_in = copy.deepcopy(unet_conv_in)
                self.unet_conv_out = copy.deepcopy(unet_conv_out)
                unet.conv_in = self.unet_conv_in
                unet.conv_out = self.unet_conv_out

    def apply_to(self, text_encoder_modules, unet_module, train_text_encoder, train_unet):
        if self.is_adalora:
            if train_unet:
                print("Applying AdaLoRA to UNet (transformer)")
                self.adalora_unet_model = get_peft_model(unet_module, self.adalora_config, adapter_name="adalora_unet")
                self.adalora_unet_model.train()
            
            if train_text_encoder and text_encoder_modules:
                print("Applying AdaLoRA to Text Encoder(s)")
                if isinstance(text_encoder_modules, list):
                    self.adalora_text_encoder_model = []
                    for i, te_module in enumerate(text_encoder_modules):
                        peft_te = get_peft_model(te_module, self.adalora_config, adapter_name=f"adalora_te_{i}")
                        peft_te.train()
                        self.adalora_text_encoder_model.append(peft_te)
                else:
                    self.adalora_text_encoder_model = get_peft_model(text_encoder_modules, self.adalora_config, adapter_name="adalora_te")
                    self.adalora_text_encoder_model.train()
        else:
            super().apply_to(text_encoder_modules, unet_module, train_text_encoder, train_unet)

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        if self.is_adalora:
            params = []
            if self.adalora_unet_model:
                params.append({"lr": unet_lr, "params": list(self.adalora_unet_model.parameters())})
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for te_model in self.adalora_text_encoder_model:
                        params.append({"lr": text_encoder_lr, "params": list(te_model.parameters())})
                else:
                    params.append({"lr": text_encoder_lr, "params": list(self.adalora_text_encoder_model.parameters())})
            return params
        else:
            all_params = super().prepare_optimizer_params(text_encoder_lr, unet_lr, default_lr)

            if self.full_train_in_out:
                base_model = self.base_model_ref() if self.base_model_ref is not None else None
                if self.is_pixart or self.is_auraflow or self.is_flux or (base_model is not None and base_model.arch == "wan21"):
                    all_params.append({"lr": unet_lr, "params": list(self.transformer_pos_embed.parameters())})
                    all_params.append({"lr": unet_lr, "params": list(self.transformer_proj_out.parameters())})
                else:
                    all_params.append({"lr": unet_lr, "params": list(self.unet_conv_in.parameters())})
                    all_params.append({"lr": unet_lr, "params": list(self.unet_conv_out.parameters())})

            return all_params

    def update_and_allocate(self, global_step: int):
        if self.is_adalora:
            if self.adalora_unet_model and hasattr(self.adalora_unet_model.base_model, 'update_and_allocate'):
                self.adalora_unet_model.base_model.update_and_allocate(global_step)
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for te_model in self.adalora_text_encoder_model:
                        if hasattr(te_model.base_model, 'update_and_allocate'):
                            te_model.base_model.update_and_allocate(global_step)
                elif hasattr(self.adalora_text_encoder_model.base_model, 'update_and_allocate'):
                    self.adalora_text_encoder_model.base_model.update_and_allocate(global_step)

    def get_all_modules(self: 'LoRASpecialNetwork') -> List[torch.nn.Module]:
        if self.is_adalora:
            modules = []
            if self.adalora_unet_model:
                modules.append(self.adalora_unet_model)
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    modules.extend(self.adalora_text_encoder_model)
                else:
                    modules.append(self.adalora_text_encoder_model)
            return modules
        else:
            return super().get_all_modules()

    def get_state_dict(self: 'LoRASpecialNetwork', extra_state_dict=None, dtype=torch.float16):
        if self.is_adalora:
            save_dict = OrderedDict()
            if self.adalora_unet_model:
                unet_state_dict = get_peft_model_state_dict(self.adalora_unet_model, adapter_name="adalora_unet")
                for k, v in unet_state_dict.items():
                    save_dict[f"transformer.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for i, te_model in enumerate(self.adalora_text_encoder_model):
                        te_state_dict = get_peft_model_state_dict(te_model, adapter_name=f"adalora_te_{i}")
                        for k, v in te_state_dict.items():
                            save_dict[f"text_encoder.{i}.{k}"] = v.detach().clone().to("cpu").to(dtype)
                else:
                    te_state_dict = get_peft_model_state_dict(self.adalora_text_encoder_model, adapter_name="adalora_te")
                    for k, v in te_state_dict.items():
                        save_dict[f"text_encoder.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            if extra_state_dict is not None:
                for key in list(extra_state_dict.keys()):
                    v = extra_state_dict[key]
                    save_dict[key] = v.detach().clone().to("cpu").to(dtype)
            
            return save_dict
        else:
            keymap = self.get_keymap()

            save_keymap = {}
            if keymap is not None:
                for ldm_key, diffusers_key in keymap.items():
                    save_keymap[diffusers_key] = ldm_key

            state_dict = self.state_dict()
            save_dict = OrderedDict()

            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_key = save_keymap[key] if key in save_keymap else key
                save_dict[save_key] = v
                del state_dict[key]

            if extra_state_dict is not None:
                for key in list(extra_state_dict.keys()):
                    v = extra_state_dict[key]
                    v = v.detach().clone().to("cpu").to(dtype)
                    save_dict[key] = v

            if self.peft_format:
                new_save_dict = {}
                for key, value in save_dict.items():
                    if key.endswith('.alpha'):
                        continue
                    new_key = key
                    new_key = new_key.replace('lora_down', 'lora_A')
                    new_key = new_key.replace('lora_up', 'lora_B')
                    new_key = new_key.replace('$$', '.')
                    new_save_dict[new_key] = value

                save_dict = new_save_dict
            
            if self.network_type.lower() == "lokr":
                new_save_dict = {}
                for key, value in save_dict.items():
                    new_key = key
                    new_key = new_key.replace('lora_transformer_', 'lycoris_')
                    new_save_dict[new_key] = value

                save_dict = new_save_dict
            
            if self.base_model_ref is not None:
                save_dict = self.base_model_ref().convert_lora_weights_before_save(save_dict)
            return save_dict

    def load_weights(self: 'LoRASpecialNetwork', file, force_weight_mapping=False):
        if self.is_adalora:
            if isinstance(file, str):
                if os.path.splitext(file)[1] == ".safetensors":
                    from safetensors.torch import load_file
                    weights_sd = load_file(file, device="cpu")
                else:
                    weights_sd = torch.load(file, map_location="cpu")
            else:
                weights_sd = file

            if self.base_model_ref is not None:
                weights_sd = self.base_model_ref().convert_lora_weights_before_load(weights_sd)

            unet_state_dict = OrderedDict()
            text_encoder_state_dict = OrderedDict()
            extra_dict = OrderedDict()

            for key, value in weights_sd.items():
                if key.startswith("transformer."):
                    unet_state_dict[key.replace("transformer.", "")] = value
                elif key.startswith("text_encoder."):
                    te_key_suffix = key.replace('text_encoder.', '')
                    if '.' in te_key_suffix and te_key_suffix.split('.')[0].isdigit():
                        te_idx = int(te_key_suffix.split('.')[0])
                        text_encoder_state_dict[key.replace(f"text_encoder.{te_idx}.", f"{te_idx}.")] = value
                    else:
                        text_encoder_state_dict[te_key_suffix] = value
                else:
                    extra_dict[key] = value

            if self.adalora_unet_model:
                set_peft_model_state_dict(self.adalora_unet_model, unet_state_dict, adapter_name="adalora_unet")
            
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for i, te_model in enumerate(self.adalora_text_encoder_model):
                        single_te_sd = OrderedDict()
                        for k, v in text_encoder_state_dict.items():
                            if k.startswith(f"{i}."):
                                single_te_sd[k.replace(f"{i}.", "")] = v
                            elif not any(key.startswith(f"{j}.") for j in range(len(self.adalora_text_encoder_model))):
                                single_te_sd[k] = v
                        set_peft_model_state_dict(te_model, single_te_sd, adapter_name=f"adalora_te_{i}")
                else:
                    set_peft_model_state_dict(self.adalora_text_encoder_model, text_encoder_state_dict, adapter_name="adalora_te")
            
            if len(extra_dict.keys()) == 0:
                extra_dict = None
            return extra_dict
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
    def _update_torch_multiplier(self: 'LoRASpecialNetwork'):
        if self.is_adalora:
            device = torch.device("cpu")
            dtype = torch.float32
            if hasattr(self, 'adalora_unet_model') and self.adalora_unet_model is not None:
                device = self.adalora_unet_model.device
                dtype = self.adalora_unet_model.dtype
            elif hasattr(self, 'adalora_text_encoder_model') and self.adalora_text_encoder_model is not None:
                if isinstance(self.adalora_text_encoder_model, list) and len(self.adalora_text_encoder_model) > 0:
                    device = self.adalora_text_encoder_model[0].device
                    dtype = self.adalora_text_encoder_model[0].dtype
                elif not isinstance(self.adalora_text_encoder_model, list):
                    device = self.adalora_text_encoder_model.device
                    dtype = self.adalora_text_encoder_model.dtype

            self.torch_multiplier = torch.tensor(1.0).to(device, dtype=dtype)
        else:
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
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float], List[List[float]]]):
        if self._multiplier == value:
            return
        self._multiplier = value
        self._update_torch_multiplier()

    def __enter__(self: 'Network'):
        self.is_active = True

    def __exit__(self: 'Network', exc_type, exc_value, tb):
        self.is_active = False

    def force_to(self: 'Network', device, dtype):
        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)

    def _update_checkpointing(self: 'Network'):
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
        if self.is_adalora:
            if self.adalora_unet_model and hasattr(self.adalora_unet_model, "gradient_checkpointing_enable"):
                self.adalora_unet_model.gradient_checkpointing_enable()
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for te_model in self.adalora_text_encoder_model:
                        if hasattr(te_model, "gradient_checkpointing_enable"):
                            te_model.gradient_checkpointing_enable()
                elif hasattr(self.adalora_text_encoder_model, "gradient_checkpointing_enable"):
                    self.adalora_text_encoder_model.gradient_checkpointing_enable()
        else:
            super().enable_gradient_checkpointing()
            self._update_checkpointing()
    
    def disable_gradient_checkpointing(self: 'LoRASpecialNetwork'):
        self.is_checkpointing = False
        if self.is_adalora:
            if self.adalora_unet_model and hasattr(self.adalora_unet_model, "gradient_checkpointing_disable"):
                self.adalora_unet_model.gradient_checkpointing_disable()
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for te_model in self.adalora_text_encoder_model:
                        if hasattr(te_model, "gradient_checkpointing_disable"):
                            te_model.gradient_checkpointing_disable()
                elif hasattr(self.adalora_text_encoder_model, "gradient_checkpointing_disable"):
                    self.adalora_text_encoder_model.gradient_checkpointing_disable()
        else:
            super().disable_gradient_checkpointing()
            self._update_checkpointing()

    def merge_in(self: 'LoRASpecialNetwork', merge_weight=1.0):
        if self.is_adalora:
            print("Warning: merge_in is not directly supported for AdaLoRA networks via this method. PEFT handles merging internally.")
        else:
            if self.network_type.lower() == 'dora':
                return
            self.is_merged_in = True
            for module in self.get_all_modules():
                if hasattr(module, 'merge_in'):
                    module.merge_in(merge_weight)
    
    def merge_out(self: 'LoRASpecialNetwork', merge_weight=1.0):
        if self.is_adalora:
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
        if extract_mode_param is None:
            raise ValueError("extract_mode_param must be set")
        for module in tqdm(self.get_all_modules(), desc="Extracting weights"):
            if hasattr(module, 'extract_weight'):
                module.extract_weight(
                    extract_mode=extract_mode,
                    extract_mode_param=extract_mode_param
                )

    def setup_lorm(self: 'Network', state_dict: Optional[Dict[str, Any]] = None):
        for module in tqdm(self.get_all_modules(), desc="Extracting LoRM"):
            if hasattr(module, 'setup_lorm'):
                module.setup_lorm(state_dict=state_dict)

    def calculate_lorem_parameter_reduction(self):
        params_reduced = 0
        for module in self.get_all_modules():
            if hasattr(module, 'org_module') and hasattr(module, 'lora_down') and hasattr(module, 'lora_up'):
                num_orig_module_params = count_parameters(module.org_module[0])
                num_lorem_params = count_parameters(module.lora_down) + count_parameters(module.lora_up)
                params_reduced += (num_orig_module_params - num_lorem_params)

        return params_reduced