import copy
import json
import math
import weakref
import os
import re
import fnmatch
import sys
from typing import List, Optional, Dict, Type, Union
import torch
from diffusers import UNet2DConditionModel, PixArtTransformer2DModel, AuraFlowTransformer2DModel, WanTransformer3DModel
from transformers import CLIPTextModel
from toolkit.models.lokr import LokrModule

from .config_modules import NetworkConfig
from .lorm import count_parameters
from .network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin

from toolkit.kohya_lora import LoRANetwork
from toolkit.models.DoRA import DoRAModule
from typing import TYPE_CHECKING

from pathlib import Path

# --- NEW IMPORTS FOR ADALORA ---
from peft import AdaLoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
import peft.tuners.adalora.model  # Import the module for isinstance check

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

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
        self.orig_module_ref = weakref.ref(org_module)
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
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    UNET_TARGET_REPLACE_MODULE = ["UNet2DConditionModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["UNET2DConditionModel"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    PEFT_PREFIX_UNET = "unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
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
            ignore_if_contains=None,
            only_if_contains=None,
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
            network_config: Optional[NetworkConfig] = None,
            **kwargs
    ) -> None:
        """
        LoRA network: すごく引数が多いが、パターンは以下の通り
        1. lora_dimとalphaを指定
        2. lora_dim、alpha、conv_lora_dim、conv_alphaを指定
        3. block_dimsとblock_alphasを指定 :  Conv2d3x3には適用しない
        4. block_dims、block_alphas、conv_block_dims、conv_block_alphasを指定 : Conv2d3x3にも適用する
        5. modules_dimとmodules_alphaを指定 (推論用)
        """
        # 1. Initialize torch.nn.Module first (good practice)
        torch.nn.Module.__init__(self)

        # 2. Store the authoritative network_config
        self.network_config = network_config

        # 3. Create a copy of kwargs for the mixin, removing network_config
        _kwargs_for_mixin = kwargs.copy()
        _kwargs_for_mixin.pop("network_config", None)

        # 4. Call ToolkitNetworkMixin.__init__
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_sdxl=is_sdxl,
            is_v2=is_v2,
            is_lorm=is_lorm,
            base_model_ref=weakref.ref(base_model) if base_model is not None else None,
            network_type=network_type,
            network_config=self.network_config,
            **_kwargs_for_mixin,
        )
        if ignore_if_contains is None:
            ignore_if_contains = []
        self.ignore_if_contains = ignore_if_contains
        self.transformer_only = transformer_only
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
        self.multiplier = multiplier
        self.is_sdxl = is_sdxl
        self.is_ssd = kwargs.get('is_ssd', False)
        self.is_vega = kwargs.get('is_vega', False)
        self.is_v2 = is_v2
        self.is_v3 = is_v3
        self.is_pixart = is_pixart
        self.is_auraflow = is_auraflow
        self.is_flux = is_flux
        self.is_lumina2 = is_lumina2
        self.network_type = network_type
        self.is_assistant_adapter = is_assistant_adapter
        self.peft_adapter_name = "default_adalora_adapter"

        self.peft_adapted_unet = None
        self.peft_adapted_text_encoders = []

        if self.network_type.lower() == "dora":
            self.module_class = DoRAModule
            module_class = DoRAModule
        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule
        else:
            self.module_class = LoRAModule
            module_class = LoRAModule

        self.peft_format = peft_format
        self.is_transformer = is_transformer

        if self.is_flux or self.is_v3 or self.is_lumina2 or is_transformer:
            if self.network_type.lower() != "lokr":
                self.peft_format = True

        if self.peft_format and self.network_type.lower() != "adalora":
            self.alpha = self.lora_dim
            alpha = self.alpha
            self.conv_alpha = self.conv_lora_dim
            conv_alpha = self.conv_alpha

        self.full_train_in_out = full_train_in_out

        # --- ADALORA SPECIFIC INITIALIZATION ---
        if self.network_type.lower() == "adalora":
            print(f"Initializing AdaLoRA network")
            if self.network_config is None:
                raise ValueError("NetworkConfig is required for AdaLoRA")

            # Debug: Inspect UNet structure to find compatible target_modules
            print("Debug: Inspecting UNet structure for target_modules")
            linear_modules = []
            for name, module in unet.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_modules.append(name)
            print(f"Found {len(linear_modules)} Linear modules in UNet: {linear_modules[:10]}...")  # Print first 10 for brevity

            adalora_config = AdaLoraConfig(
                r=self.network_config.adalora_init_r,
                lora_alpha=self.network_config.linear_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj", "norm_out.linear"],  # Refined target_modules
                lora_dropout=self.network_config.dropout if self.network_config.dropout is not None else 0.0,
                tinit=self.network_config.adalora_tinit,
                tfinal=self.network_config.adalora_tfinal,
                deltaT=self.network_config.adalora_deltaT,
                beta1=self.network_config.adalora_beta1,
                beta2=self.network_config.adalora_beta2,
                orth_reg_weight=self.network_config.adalora_orth_reg_weight,
                total_step=self.network_config.adalora_total_step,
                init_lora_weights="lora_only",
            )

            text_encoders_list = text_encoder if isinstance(text_encoder, list) else [text_encoder]
            if train_text_encoder:
                self.peft_adapted_text_encoders = torch.nn.ModuleList()
                for i, te_model in enumerate(text_encoders_list):
                    if (not use_text_encoder_1 and i == 0) or (not use_text_encoder_2 and i == 1):
                        continue
                    print(f"Adapting Text Encoder {i} with AdaLoRA...")
                    te_adalora_config = copy.deepcopy(adalora_config)
                    if hasattr(te_model, 'config') and hasattr(te_model.config, 'model_type') and 'clip' in te_model.config.model_type:
                        te_adalora_config.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                    elif hasattr(te_model, 'config') and hasattr(te_model.config, 'model_type') and 't5' in te_model.config.model_type:
                        te_adalora_config.target_modules = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
                    else:
                        te_adalora_config.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                    te_adapter_name = f"{self.peft_adapter_name}_te_{i}"
                    te_adalora_config.peft_type = "ADALORA"
                    te_peft_model = get_peft_model(te_model, te_adalora_config, te_adapter_name)
                    self.peft_adapted_text_encoders.append(te_peft_model)
                    print(f"Text Encoder {i} AdaLoRA trainable params: {te_peft_model.print_trainable_parameters()}")
                if base_model is not None:
                    base_model.peft_adapted_text_encoders = self.peft_adapted_text_encoders

            if train_unet:
                print(f"Adapting UNet with AdaLoRA...")
                unet_adalora_config = copy.deepcopy(adalora_config)
                unet_adalora_config.target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj", "norm_out.linear"]  # Refined target_modules
                unet_adapter_name = f"{self.peft_adapter_name}_unet"
                unet_adalora_config.peft_type = "ADALORA"
                self.peft_adapted_unet = get_peft_model(unet, unet_adalora_config, unet_adapter_name)
                print(f"UNet AdaLoRA trainable params: {self.peft_adapted_unet.print_trainable_parameters()}")
                if base_model is not None:
                    base_model.peft_adapted_unet = self.peft_adapted_unet

            if self.full_train_in_out and base_model is not None:
                print("AdaLoRA: full train in out layers enabled")
                if self.is_pixart or self.is_auraflow or self.is_flux or self.is_lumina2 or (base_model is not None and base_model.arch == "wan21"):
                    if hasattr(unet, 'pos_embed'):
                        self.transformer_pos_embed = copy.deepcopy(unet.pos_embed)
                        unet.pos_embed = self.transformer_pos_embed
                    elif hasattr(unet, 'patch_embedding') and base_model.arch == "wan21":
                        self.transformer_pos_embed = copy.deepcopy(unet.patch_embedding)
                        unet.patch_embedding = self.transformer_pos_embed
                    if hasattr(unet, 'proj_out'):
                        self.transformer_proj_out = copy.deepcopy(unet.proj_out)
                        unet.proj_out = self.transformer_proj_out
                else:
                    if hasattr(unet, 'conv_in'):
                        unet_conv_in: torch.nn.Conv2d = unet.conv_in
                        self.unet_conv_in = copy.deepcopy(unet_conv_in)
                        unet.conv_in = self.unet_conv_in
                    if hasattr(unet, 'conv_out'):
                        unet_conv_out: torch.nn.Conv2d = unet.conv_out
                        self.unet_conv_out = copy.deepcopy(unet_conv_out)
                        unet.conv_out = self.unet_conv_out

            self.text_encoder_loras = []
            self.unet_loras = []
            print(f"Initialized AdaLoRA for Text Encoder (PEFT): {len(self.peft_adapted_text_encoders)} adapted models.")
            print(f"Initialized AdaLoRA for U-Net (PEFT): {'adapted' if self.peft_adapted_unet else 'not adapted'}.")
            return

        # --- END ADALORA SPECIFIC INITIALIZATION ---

        def create_modules(
                is_unet: bool,
                text_encoder_idx: Optional[int],
                root_module: torch.nn.Module,
                target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            def process_selective_layers():
                expanded_only_if_contains = []
                current_base_model = self.base_model_ref() if self.base_model_ref is not None else None
                if current_base_model is None:
                    print("Warning: base_model is not available for selective layer processing. Skipping only_if_contains filter.")
                    return []
                transformer_block_handles = current_base_model.get_transformer_block_names()
                for layer in self.only_if_contains:
                    for handle in transformer_block_handles:
                        module_list = getattr(root_module, handle)
                        layers_list = [name for name, _ in module_list.named_modules()]
                        try:
                            pattern = re.compile(layer)
                        except re.error:
                            pattern = re.compile(fnmatch.translate(layer))
                        matched_layers = [f"transformer.{handle}.{item}" for item in layers_list if pattern.match(item)]
                        expanded_only_if_contains += matched_layers
                self.only_if_contains = list(set(expanded_only_if_contains))
                print(f"Filtered only_if_contains layers: {self.only_if_contains}")
                return self.only_if_contains

            if isinstance(self.only_if_contains, List):
                processed_only_if_contains = process_selective_layers()
                self.only_if_contains = processed_only_if_contains
            elif isinstance(self.only_if_contains, str):
                layer_regex_path = Path(self.only_if_contains)
                if layer_regex_path.exists():
                    self.only_if_contains = layer_regex_path.read_text(encoding="utf-8").splitlines()
                    processed_only_if_contains = process_selective_layers()
                    self.only_if_contains = processed_only_if_contains
                else:
                    print(f"Warning: only_if_contains path '{self.only_if_contains}' does not exist. Skipping selective layer processing.")
                    self.only_if_contains = None

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
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ in LINEAR_MODULES
                        is_conv2d = child_module.__class__.__name__ in CONV_MODULES
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                        lora_name = [prefix, name, child_name]
                        lora_name = [x for x in lora_name if x and x != ""]
                        lora_name = ".".join(lora_name)
                        lora_name = lora_name.replace("..", ".")
                        clean_name = lora_name
                        # Sanitize lora_name to replace dots with underscores for PyTorch module compatibility
                        sanitized_lora_name = clean_name.replace(".", "_")
                        skip = False
                        if any([word in clean_name for word in self.ignore_if_contains]):
                            skip = True
                        if count_parameters(child_module) < parameter_threshold:
                            skip = True
                        if self.transformer_only and is_unet:
                            transformer_block_names = None
                            current_base_model = self.base_model_ref() if self.base_model_ref is not None else None
                            if current_base_model is not None:
                                transformer_block_names = current_base_model.get_transformer_block_names()
                            if transformer_block_names is not None:
                                if not any([name in clean_name for name in transformer_block_names]):
                                    skip = True
                            else:
                                if self.is_pixart and "transformer_blocks" not in clean_name:
                                    skip = True
                                if self.is_flux and "transformer_blocks" not in clean_name:
                                    skip = True
                                if self.is_lumina2 and "layers$$" not in clean_name and "noise_refiner$$" not in clean_name and "context_refiner$$" not in clean_name:
                                    skip = True
                                if self.is_v3 and "transformer_blocks" not in clean_name:
                                    skip = True
                                if hasattr(root_module, 'transformer_blocks') and "transformer_blocks" not in clean_name:
                                    skip = True
                                if hasattr(root_module, 'blocks') and "blocks" not in clean_name:
                                    skip = True
                                if hasattr(root_module, 'single_blocks') and "single_blocks" not in clean_name and "double_blocks" not in clean_name:
                                    skip = True
                        if (is_linear or is_conv2d) and not skip:
                            if self.only_if_contains is not None:
                                if not any([re.search(word, clean_name) for word in self.only_if_contains]) and \
                                   not any([re.search(word, clean_name) for word in self.only_if_contains]):
                                    continue
                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if clean_name in modules_dim:
                                    dim = modules_dim[clean_name]
                                    alpha = modules_alpha[clean_name]
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha
                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(clean_name)
                                continue
                            module_kwargs = {}
                            if self.network_type.lower() == "lokr":
                                module_kwargs["factor"] = self.network_config.lokr_factor
                            lora = module_class(
                                sanitized_lora_name,  # Use sanitized name for PyTorch module
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **module_kwargs
                            )
                            loras.append(lora)
                            if self.network_type.lower() == "lokr":
                                try:
                                    lora_shape_dict[clean_name] = [list(lora.lokr_w1.weight.shape), list(lora.lokr_w2.weight.shape)]
                                except:
                                    pass
                            else:
                                lora_shape_dict[clean_name] = [list(lora.lora_down.weight.shape), list(lora.lora_up.weight.shape)]
            return loras, skipped

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self.text_encoder_loras = []
        skipped_te = []
        if train_text_encoder:
            for i, te_model in enumerate(text_encoders):
                if not use_text_encoder_1 and i == 0:
                    continue
                if not use_text_encoder_2 and i == 1:
                    continue
                if len(text_encoders) > 1:
                    index = i + 1
                    print(f"create LoRA for Text Encoder {index}:")
                else:
                    index = None
                    print(f"create LoRA for Text Encoder:")
                replace_modules = LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                if self.is_pixart:
                    replace_modules = ["T5EncoderModel"]
                text_encoder_loras, skipped = create_modules(False, index, te_model, replace_modules)
                self.text_encoder_loras.extend(text_encoder_loras)
                skipped_te += skipped
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        target_modules = target_lin_modules
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += target_conv_modules
        if is_v3:
            target_modules = ["SD3Transformer2DModel"]
        if is_pixart:
            target_modules = ["PixArtTransformer2DModel"]
        if is_auraflow:
            target_modules = ["AuraFlowTransformer2DModel"]
        if is_flux:
            target_modules = ["FluxTransformer2DModel"]
        if is_lumina2:
            target_modules = ["Lumina2Transformer2DModel"]
        if train_unet:
            self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
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
            if self.is_pixart:
                transformer: PixArtTransformer2DModel = unet
                if hasattr(transformer, 'pos_embed'):
                    self.transformer_pos_embed = copy.deepcopy(transformer.pos_embed)
                    transformer.pos_embed = self.transformer_pos_embed
                if hasattr(transformer, 'proj_out'):
                    self.transformer_proj_out = copy.deepcopy(transformer.proj_out)
                    transformer.proj_out = self.transformer_proj_out
            elif self.is_auraflow:
                transformer: AuraFlowTransformer2DModel = unet
                if hasattr(transformer, 'pos_embed'):
                    self.transformer_pos_embed = copy.deepcopy(transformer.pos_embed)
                    transformer.pos_embed = self.transformer_pos_embed
                if hasattr(transformer, 'proj_out'):
                    self.transformer_proj_out = copy.deepcopy(transformer.proj_out)
                    transformer.proj_out = self.transformer_proj_out
            elif base_model is not None and base_model.arch == "wan21":
                transformer: WanTransformer3DModel = unet
                if hasattr(transformer, 'patch_embedding'):
                    self.transformer_pos_embed = copy.deepcopy(transformer.patch_embedding)
                    transformer.patch_embedding = self.transformer_pos_embed
                if hasattr(transformer, 'proj_out'):
                    self.transformer_proj_out = copy.deepcopy(transformer.proj_out)
                    transformer.proj_out = self.transformer_proj_out
            else:
                if hasattr(unet, 'conv_in'):
                    unet_conv_in: torch.nn.Conv2d = unet.conv_in
                    self.unet_conv_in = copy.deepcopy(unet_conv_in)
                    unet.conv_in = self.unet_conv_in
                if hasattr(unet, 'conv_out'):
                    unet_conv_out: torch.nn.Conv2d = unet.conv_out
                    self.unet_conv_out = copy.deepcopy(unet_conv_out)
                    unet.conv_out = self.unet_conv_out

    def apply_to(self, text_encoder: Union[List[CLIPTextModel], CLIPTextModel], unet, **kwargs):
        # For AdaLoRA, the models are already adapted in __init__, so skip
        if self.network_type.lower() == "adalora":
            print("AdaLoRA models already adapted during initialization. Skipping apply_to.")
            return

        # Filter out kwargs that LoRANetwork.apply_to does not accept
        valid_kwargs = {}
        for key in kwargs:
            if key in ['multiplier']:  # LoRANetwork.apply_to accepts multiplier
                valid_kwargs[key] = kwargs[key]

        # Call parent LoRANetwork.apply_to with only valid arguments
        super().apply_to(text_encoder, unet, **valid_kwargs)

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        all_params = []

        if self.network_type.lower() == "adalora":
            print("AdaLoRA: Preparing optimizer parameters from PEFT adapted models.")
            if self.peft_adapted_unet:
                unet_params = [p for p in self.peft_adapted_unet.parameters() if p.requires_grad]
                if unet_params:
                    all_params.append({"lr": unet_lr, "params": unet_params, "weight_decay": 0.0})
                # Debug: Check for parameters with requires_grad=True
                print(f"UNet trainable parameters: {len(unet_params)}")
                for name, param in self.peft_adapted_unet.named_parameters():
                    if param.requires_grad:
                        print(f"UNet trainable param: {name}, shape: {param.shape}")
            for i, te_model in enumerate(self.peft_adapted_text_encoders):
                te_params = [p for p in te_model.parameters() if p.requires_grad]
                if te_params:
                    all_params.append({"lr": text_encoder_lr, "params": te_params, "weight_decay": 0.0})
                # Debug: Check for parameters with requires_grad=True
                print(f"Text Encoder {i} trainable parameters: {len(te_params)}")
                for name, param in te_model.named_parameters():
                    if param.requires_grad:
                        print(f"Text Encoder {i} trainable param: {name}, shape: {param.shape}")
            if self.full_train_in_out:
                base_model = self.base_model_ref() if self.base_model_ref is not None else None
                if self.is_pixart or self.is_auraflow or self.is_flux or self.is_lumina2 or (base_model is not None and base_model.arch == "wan21"):
                    if hasattr(self, 'transformer_pos_embed') and self.transformer_pos_embed is not None:
                        pos_embed_params = list(self.transformer_pos_embed.parameters())
                        all_params.append({"lr": unet_lr, "params": pos_embed_params})
                        print(f"Transformer pos_embed trainable parameters: {len(pos_embed_params)}")
                    if hasattr(self, 'transformer_proj_out') and self.transformer_proj_out is not None:
                        proj_out_params = list(self.transformer_proj_out.parameters())
                        all_params.append({"lr": unet_lr, "params": proj_out_params})
                        print(f"Transformer proj_out trainable parameters: {len(proj_out_params)}")
                else:
                    if hasattr(self, 'unet_conv_in') and self.unet_conv_in is not None:
                        conv_in_params = list(self.unet_conv_in.parameters())
                        all_params.append({"lr": unet_lr, "params": conv_in_params})
                        print(f"UNet conv_in trainable parameters: {len(conv_in_params)}")
                    if hasattr(self, 'unet_conv_out') and self.unet_conv_out is not None:
                        conv_out_params = list(self.unet_conv_out.parameters())
                        all_params.append({"lr": unet_lr, "params": conv_out_params})
                        print(f"UNet conv_out trainable parameters: {len(conv_out_params)}")
            return all_params

        all_params = super().prepare_optimizer_params(text_encoder_lr, unet_lr, default_lr)
        if self.full_train_in_out:
            base_model = self.base_model_ref() if self.base_model_ref is not None else None
            if self.is_pixart or self.is_auraflow or self.is_flux or self.is_lumina2 or (base_model is not None and base_model.arch == "wan21"):
                if hasattr(self, 'transformer_pos_embed') and self.transformer_pos_embed is not None:
                    all_params.append({"lr": unet_lr, "params": list(self.transformer_pos_embed.parameters())})
                if hasattr(self, 'transformer_proj_out') and self.transformer_proj_out is not None:
                    all_params.append({"lr": unet_lr, "params": list(self.transformer_proj_out.parameters())})
            else:
                if hasattr(self, 'unet_conv_in') and self.unet_conv_in is not None:
                    all_params.append({"lr": unet_lr, "params": list(self.unet_conv_in.parameters())})
                if hasattr(self, 'unet_conv_out') and self.unet_conv_out is not None:
                    all_params.append({"lr": unet_lr, "params": list(self.unet_conv_out.parameters())})
        return all_params

    def update_and_allocate_adalora(self, step_num: int):
        if self.network_type.lower() != "adalora":
            return
        unet_adapter_name = f"{self.peft_adapter_name}_unet"
        if self.peft_adapted_unet and unet_adapter_name in self.peft_adapted_unet.peft_config:
            adalora_tuner_unet = self.peft_adapted_unet.base_model
            if isinstance(adalora_tuner_unet, peft.tuners.adalora.model.AdaLoraModel):
                try:
                    adalora_tuner_unet.update_and_allocate(step_num)
                except TypeError as e:
                    if "unsupported operand type(s) for *" in str(e):
                        print(f"Warning: Skipping UNet update_and_allocate due to None gradients at step {step_num}: {e}")
                        none_grad_params = [(name, param) for name, param in adalora_tuner_unet.named_parameters() if param.grad is None and param.requires_grad]
                        for name, param in none_grad_params:
                            print(f"Parameter with None gradient: {name}, shape: {param.shape}")
                        if not none_grad_params:
                            print("No parameters with None gradients found.")
                    else:
                        raise e
            else:
                print(f"Error: UNet tuner (via .base_model) for adapter '{unet_adapter_name}' is not an AdaLoraModel instance but {type(adalora_tuner_unet)}. Cannot call update_and_allocate.")
        for i, te_model in enumerate(self.peft_adapted_text_encoders):
            te_adapter_name = f"{self.peft_adapter_name}_te_{i}"
            if te_adapter_name in te_model.peft_config:
                adalora_tuner_te = te_model.base_model
                if isinstance(adalora_tuner_te, peft.tuners.adalora.model.AdaLoraModel):
                    try:
                        adalora_tuner_te.update_and_allocate(step_num)
                    except TypeError as e:
                        if "unsupported operand type(s) for *" in str(e):
                            print(f"Warning: Skipping Text Encoder {i} update_and_allocate due to None gradients at step {step_num}: {e}")
                            none_grad_params = [(name, param) for name, param in adalora_tuner_te.named_parameters() if param.grad is None and param.requires_grad]
                            for name, param in none_grad_params:
                                print(f"Parameter with None gradient: {name}, shape: {param.shape}")
                            if not none_grad_params:
                                print("No parameters with None gradients found.")
                        else:
                            raise e
                else:
                    print(f"Error: Text Encoder {i} tuner (via .base_model) for adapter '{te_adapter_name}' is not an AdaLoraModel instance but {type(adalora_tuner_te)}. Cannot call update_and_allocate.")

    def get_adalora_rank_pattern(self):
        if self.network_type.lower() != "adalora":
            return {}
        rank_patterns = {}
        unet_adapter_name = f"{self.peft_adapter_name}_unet"
        if self.peft_adapted_unet and unet_adapter_name in self.peft_adapted_unet.peft_config:
            adalora_tuner_unet = self.peft_adapted_unet.base_model
            if isinstance(adalora_tuner_unet, peft.tuners.adalora.model.AdaLoraModel):
                try:
                    rank_patterns['unet'] = adalora_tuner_unet.get_nb_scaled_parameters_per_rank()
                except AttributeError:
                    print(f"Warning: get_nb_scaled_parameters_per_rank not supported in this version of peft (version {peft.__version__}). Skipping rank pattern logging for UNet.")
                    rank_patterns['unet'] = {}
            else:
                print(f"Error: UNet tuner (via .base_model) for adapter '{unet_adapter_name}' is not an AdaLoraModel instance but {type(adalora_tuner_unet)}. Cannot retrieve rank pattern.")
                rank_patterns['unet'] = {}
        for i, te_model in enumerate(self.peft_adapted_text_encoders):
            te_adapter_name = f"{self.peft_adapter_name}_te_{i}"
            if te_adapter_name in te_model.peft_config:
                adalora_tuner_te = te_model.base_model
                if isinstance(adalora_tuner_te, peft.tuners.adalora.model.AdaLoraModel):
                    try:
                        rank_patterns[f'text_encoder_{i}'] = adalora_tuner_te.get_nb_scaled_parameters_per_rank()
                    except AttributeError:
                        print(f"Warning: get_nb_scaled_parameters_per_rank not supported in this version of peft (version {peft.__version__}). Skipping rank pattern logging for Text Encoder {i}.")
                        rank_patterns[f'text_encoder_{i}'] = {}
                else:
                    print(f"Error: Text Encoder {i} tuner (via .base_model) for adapter '{te_adapter_name}' is not an AdaLoraModel instance but {type(adalora_tuner_te)}. Cannot retrieve rank pattern.")
                    rank_patterns[f'text_encoder_{i}'] = {}
        return rank_patterns