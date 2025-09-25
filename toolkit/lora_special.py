import copy
import json
import math
import weakref
import os
import re
import fnmatch
import sys
from collections import OrderedDict 
from typing import List, Optional, Dict, Type, Union, Any
import torch
from diffusers import UNet2DConditionModel, PixArtTransformer2DModel, AuraFlowTransformer2DModel, WanTransformer3DModel
from transformers import CLIPTextModel
from toolkit.paths import KEYMAPS_ROOT
from toolkit.models.lokr import LokrModule
from toolkit.models.DoRA import DoRAModule
from toolkit.metadata import add_model_hash_to_meta

from .config_modules import NetworkConfig
from .lorm import count_parameters
from .network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin

from toolkit.kohya_lora import LoRANetwork
from typing import TYPE_CHECKING

from pathlib import Path


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
            # --- START LoRA+ additions to LoRAModule __init__ ---
            loraplus_enabled: bool = False,
            loraplus_lambda_lr: float = 1.0,
            # --- END LoRA+ additions ---
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

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
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

        # --- START LoRA+ additions to LoRAModule properties ---
        self.loraplus_enabled = loraplus_enabled
        self.loraplus_lambda_lr = loraplus_lambda_lr
        # --- END LoRA+ additions ---

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        # del self.org_module


class LoRASpecialNetwork(ToolkitNetworkMixin, LoRANetwork):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    # UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "ResnetBlock2D"]
    UNET_TARGET_REPLACE_MODULE = ["UNet2DConditionModel"]
    # UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["UNet2DConditionModel"]
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
            # --- START LoRA+ additions to LoRASpecialNetwork __init__ ---
            loraplus_enabled: bool = False,
            loraplus_lambda_lr: float = 1.0,
            # --- END LoRA+ additions ---
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
        # call the parent of the parent we are replacing (LoRANetwork) init
        torch.nn.Module.__init__(self)
        _network_config_from_kwargs = kwargs.pop("network_config", None)
        self.network_config: NetworkConfig = _network_config_from_kwargs
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_sdxl=is_sdxl,
            is_v2=is_v2,
            is_lorm=is_lorm,
            network_config=self.network_config, # Pass it to the mixin
            **kwargs
        )

        if ignore_if_contains is None:
            ignore_if_contains = []
        self.ignore_if_contains = ignore_if_contains
        self.transformer_only = transformer_only
        self.base_model_ref = None
        if base_model is not None:
            self.base_model_ref = weakref.ref(base_model)

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
        # triggers the state updates
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
        if self.network_type.lower() == "dora":
            self.module_class = DoRAModule
            module_class = DoRAModule
        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule
        # self.network_config: NetworkConfig = kwargs.get("network_config", None) # Removed, handled above

        self.peft_format = peft_format
        self.is_transformer = is_transformer
        
        # --- START LoRA+ additions to LoRASpecialNetwork properties ---
        self.loraplus_enabled = loraplus_enabled
        self.loraplus_lambda_lr = loraplus_lambda_lr
        # --- END LoRA+ additions ---

        # always do peft for flux only for now
        if self.is_flux or self.is_v3 or self.is_lumina2 or is_transformer:
            # don't do peft format for lokr
            if self.network_type.lower() != "lokr":
                self.peft_format = True

        if self.peft_format:
            # no alpha for peft
            self.alpha = self.lora_dim
            alpha = self.alpha
            self.conv_alpha = self.conv_lora_dim
            conv_alpha = self.conv_alpha

        self.full_train_in_out = full_train_in_out

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        elif block_dims is not None:
            print(f"create LoRA network from block_dims")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            print(f"block_dims: {block_dims}")
            print(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                print(f"conv_block_dims: {conv_block_dims}")
                print(f"conv_block_alphas: {conv_block_alphas}")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            print(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")
            if self.conv_lora_dim is not None:
                print(
                    f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

        # create module instances
        def create_modules(
                is_unet: bool,
                text_encoder_idx: Optional[int],  # None, 1, 2
                root_module: torch.nn.Module,
                target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:


            #############
            ### need to handle regex for self.only_if_contains!!!
            def process_selective_layers ():
                expanded_only_if_contains = []
                # Ensure base_model is not None before calling its method
                if self.base_model_ref is not None:
                    base_model_instance = self.base_model_ref()
                    if base_model_instance is not None:
                        transformer_block_handles = base_model_instance.get_transformer_block_names()
                    else:
                        transformer_block_handles = [] # Fallback if ref is dead
                else:
                    transformer_block_handles = [] # Fallback if base_model_ref is None

                for layer in self.only_if_contains:
                    for handle in transformer_block_handles:
                        module_list = getattr(root_module, handle) 
                        layers_list = [name for name, _ in module_list.named_modules()]
                        # pattern = re.compile(layer)
                        try:
                            pattern = re.compile(layer)
                        except re.error:
                            pattern = re.compile(fnmatch.translate(layer))
                        # Get matching layers here!!!!!
                        matched_layers = [f"transformer.{handle}.{item}" for item in layers_list if pattern.match(item)]
                        expanded_only_if_contains += matched_layers

                self.only_if_contains = list(set(expanded_only_if_contains))
                print(self.only_if_contains)


            if isinstance(self.only_if_contains, List):
                process_selective_layers()

            elif isinstance(self.only_if_contains, str): ## it will be a path to the txt file
                layer_regex_path = Path(self.only_if_contains)
                if layer_regex_path.exists():
                    self.only_if_contains = layer_regex_path.read_text(encoding="utf-8").splitlines()
                    process_selective_layers()
                else:
                    print(f"Warning: only_if_contains path '{self.only_if_contains}' does not exist. Skipping selective layers.")
                    self.only_if_contains = None # Clear if file not found


            #############



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
            attached_modules = []
            lora_shape_dict = {}
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ in LINEAR_MODULES
                        is_conv2d = child_module.__class__.__name__ in CONV_MODULES
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)


                        lora_name = [prefix, name, child_name]
                        # filter out blank
                        lora_name = [x for x in lora_name if x and x != ""]
                        lora_name = ".".join(lora_name)
                        # if it doesnt have a name, it wil have two dots
                        lora_name = lora_name.replace("..", ".") # Corrected: use lora_name here
                        clean_name = lora_name
                        if self.peft_format:
                            # we replace this on saving
                            lora_name = lora_name.replace(".", "$$")
                        else:
                            lora_name = lora_name.replace(".", "_")

                        skip = False
                        if any([word in clean_name for word in self.ignore_if_contains]):
                            skip = True

                        # see if it is over threshold
                        if count_parameters(child_module) < parameter_threshold:
                            skip = True
                        
                        if self.transformer_only and is_unet:
                            transformer_block_names = None
                            # Ensure base_model is not None before calling its method
                            if self.base_model_ref is not None:
                                base_model_instance = self.base_model_ref()
                                if base_model_instance is not None:
                                    transformer_block_names = base_model_instance.get_transformer_block_names()
                                else:
                                    transformer_block_names = []
                            else:
                                transformer_block_names = []
                            
                            if transformer_block_names is not None and any(transformer_block_names): # Check if list is not empty
                                if not any([name_part in lora_name for name_part in transformer_block_names]):
                                    skip = True
                            else: # Fallback if transformer_block_names is not available or empty
                                if self.is_pixart:
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                if self.is_flux:
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                if self.is_lumina2:
                                    if "layers$$" not in lora_name and "noise_refiner$$" not in lora_name and "context_refiner$$" not in lora_name:
                                        skip = True
                                if  self.is_v3:
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                
                                # handle custom models
                                if hasattr(root_module, 'transformer_blocks'):
                                    if "transformer_blocks" not in lora_name:
                                        skip = True
                                        
                                if hasattr(root_module, 'blocks'):
                                    if "blocks" not in lora_name:
                                        skip = True
                                
                                if hasattr(root_module, 'single_blocks'):
                                    if "single_blocks" not in lora_name and "double_blocks" not in lora_name:
                                        skip = True

                        if (is_linear or is_conv2d) and not skip:

                            if self.only_if_contains is not None:
                                # Ensure self.only_if_contains is iterable
                                if not isinstance(self.only_if_contains, (list, tuple)):
                                    # If it somehow got set to a non-iterable, clear it or log error
                                    print(f"Warning: self.only_if_contains is not a list/tuple: {self.only_if_contains}. Clearing.")
                                    self.only_if_contains = None
                                
                                if self.only_if_contains: # Check if list is not empty
                                    if not any([word in clean_name for word in self.only_if_contains]) and not any([word in lora_name for word in self.only_if_contains]):
                                        continue

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (
                                        self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue
                            
                            module_kwargs = {}
                            
                            if self.network_type.lower() == "lokr":
                                module_kwargs["factor"] = self.network_config.lokr_factor

                            lora = module_class(
                                lora_name,
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
                                # --- START LoRA+ additions to LoRAModule init ---
                                loraplus_enabled=self.loraplus_enabled,
                                loraplus_lambda_lr=self.loraplus_lambda_lr,
                                # --- END LoRA+ additions ---
                                **module_kwargs
                            )
                            loras.append(lora)
                            if self.network_type.lower() == "lokr":
                                try:
                                    lora_shape_dict[lora_name] = [list(lora.lokr_w1.weight.shape), list(lora.lokr_w2.weight.shape)]
                                except:
                                    pass
                            else:
                                lora_shape_dict[lora_name] = [list(lora.lora_down.weight.shape), list(lora.lora_up.weight.shape)]
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討
        self.text_encoder_loras = []
        skipped_te = []
        if train_text_encoder:
            for i, text_encoder in enumerate(text_encoders):
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

                text_encoder_loras, skipped = create_modules(False, index, text_encoder, replace_modules)
                self.text_encoder_loras.extend(text_encoder_loras)
                skipped_te += skipped
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
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

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        if self.full_train_in_out:
            print("full train in out")
            # we are going to retrain the main in out layers for VAE change usually
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

                # clone these and replace their forwards with ours
                self.unet_conv_in = copy.deepcopy(unet_conv_in)
                self.unet_conv_out = copy.deepcopy(unet_conv_out)
                unet.conv_in = self.unet_conv_in
                unet.conv_out = self.unet_conv_out

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        # --- START LoRA+ additions to prepare_optimizer_params ---
        if self.loraplus_enabled:
            print(f"LoRA+ enabled: Applying different learning rates for LoRA_A (down) and LoRA_B (up) with lambda_lr={self.loraplus_lambda_lr}.")
            optimizer_grouped_parameters = []

            # Parameters for LoRA_down (A matrix) and LoRA_up (B matrix)
            lora_down_params = []
            lora_up_params = []
            
            # Parameters for other network-specific components that don't fit A/B model (e.g., DoRA's magnitude)
            other_lora_params = []

            for lora_module in self.text_encoder_loras + self.unet_loras:
                # Standard LoRA/LoCon parameters
                if hasattr(lora_module, 'lora_down') and lora_module.lora_down is not None:
                    lora_down_params.extend(lora_module.lora_down.parameters())
                if hasattr(lorora_module, 'lora_up') and lora_module.lora_up is not None:
                    lora_up_params.extend(lora_module.lora_up.parameters())
                
                # DoRA specific parameters (magnitude)
                if isinstance(lora_module, DoRAModule) and hasattr(lora_module, 'magnitude'):
                    if lora_module.magnitude is not None:
                        other_lora_params.append(lora_module.magnitude)
                
                # LoKR specific parameters
                if isinstance(lora_module, LokrModule):
                    # For LoKR, lokr_w1 (and w1_a) can be considered the 'down' part (A)
                    # and lokr_w2 (and w2_a) as the 'up' part (B) for LoRA+ analogy.
                    if hasattr(lora_module, 'lokr_w1') and lora_module.lokr_w1 is not None:
                        lora_down_params.extend(lora_module.lokr_w1.parameters())
                    if hasattr(lora_module, 'lokr_w1_a') and lora_module.lokr_w1_a is not None:
                        lora_down_params.extend(lora_module.lokr_w1_a.parameters()) # If present
                    
                    if hasattr(lora_module, 'lokr_w2') and lora_module.lokr_w2 is not None:
                        lora_up_params.extend(lora_module.lokr_w2.parameters())
                    if hasattr(lora_module, 'lokr_w2_a') and lora_module.lokr_w2_a is not None:
                        lora_up_params.extend(lora_module.lokr_w2_a.parameters()) # If present
                
                # If there are any other parameters in a LoRA-like module that should be included
                # and don't fit the A/B split, add them to other_lora_params here.

            # Grouping parameters for optimizer
            # Apply base LR to 'down' components (A matrix)
            if len(lora_down_params) > 0:
                optimizer_grouped_parameters.append({"params": lora_down_params, "lr": default_lr})
            
            # Apply lambda-scaled LR to 'up' components (B matrix)
            if len(lora_up_params) > 0:
                optimizer_grouped_parameters.append({"params": lora_up_params, "lr": default_lr * self.loraplus_lambda_lr})
            
            # Apply base LR to other LoRA-related parameters
            if len(other_lora_params) > 0:
                optimizer_grouped_parameters.append({"params": other_lora_params, "lr": default_lr})

            # Add parameters from `full_train_in_out` if enabled, maintaining existing logic
            # These parameters are not part of the LoRA A/B matrices, so they get the unet_lr
            if self.full_train_in_out:
                base_model = self.base_model_ref() if self.base_model_ref is not None else None
                if self.is_pixart or self.is_auraflow or self.is_flux or (base_model is not None and base_model.arch == "wan21"):
                    optimizer_grouped_parameters.append({"lr": unet_lr, "params": list(self.transformer_pos_embed.parameters())})
                    optimizer_grouped_parameters.append({"lr": unet_lr, "params": list(self.transformer_proj_out.parameters())})
                else:
                    optimizer_grouped_parameters.append({"lr": unet_lr, "params": list(self.unet_conv_in.parameters())})
                    optimizer_grouped_parameters.append({"lr": unet_lr, "params": list(self.unet_conv_out.parameters())})

            return optimizer_grouped_parameters
        else:
            # If LoRA+ is not enabled, fall back to the original `ToolkitNetworkMixin`'s `prepare_optimizer_params`
            # This ensures original functionality is undisturbed when LoRA+ is off.
            return super().prepare_optimizer_params(text_encoder_lr, unet_lr, default_lr)
        # --- END LoRA+ additions ---

    def get_keymap(self: 'LoRASpecialNetwork', force_weight_mapping=False):
        use_weight_mapping = False

        if self.is_ssd:
            keymap_tail = 'ssd'
            use_weight_mapping = True
        elif self.is_vega:
            keymap_tail = 'vega'
            use_weight_mapping = True
        elif self.is_sdxl:
            keymap_tail = 'sdxl'
        elif self.is_v2:
            keymap_tail = 'sd2'
        else:
            keymap_tail = 'sd1'
            # todo double check this
            # use_weight_mapping = True

        if force_weight_mapping:
            use_weight_mapping = True

        # load keymap
        keymap_name = f"stable_diffusion_locon_{keymap_tail}.json"
        if use_weight_mapping:
            keymap_name = f"stable_diffusion_{keymap_tail}.json"

        keymap_path = os.path.join(KEYMAPS_ROOT, keymap_name)

        keymap = None
        # check if file exists
        if os.path.exists(keymap_path):
            with open(keymap_path, 'r') as f:
                keymap = json.load(f)['ldm_diffusers_keymap']

        if use_weight_mapping and keymap is not None:
            # get keymap from weights
            keymap = get_lora_keymap_from_model_keymap(keymap)

        # upgrade keymaps for DoRA
        if self.network_type.lower() == 'dora':
            if keymap is not None:
                new_keymap = {}
                for ldm_key, diffusers_key in keymap.items():
                    ldm_key = ldm_key.replace('.alpha', '.magnitude')
                    # ldm_key = ldm_key.replace('.lora_down.weight', '.lora_down')
                    # ldm_key = ldm_key.replace('.lora_up.weight', '.lora_up')

                    diffusers_key = diffusers_key.replace('.alpha', '.magnitude')
                    # diffusers_key = diffusers_key.replace('.lora_down.weight', '.lora_down')
                    # diffusers_key = diffusers_key.replace('.lora_up.weight', '.lora_up')

                    new_keymap[ldm_key] = diffusers_key

                keymap = new_keymap

        return keymap
    
    def get_state_dict(self: 'LoRASpecialNetwork', extra_state_dict=None, dtype=torch.float16):
        keymap = self.get_keymap()

        save_keymap = {}
        if keymap is not None:
            for ldm_key, diffusers_key in keymap.items():
                #  invert them
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
            # add extra items to state dict
            for key in list(extra_state_dict.keys()):
                v = extra_state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_dict[key] = v

        if self.peft_format:
            # lora_down = lora_A
            # lora_up = lora_B
            # no alpha

            new_save_dict = {}
            for key, value in save_dict.items():
                if key.endswith('.alpha'):
                    continue
                new_key = key
                new_key = new_key.replace('lora_down', 'lora_A')
                new_key = new_key.replace('lora_up', 'lora_B')
                # replace all $$ with .
                new_key = new_key.replace('$$', '.')
                new_save_dict[new_key] = value

            save_dict = new_save_dict
        
                
        if self.network_type.lower() == "lokr":
            new_save_dict = {}
            for key, value in save_dict.items():
                # lora_transformer_transformer_blocks_7_attn_to_v.lokr_w1 to lycoris_transformer_blocks_7_attn_to_v.lokr_w1
                new_key = key
                new_key = new_key.replace('lora_transformer_', 'lycoris_')
                new_save_dict[new_key] = value

            save_dict = new_save_dict
        
        if self.base_model_ref is not None:
            base_model_instance = self.base_model_ref()
            if base_model_instance is not None:
                save_dict = base_model_instance.convert_lora_weights_before_save(save_dict)
        return save_dict

    def save_weights(
            self: 'LoRASpecialNetwork',
            file, dtype=torch.float16,
            metadata=None,
            extra_state_dict: Optional[OrderedDict] = None
    ):
        save_dict = self.get_state_dict(extra_state_dict=extra_state_dict, dtype=dtype)
        
        if metadata is not None and len(metadata) == 0:
            metadata = None

        if metadata is None:
            metadata = OrderedDict()
        metadata = add_model_hash_to_meta(save_dict, metadata)
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(save_dict, file, metadata)
        else:
            torch.save(save_dict, file)

    def load_weights(self: 'LoRASpecialNetwork', file, force_weight_mapping=False):
        # allows us to save and load to and from ldm weights
        keymap = self.get_keymap(force_weight_mapping)
        keymap = {} if keymap is None else keymap

        if isinstance(file, str):
            if os.path.splitext(file)[1] == ".safetensors":
                from safetensors.torch import load_file

                weights_sd = load_file(file)
            else:
                weights_sd = torch.load(file, map_location="cpu")
        else:
            # probably a state dict
            weights_sd = file
        
        if self.base_model_ref is not None:
            base_model_instance = self.base_model_ref()
            if base_model_instance is not None:
                weights_sd = base_model_instance.convert_lora_weights_before_load(weights_sd)

        load_sd = OrderedDict()
        for key, value in weights_sd.items():
            load_key = keymap[key] if key in keymap else key
            # replace old double __ with single _
            if self.is_pixart:
                load_key = load_key.replace('__', '_')

            if self.peft_format:
                # lora_down = lora_A
                # lora_up = lora_B
                # no alpha
                if load_key.endswith('.alpha'):
                    continue
                load_key = load_key.replace('lora_A', 'lora_down')
                load_key = load_key.replace('lora_B', 'lora_up')
                # replace all . with $$
                load_key = load_key.replace('.', '$$')
                load_key = load_key.replace('$$lora_down$$', '.lora_down.')
                load_key = load_key.replace('$$lora_up$$', '.lora_up.')
            
            if self.network_type.lower() == "lokr":
                # lora_transformer_transformer_blocks_7_attn_to_v.lokr_w1 to lycoris_transformer_blocks_7_attn_to_v.lokr_w1
                load_key = load_key.replace('lycoris_', 'lora_transformer_')

            load_sd[load_key] = value

        # extract extra items from state dict
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
        # builds a tensor for fast usage in the forward pass of the network modules
        # without having to set it in every single module every time it changes
        multiplier = self._multiplier
        # get first module
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
        # it takes time to update all the multipliers, so we only do it if the value has changed
        if self._multiplier == value:
            return
        # if we are setting a single value but have a list, keep the list if every item is the same as value
        self._multiplier = value
        self._update_torch_multiplier()

    # called when the context manager is entered
    # ie: with network:
    def __enter__(self: 'LoRASpecialNetwork'):
        self.is_active = True

    def __exit__(self: 'LoRASpecialNetwork', exc_type, exc_value, tb):
        self.is_active = False

    def force_to(self: 'LoRASpecialNetwork', device, dtype):
        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)

    def get_all_modules(self: 'LoRASpecialNetwork') -> List['LoRAModule']:
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        return loras

    def _update_checkpointing(self: 'LoRASpecialNetwork'):
        for module in self.get_all_modules():
            if self.is_checkpointing:
                module.enable_gradient_checkpointing()
            else:
                module.disable_gradient_checkpointing()

    def enable_gradient_checkpointing(self: 'LoRASpecialNetwork'):
        # not supported
        self.is_checkpointing = True
        self._update_checkpointing()

    def disable_gradient_checkpointing(self: 'LoRASpecialNetwork'):
        # not supported
        self.is_checkpointing = False
        self._update_checkpointing()

    def merge_in(self: 'LoRASpecialNetwork', merge_weight=1.0):
        if self.network_type.lower() == 'dora':
            return
        self.is_merged_in = True
        for module in self.get_all_modules():
            module.merge_in(merge_weight)

    def merge_out(self: 'LoRASpecialNetwork', merge_weight=1.0):
        if not self.is_merged_in:
            return
        self.is_merged_in = False
        for module in self.get_all_modules():
            module.merge_out(merge_weight)

    def extract_weight(
            self: 'LoRASpecialNetwork',
            extract_mode: str = "existing", # Use str for Literal for older Python versions
            extract_mode_param: Union[int, float] = None,
    ):
        if extract_mode_param is None:
            raise ValueError("extract_mode_param must be set")
        for module in tqdm(self.get_all_modules(), desc="Extracting weights"):
            module.extract_weight(
                extract_mode=extract_mode,
                extract_mode_param=extract_mode_param
            )

    def setup_lorm(self: 'LoRASpecialNetwork', state_dict: Optional[Dict[str, Any]] = None):
        for module in tqdm(self.get_all_modules(), desc="Extracting LoRM"):
            module.setup_lorm(state_dict=state_dict)

    def calculate_lorem_parameter_reduction(self: 'LoRASpecialNetwork'):
        params_reduced = 0
        for module in self.get_all_modules():
            num_orig_module_params = count_parameters(module.org_module[0])
            num_lorem_params = count_parameters(module.lora_down) + count_parameters(module.lora_up)
            params_reduced += (num_orig_module_params - num_lorem_params)

        return params_reduced
