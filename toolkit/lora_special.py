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
            total_training_steps: Optional[int] = None, # <--- NEW PARAMETER HERE
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
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_sdxl=is_sdxl,
            is_v2=is_v2,
            is_lorm=is_lorm,
            network_config=kwargs.get("network_config", None), # Ensure network_config is passed to mixin
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
        
        # --- NEW ADALORA ATTRIBUTES (start) ---
        self.is_adalora = (network_type.lower() == "adalora")
        self.adalora_config = None
        self.adalora_unet_model = None # Placeholder for PEFT-wrapped UNet
        self.adalora_text_encoder_model = None # Placeholder for PEFT-wrapped Text Encoder(s)
        # --- NEW ADALORA ATTRIBUTES (end) ---

        if self.network_type.lower() == "dora":
            self.module_class = DoRAModule
            module_class = DoRAModule
        elif self.network_type.lower() == "lokr":
            self.module_class = LokrModule
            module_class = LokrModule
        self.network_config: NetworkConfig = kwargs.get("network_config", None)

        self.peft_format = peft_format
        self.is_transformer = is_transformer
        
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

        if self.is_adalora:
            print(f"Creating AdaLoRA network. Initializing AdaLoraConfig.")
            # Extract AdaLoRA specific parameters from network_config (kwargs)
            adalora_rank = lora_dim # `r` parameter in AdaLoraConfig (max rank)
            adalora_alpha = alpha # `lora_alpha`
            adalora_dropout = dropout if dropout is not None else 0.0 # `lora_dropout`
            
            # Default AdaLoRA parameters (can be overridden by kwargs/network_config)
            target_r = kwargs.get('adalora_target_r', adalora_rank // 4 if adalora_rank else 0)
            init_r = kwargs.get('adalora_init_r', adalora_rank)
            tinit = kwargs.get('adalora_tinit', total_training_steps // 4 if total_training_steps else 0)
            tfinal = kwargs.get('adalora_tfinal', 3 * total_training_steps // 4 if total_training_steps else 0)
            deltaT = kwargs.get('adalora_deltaT', 10)
            beta1 = kwargs.get('adalora_beta1', 0.85)
            beta2 = kwargs.get('adalora_beta2', 0.85)
            orth_reg_weight = kwargs.get('adalora_orth_reg_weight', 0.5)

            # Determine target_modules for AdaLoRA
            adalora_target_modules = kwargs.get('target_lin_modules', ["to_k", "to_q", "to_v", "to_out.0"])
            
            # This part attempts to align target_modules with model architecture if specific flags are set.
            # PEFT needs module names as strings that it can find in the model's `named_modules()`.
            # For Qwen-Image, "to_k", "to_q", "to_v", "to_out.0" are common targets within the transformer blocks.
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
                if not adalora_target_modules: # If not explicitly set by kwargs
                    adalora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

            # PEFT requires the `target_modules` to be a list of strings
            # If `only_if_contains` in ai-toolkit's config specifies regex,
            # this would need a more complex mapping to PEFT's expected format.
            # For simplicity, if your `only_if_contains` maps to direct module names, you can use that.
            # Otherwise, it's safer to specify direct PEFT-compatible module names in your YAML for AdaLoRA.
            if self.only_if_contains and base_model is not None:
                # This is a basic attempt to resolve `only_if_contains` patterns to direct module names
                # It might not cover all cases or complex regex. You may need to refine this.
                resolved_modules = []
                # Assuming unet is available here, if not, this will need a more complex way to iterate.
                if hasattr(unet, 'named_modules'):
                    for module_name_pattern in self.only_if_contains:
                        for name, _ in unet.named_modules():
                            try:
                                if re.search(module_name_pattern, name):
                                    # For PEFT, often the last part of the module path is sufficient,
                                    # e.g., for "transformer.single_transformer_blocks.7.attn.to_q", "to_q" is the target.
                                    # This is a heuristic and might need tuning for your specific model architecture.
                                    resolved_modules.append(name.split('.')[-1])
                            except re.error:
                                pass # Handle invalid regex patterns
                if resolved_modules:
                    print(f"Warning: `only_if_contains` patterns were found, attempting to resolve to PEFT target modules. Result: {list(set(resolved_modules))}")
                    # If you are confident your `only_if_contains` resolves to specific PEFT-compatible module names:
                    # Merge with existing target_modules, ensuring no duplicates
                    adalora_target_modules.extend(resolved_modules)
                    adalora_target_modules = list(set(adalora_target_modules)) # Remove duplicates


            self.adalora_config = AdaLoraConfig(
                peft_type="ADALORA",
                task_type="FEATURE_EXTRACTION", # Generic task type for vision models
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
                total_step=total_training_steps, # CRUCIAL: Pass the total steps
            )
            print(f"AdaLoRA configured with max rank {adalora_rank}, target rank {target_r}, "
                  f"tinit {tinit}, tfinal {tfinal}, total_step {total_training_steps}")

            # For AdaLoRA, we don't use the custom create_modules logic, as PEFT handles the module injection.
            self.text_encoder_loras = []
            self.unet_loras = []

        else: # Existing logic for LoRA, DoRA, LoKr
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
                # #############
                # ### need to handle regex for self.only_if_contains!!!
                # # if self.only_if_contains:
                # #     expanded_only_if_contains = []
                # #     for layers in self.only_if_contains:
                # #         if  ".*." in layers:
                # #             transformer_block_names = base_model.get_transformer_block_names()
                # #             # num_blocks = len(root_module.transformer_blocks)
                # #             for block_name in transformer_block_names:
                # #                 blocks = getattr(root_module, block_name)
                # #                 num_blocks = len(blocks)
                # #                 for block_id in range(num_blocks):
                # #                     expanded_only_if_contains.append(layers.replace("*", str(block_id)))
                # #         else:
                # #             expanded_only_if_contains.append(layers)
                #
                # #     self.only_if_contains = expanded_only_if_contains
                # #     # import pdb; pdb.set_trace()
                # #     print(self.only_if_contains)
                # if self.only_if_contains:
                #     expanded_only_if_contains = []
                #     transformer_block_handles = base_model.get_transformer_block_names()
                #     for layer in self.only_if_contains:
                #         for handle in transformer_block_handles:
                #             module_list = getattr(root_module, handle)
                #             layers_list = [name for name, _ in module_list.named_modules()]
                #             # pattern = re.compile(layer)
                #             try:
                #                 pattern = re.compile(layer)
                #             except re.error:
                #                 pattern = re.compile(fnmatch.translate(layer))
                #             # Get matching layers here!!!!!
                #             matched_layers = [f"transformer.{handle}.{item}" for item in layers_list if pattern.match(item)]
                #             expanded_only_if_contains += matched_layers
                #
                #     self.only_if_contains = list(set(expanded_only_if_contains))
                #     print(self.only_if_contains)

                # #############
                # This block was in the original `create_modules` function for non-AdaLoRA and should be preserved
                # if `only_if_contains` logic is intended for non-PEFT LoRA.
                # Assuming this block is outside the `create_modules` function as in the previous output.

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
                            lora_name.replace("..", ".")
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
                                if base_model is not None:
                                    transformer_block_names = base_model.get_transformer_block_names()
                                
                                if transformer_block_names is not None:
                                    if not any([name in lora_name for name in transformer_block_names]):
                                        skip = True
                                else:
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
                for i, text_encoder_module in enumerate(text_encoders): # Renamed `text_encoder` to `text_encoder_module` to avoid conflict
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

                    text_encoder_loras, skipped = create_modules(False, index, text_encoder_module, replace_modules)
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

        if self.full_train_in_out and not self.is_adalora: # Ensure full_train_in_out is not applied for AdaLoRA
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

    # --- NEW apply_to METHOD FOR ADALORA (start) ---
    def apply_to(self, text_encoder_modules, unet_module, train_text_encoder, train_unet):
        if self.is_adalora:
            if train_unet:
                print("Applying AdaLoRA to UNet (transformer)")
                # `get_peft_model` modifies `unet_module` in-place
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
            # Existing apply_to logic for LoRA/DoRA/LoKr
            super().apply_to(text_encoder_modules, unet_module, train_text_encoder, train_unet)
    # --- NEW apply_to METHOD FOR ADALORA (end) ---

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
            # call Lora prepare_optimizer_params
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

    # --- NEW update_and_allocate METHOD FOR ADALORA (start) ---
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
    # --- NEW update_and_allocate METHOD FOR ADALORA (end) ---

    # --- NEW get_all_modules for AdaLoRA (start) ---
    def get_all_modules(self: 'LoRASpecialNetwork') -> List[torch.nn.Module]: # Change return type to torch.nn.Module
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
    # --- NEW get_all_modules for AdaLoRA (end) ---

    # --- NEW get_state_dict for AdaLoRA (start) ---
    def get_state_dict(self: 'LoRASpecialNetwork', extra_state_dict=None, dtype=torch.float16):
        if self.is_adalora:
            save_dict = OrderedDict()
            # If UNet was trained with AdaLoRA
            if self.adalora_unet_model:
                unet_state_dict = get_peft_model_state_dict(self.adalora_unet_model, adapter_name="adalora_unet")
                for k, v in unet_state_dict.items():
                    # Prefix with 'transformer.' to indicate it belongs to the UNet (transformer)
                    save_dict[f"transformer.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            # If Text Encoder(s) were trained with AdaLoRA
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for i, te_model in enumerate(self.adalora_text_encoder_model):
                        te_state_dict = get_peft_model_state_dict(te_model, adapter_name=f"adalora_te_{i}")
                        for k, v in te_state_dict.items():
                            # Prefix with 'text_encoder.0.' for the first, 'text_encoder.1.' for the second
                            save_dict[f"text_encoder.{i}.{k}"] = v.detach().clone().to("cpu").to(dtype)
                else:
                    te_state_dict = get_peft_model_state_dict(self.adalora_text_encoder_model, adapter_name="adalora_te")
                    for k, v in te_state_dict.items():
                        # Prefix with 'text_encoder.' if only one TE or general case
                        save_dict[f"text_encoder.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            # Add any extra state dict items
            if extra_state_dict is not None:
                for key in list(extra_state_dict.keys()):
                    v = extra_state_dict[key]
                    save_dict[key] = v.detach().clone().to("cpu").to(dtype)
            
            return save_dict
        else:
            # Original content of this function for non-AdaLoRA
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
                save_dict = self.base_model_ref().convert_lora_weights_before_save(save_dict)
            return save_dict
    # --- NEW get_state_dict for AdaLoRA (end) ---

    # --- NEW load_weights for AdaLoRA (start) ---
    def load_weights(self: 'LoRASpecialNetwork', file, force_weight_mapping=False):
        if self.is_adalora:
            # For AdaLoRA, we need to load into the PEFT-wrapped models
            if isinstance(file, str):
                if os.path.splitext(file)[1] == ".safetensors":
                    from safetensors.torch import load_file
                    weights_sd = load_file(file, device="cpu") # Load to CPU first
                else:
                    weights_sd = torch.load(file, map_location="cpu")
            else:
                weights_sd = file # Assume it's already a state dict

            if self.base_model_ref is not None:
                weights_sd = self.base_model_ref().convert_lora_weights_before_load(weights_sd)

            # Distribute weights to the correct PEFT models
            unet_state_dict = OrderedDict()
            text_encoder_state_dict = OrderedDict()
            extra_dict = OrderedDict()

            for key, value in weights_sd.items():
                if key.startswith("transformer."):
                    unet_state_dict[key.replace("transformer.", "")] = value
                elif key.startswith("text_encoder."):
                    # Handle multiple text encoders (e.g., 0.model.layers...)
                    te_key_suffix = key.replace('text_encoder.', '')
                    if '.' in te_key_suffix and te_key_suffix.split('.')[0].isdigit():
                        te_idx = int(te_key_suffix.split('.')[0])
                        # Store with index prefix in `text_encoder_state_dict` for easy distribution to list of TEs
                        text_encoder_state_dict[key.replace(f"text_encoder.{te_idx}.", f"{te_idx}.")] = value
                    else:
                        # General case for single TE or non-indexed keys
                        text_encoder_state_dict[te_key_suffix] = value
                else:
                    extra_dict[key] = value

            if self.adalora_unet_model:
                set_peft_model_state_dict(self.adalora_unet_model, unet_state_dict, adapter_name="adalora_unet")
            
            if self.adalora_text_encoder_model:
                if isinstance(self.adalora_text_encoder_model, list):
                    for i, te_model in enumerate(self.adalora_text_encoder_model):
                        # Construct state_dict for this specific text encoder
                        single_te_sd = OrderedDict()
                        for k, v in text_encoder_state_dict.items():
                            if k.startswith(f"{i}."): # If key has the index prefix
                                single_te_sd[k.replace(f"{i}.", "")] = v
                            # Else, if key is general (no index prefix) and `text_encoder_state_dict` has such,
                            # it implies a single TE or a key meant for all TEs. Handle with caution.
                            # For robustness, we assume if it has an index prefix, it's specific.
                            # Otherwise, it might be a general key intended for all.
                            elif not any(key.startswith(f"{j}.") for j in range(len(self.adalora_text_encoder_model))):
                                single_te_sd[k] = v # Add general keys to each TE if no specific index was found
                        set_peft_model_state_dict(te_model, single_te_sd, adapter_name=f"adalora_te_{i}")
                else:
                    set_peft_model_state_dict(self.adalora_text_encoder_model, text_encoder_state_dict, adapter_name="adalora_te")
            
            if len(extra_dict.keys()) == 0:
                extra_dict = None
            return extra_dict
        else:
            # Original content of this function for non-AdaLoRA
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
                weights_sd = self.base_model_ref().convert_lora_weights_before_load(weights_sd)

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
    # --- NEW load_weights for AdaLoRA (end) ---

    @torch.no_grad()
    def _update_torch_multiplier(self: 'LoRASpecialNetwork'): # Specify LoRASpecialNetwork type
        if self.is_adalora:
            # For AdaLoRA, the multiplier is handled internally by PEFT's AdaLoraModel during `update_and_allocate`
            # and is not applied via a global `torch_multiplier`.
            # We can still set a placeholder if needed, or ensure it's not used.
            # Using device/dtype from unet if available, otherwise default
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
            # Original content of this function for non-AdaLoRA
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

    # Note: get_all_modules has been moved and modified above for AdaLoRA.
    # The original implementation would be within the `else` block of the new `get_all_modules`.

    def _update_checkpointing(self: 'Network'):
        for module in self.get_all_modules():
            if self.is_checkpointing:
                if hasattr(module, 'enable_gradient_checkpointing'): # Check if method exists on module
                    module.enable_gradient_checkpointing()
                elif hasattr(module, 'gradient_checkpointing_enable'): # For PEFT models
                    module.gradient_checkpointing_enable()
            else:
                if hasattr(module, 'disable_gradient_checkpointing'):
                    module.disable_gradient_checkpointing()
                elif hasattr(module, 'gradient_checkpointing_disable'): # For PEFT models
                    module.gradient_checkpointing_disable()

    def enable_gradient_checkpointing(self: 'LoRASpecialNetwork'): # Specify LoRASpecialNetwork type
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
            # Original content of this function for non-AdaLoRA
            # not supported
            super().enable_gradient_checkpointing()
            self._update_checkpointing() # Call _update_checkpointing for custom LoRAModules
    
    def disable_gradient_checkpointing(self: 'LoRASpecialNetwork'): # Specify LoRASpecialNetwork type
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
            # Original content of this function for non-AdaLoRA
            # not supported
            super().disable_gradient_checkpointing()
            self._update_checkpointing() # Call _update_checkpointing for custom LoRAModules

    def merge_in(self: 'LoRASpecialNetwork', merge_weight=1.0): # Specify LoRASpecialNetwork type
        if self.is_adalora:
            print("Warning: merge_in is not directly supported for AdaLoRA networks via this method. PEFT handles merging internally.")
        else:
            # Original content of this function for non-AdaLoRA
            if self.network_type.lower() == 'dora':
                return
            self.is_merged_in = True
            for module in self.get_all_modules():
                if hasattr(module, 'merge_in'):
                    module.merge_in(merge_weight)
    
    def merge_out(self: 'LoRASpecialNetwork', merge_weight=1.0): # Specify LoRASpecialNetwork type
        if self.is_adalora:
            print("Warning: merge_out is not directly supported for AdaLoRA networks via this method. PEFT handles merging internally.")
        else:
            # Original content of this function for non-AdaLoRA
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
            if hasattr(module, 'extract_weight'): # Ensure module has this method
                module.extract_weight(
                    extract_mode=extract_mode,
                    extract_mode_param=extract_mode_param
                )

    def setup_lorm(self: 'Network', state_dict: Optional[Dict[str, Any]] = None):
        for module in tqdm(self.get_all_modules(), desc="Extracting LoRM"):
            if hasattr(module, 'setup_lorm'): # Ensure module has this method
                module.setup_lorm(state_dict=state_dict)

    def calculate_lorem_parameter_reduction(self):
        params_reduced = 0
        for module in self.get_all_modules():
            if hasattr(module, 'org_module') and hasattr(module, 'lora_down') and hasattr(module, 'lora_up'):
                num_orig_module_params = count_parameters(module.org_module[0])
                num_lorem_params = count_parameters(module.lora_down) + count_parameters(module.lora_up)
                params_reduced += (num_orig_module_params - num_lorem_params)

        return params_reduced