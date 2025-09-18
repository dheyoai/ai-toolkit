import json
import os
from collections import OrderedDict
from typing import Optional, Union, List, Type, TYPE_CHECKING, Dict, Any, Literal

import torch
from optimum.quanto import QTensor
from torch import nn
import weakref

from tqdm import tqdm

from toolkit.config_modules import NetworkConfig
from toolkit.lorm import extract_conv, extract_linear, count_parameters
from toolkit.metadata import add_model_hash_to_meta
from toolkit.paths import KEYMAPS_ROOT
from toolkit.saving import get_lora_keymap_from_model_keymap
from optimum.quanto import QBytesTensor
import peft
from peft import get_peft_model_state_dict, set_peft_model_state_dict  # Added imports for AdaLoRA

if TYPE_CHECKING:
    from toolkit.lycoris_special import LycorisSpecialNetwork, LoConSpecialModule
    from toolkit.lora_special import LoRASpecialNetwork, LoRAModule
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.models.DoRA import DoRAModule
    from peft import PeftModel

# Define Network and Module type hints here
Network = Union['LycorisSpecialNetwork', 'LoRASpecialNetwork']
Module = Union['LoConSpecialModule', 'LoRAModule', 'DoRAModule']

ExtractMode = Literal[
    'existing',
    'fixed',
    'threshold',
    'ratio',
    'quantile',
    'percentage'
]

def broadcast_and_multiply(tensor, multiplier):
    num_extra_dims = tensor.dim() - multiplier.dim()

    for _ in range(num_extra_dims):
        multiplier = multiplier.unsqueeze(-1)

    try:
        result = tensor * multiplier
    except RuntimeError as e:
        print(e)
        print(tensor.size())
        print(multiplier.size())
        raise e

    return result

def add_bias(tensor, bias):
    if bias is None:
        return tensor
    bias = bias.unsqueeze(0)
    bias = torch.cat([bias] * tensor.size(0), dim=0)
    num_extra_dims = tensor.dim() - bias.dim()

    for _ in range(num_extra_dims):
        bias = bias.unsqueeze(-1)

    if bias.size(1) != tensor.size(1):
        if len(bias.size()) == 3:
            bias = bias.permute(0, 2, 1)
        elif len(bias.size()) == 4:
            bias = bias.permute(0, 3, 1, 2)

    try:
        result = tensor + bias
    except RuntimeError as e:
        print(e)
        print(tensor.size())
        print(bias.size())
        raise e

    return result

class ExtractableModuleMixin:
    def extract_weight(
            self: Module,
            extract_mode: ExtractMode = "existing",
            extract_mode_param: Union[int, float] = None,
    ):
        device = self.lora_down.weight.device
        weight_to_extract = self.org_module[0].weight
        if extract_mode == "existing":
            extract_mode = 'fixed'
            extract_mode_param = self.lora_dim
            
        if isinstance(weight_to_extract, QBytesTensor):
            weight_to_extract = weight_to_extract.dequantize()
        
        weight_to_extract = weight_to_extract.clone().detach().float()

        if self.org_module[0].__class__.__name__ in CONV_MODULES:
            down_weight, up_weight, new_dim, diff = extract_conv(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device
            )

        elif self.org_module[0].__class__.__name__ in LINEAR_MODULES:
            down_weight, up_weight, new_dim, diff = extract_linear(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device,
            )
        else:
            raise ValueError(f"Unknown module type: {self.org_module[0].__class__.__name__}")

        self.lora_dim = new_dim

        self.lora_down.weight.data = down_weight.to(self.lora_down.weight.dtype).clone().detach()
        self.lora_up.weight.data = up_weight.to(self.lora_up.weight.dtype).clone().detach()

        if self.org_module[0].bias is not None and self.lora_up.bias is not None:
            self.lora_up.bias.data = self.org_module[0].bias.data.clone().detach()

        self.alpha = (self.alpha * 0) + down_weight.shape[0]
        self.scale = self.alpha / self.lora_dim

        if hasattr(self, 'scalar'):
            self.scalar.data = torch.tensor(1.0).to(self.scalar.device, self.scalar.dtype)

class ToolkitModuleMixin:
    def __init__(
            self: Module,
            *args,
            network: Network,
            **kwargs
    ):
        self.network_ref: weakref.ref = weakref.ref(network)
        self.is_checkpointing = False
        self._multiplier: Union[float, list, torch.Tensor] = None

    def _call_forward(self: Module, x):
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return 0.0

        if hasattr(self, 'lora_mid') and self.lora_mid is not None:
            lx = self.lora_mid(self.lora_down(x))
        else:
            try:
                lx = self.lora_down(x)
            except RuntimeError as e:
                print(f"Error in {self.__class__.__name__} lora_down")
                raise e

        if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
            lx = self.dropout(lx)
        elif self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.rank_dropout > 0 and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        return lx * scale

    def lorm_forward(self: Network, x, *args, **kwargs):
        network: Network = self.network_ref()
        if not network.is_active:
            return self.org_forward(x, *args, **kwargs)
        
        orig_dtype = x.dtype
        
        if x.dtype != self.lora_down.weight.dtype:
            x = x.to(self.lora_down.weight.dtype)

        if network.lorm_train_mode == 'local':
            inputs = x.detach()
            with torch.no_grad():
                target_pred = self.org_forward(inputs, *args, **kwargs).detach()
            with torch.set_grad_enabled(True):
                lorm_pred = self.lora_up(self.lora_down(inputs.requires_grad_(True)))

                local_loss = torch.nn.functional.mse_loss(target_pred.float(), lorm_pred.float())
                local_loss.backward()

            network.module_losses.append(local_loss.detach())
            return target_pred

        else:
            x = self.lora_up(self.lora_down(x))
            if x.dtype != orig_dtype:
                x = x.to(orig_dtype)

    def forward(self: Module, x, *args, **kwargs):
        skip = False
        network: Network = self.network_ref()
        if network.is_lorm:
            return self.lorm_forward(x, *args, **kwargs)

        # The conditional forward for AdaLoRA is handled in LoRASpecialNetwork.apply_to
        # and it replaces the UNet/TextEncoder references in the base_model.
        # Individual LoRAModules don't need to check network.network_type here.

        if not network.is_active:
            skip = True

        if network.is_merged_in:
            skip = True

        if network._multiplier == 0:
            skip = True

        if skip:
            return self.org_forward(x, *args, **kwargs)
        
        if self.__class__.__name__ == "LokrModule":
            return self._call_forward(x)

        org_forwarded = self.org_forward(x, *args, **kwargs)

        if isinstance(x, QTensor):
            x = x.dequantize()
        lora_input = x.to(self.lora_down.weight.dtype)
        lora_output = self._call_forward(lora_input)
        multiplier = self.network_ref().torch_multiplier

        lora_output_batch_size = lora_output.size(0)
        multiplier_batch_size = multiplier.size(0)
        if lora_output_batch_size != multiplier_batch_size:
            num_interleaves = lora_output_batch_size // multiplier_batch_size
            multiplier = multiplier.repeat_interleave(num_interleaves)

        scaled_lora_output = broadcast_and_multiply(lora_output, multiplier)
        scaled_lora_output = scaled_lora_output.to(org_forwarded.dtype)

        if self.__class__.__name__ == "DoRAModule":
            if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
                lx = self.dropout(x)
            elif self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(x, p=self.dropout)
            else:
                lx = x
            lora_weight = self.lora_up.weight @ self.lora_down.weight
            scale = multiplier.mean()
            scaled_lora_weight = lora_weight * scale
            scaled_lora_output = scaled_lora_output + self.apply_dora(lx, scaled_lora_weight).to(org_forwarded.dtype)

        try:
            x = org_forwarded + scaled_lora_output
        except RuntimeError as e:
            print(e)
            print(org_forwarded.size())
            print(scaled_lora_output.size())
            raise e
        return x

    def enable_gradient_checkpointing(self: Module):
        self.is_checkpointing = True

    def disable_gradient_checkpointing(self: Module):
        self.is_checkpointing = False

    @torch.no_grad()
    def merge_out(self: Module, merge_out_weight=1.0):
        merge_out_weight = abs(merge_out_weight)
        self.merge_in(merge_weight=-merge_out_weight)

    @torch.no_grad()
    def merge_in(self: Module, merge_weight=1.0):
        if not self.can_merge_in:
            return
        up_weight = self.lora_up.weight.clone().float()
        down_weight = self.lora_down.weight.clone().float()

        org_sd = self.org_module[0].state_dict()
        if 'weight._data' in org_sd:
            return

        weight_key = "weight"
        if 'weight._data' in org_sd:
            weight_key = "weight._data"

        orig_dtype = org_sd[weight_key].dtype
        weight = org_sd[weight_key].float()

        multiplier = merge_weight
        scale = self.scale
        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        if len(weight.size()) == 2:
            weight = weight + multiplier * (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):
            weight = (
                    weight
                    + multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
            )
        else:
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = weight + multiplier * conved * scale

        org_sd[weight_key] = weight.to(orig_dtype)
        self.org_module[0].load_state_dict(org_sd)

    def setup_lorm(self: Module, state_dict: Optional[Dict[str, Any]] = None):
        network: Network = self.network_ref()
        lorm_config = network.network_config.lorm_config.get_config_for_module(self.lora_name)

        extract_mode = lorm_config.extract_mode
        extract_mode_param = lorm_config.extract_mode_param
        parameter_threshold = lorm_config.parameter_threshold
        self.extract_weight(
            extract_mode=extract_mode,
            extract_mode_param=extract_mode_param
        )

class ToolkitNetworkMixin:
    def __init__(
            self: Network,
            *args,
            train_text_encoder: Optional[bool] = True,
            train_unet: Optional[bool] = True,
            is_sdxl=False,
            is_v2=False,
            is_ssd=False,
            is_vega=False,
            network_config: Optional[NetworkConfig] = None,
            is_lorm=False,
            base_model_ref: Optional[weakref.ref['StableDiffusion']] = None,
            network_type: str = "lora",
            **kwargs
    ):
        self.train_text_encoder = train_text_encoder
        self.train_unet = train_unet
        self.is_checkpointing = False
        self._multiplier: float = 1.0
        self.is_active: bool = False
        self.is_sdxl = is_sdxl
        self.is_ssd = is_ssd
        self.is_vega = is_vega
        self.is_v2 = is_v2
        self.is_v1 = not is_v2 and not is_sdxl and not is_ssd and not is_vega
        self.is_merged_in = False
        self.is_lorm = is_lorm
        self.network_config: NetworkConfig = network_config
        self.module_losses: List[torch.Tensor] = []
        self.lorm_train_mode: Literal['local', None] = None
        self.can_merge_in = not is_lorm
        self.base_model_ref = base_model_ref
        self.network_type = network_type

    def get_keymap(self: Network, force_weight_mapping=False):
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
        if force_weight_mapping:
            use_weight_mapping = True

        keymap_name = f"stable_diffusion_locon_{keymap_tail}.json"
        if use_weight_mapping:
            keymap_name = f"stable_diffusion_{keymap_tail}.json"

        keymap_path = os.path.join(KEYMAPS_ROOT, keymap_name)

        keymap = None
        if os.path.exists(keymap_path):
            with open(keymap_path, 'r') as f:
                keymap = json.load(f)['ldm_diffusers_keymap']

        if use_weight_mapping and keymap is not None:
            keymap = get_lora_keymap_from_model_keymap(keymap)

        if self.network_type.lower() == 'dora':
            if keymap is not None:
                new_keymap = {}
                for ldm_key, diffusers_key in keymap.items():
                    ldm_key = ldm_key.replace('.alpha', '.magnitude')
                    new_keymap[ldm_key] = diffusers_key

                keymap = new_keymap

        return keymap
    
    def get_state_dict(self: Network, extra_state_dict=None, dtype=torch.float16):
        if self.network_type.lower() == "adalora":
            adapters_state_dict = OrderedDict()
            base_model_instance = self.base_model_ref()
            if base_model_instance is None:
                raise ValueError("Base model reference is missing for AdaLoRA saving.")

            if hasattr(base_model_instance, 'peft_adapted_text_encoders') and base_model_instance.peft_adapted_text_encoders:
                for i, te_model in enumerate(base_model_instance.peft_adapted_text_encoders):
                    if isinstance(te_model, peft.PeftModel) and \
                       hasattr(te_model, 'peft_config') and f"default_adalora_adapter_te_{i}" in te_model.peft_config:
                        te_state_dict = get_peft_model_state_dict(te_model, adapter_name=f"default_adalora_adapter_te_{i}")
                        for k, v in te_state_dict.items():
                            adapters_state_dict[f"text_encoder.{i}.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            if hasattr(base_model_instance, 'peft_adapted_unet') and base_model_instance.peft_adapted_unet:
                if isinstance(base_model_instance.peft_adapted_unet, peft.PeftModel) and \
                   hasattr(base_model_instance.peft_adapted_unet, 'peft_config') and "default_adalora_adapter_unet" in base_model_instance.peft_adapted_unet.peft_config:
                    unet_state_dict = get_peft_model_state_dict(base_model_instance.peft_adapted_unet, adapter_name="default_adalora_adapter_unet")
                    for k, v in unet_state_dict.items():
                        adapters_state_dict[f"transformer.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            # Handle full_train_in_out modules
            if self.full_train_in_out:
                if hasattr(base_model_instance, 'transformer_pos_embed'):
                    adapters_state_dict.update({"transformer_pos_embed." + k: v for k, v in base_model_instance.transformer_pos_embed.state_dict().items()})
                if hasattr(base_model_instance, 'transformer_proj_out'):
                    adapters_state_dict.update({"transformer_proj_out." + k: v for k, v in base_model_instance.transformer_proj_out.state_dict().items()})
                if hasattr(base_model_instance, 'unet_conv_in'):
                    adapters_state_dict.update({"unet_conv_in." + k: v for k, v in base_model_instance.unet_conv_in.state_dict().items()})
                if hasattr(base_model_instance, 'unet_conv_out'):
                    adapters_state_dict.update({"unet_conv_out." + k: v for k, v in base_model_instance.unet_conv_out.state_dict().items()})

            if extra_state_dict:
                adapters_state_dict.update(extra_state_dict)
            
            return adapters_state_dict

        save_dict = self.state_dict()
        
        save_dict_formatted = OrderedDict()
        for key in list(save_dict.keys()):
            v = save_dict[key]
            v = v.detach().clone().to("cpu").to(dtype)
            
            keymap_for_lora = self.get_keymap()
            save_key = key
            if keymap_for_lora is not None and key in keymap_for_lora:
                save_key = keymap_for_lora[key]
            
            if self.peft_format:
                new_key = save_key
                if new_key.endswith('.alpha'):
                    continue
                new_key = new_key.replace('lora_down', 'lora_A')
                new_key = new_key.replace('lora_up', 'lora_B')
                new_key = new_key.replace('$$', '.')
                save_key = new_key
            
            if self.network_type.lower() == "lokr":
                new_key = save_key
                new_key = new_key.replace('lora_transformer_', 'lycoris_')
                save_key = new_key
            
            save_dict_formatted[save_key] = v
            del save_dict[key]

        if extra_state_dict is not None:
            for key in list(extra_state_dict.keys()):
                v = extra_state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_dict_formatted[key] = v

        if self.base_model_ref is not None:
            save_dict_formatted = self.base_model_ref().convert_lora_weights_before_save(save_dict_formatted)
        return save_dict_formatted

    def save_weights(
            self: Network,
            file, dtype=torch.float16,
            metadata=None,
            extra_state_dict: Optional[OrderedDict] = None
    ):
        if self.network_type.lower() == "adalora":
            adapters_state_dict = self.get_state_dict(extra_state_dict=extra_state_dict, dtype=dtype)
            
            if metadata is not None and len(metadata) == 0:
                metadata = None

            if metadata is None:
                metadata = OrderedDict()
            metadata = add_model_hash_to_meta(adapters_state_dict, metadata)
            
            from safetensors.torch import save_file
            save_file(adapters_state_dict, file, metadata)
            return

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

    def load_weights(self: Network, file, force_weight_mapping=False):
        if self.network_type.lower() == "adalora":
            from safetensors.torch import load_file
            adapter_state_dict = load_file(file, device="cpu")
            base_model_instance = self.base_model_ref()
            if base_model_instance is None:
                raise ValueError("Base model reference is missing for AdaLoRA loading.")

            unet_peft_state_dict = OrderedDict()
            text_encoder_peft_state_dict_map = {}

            for key, value in adapter_state_dict.items():
                if key.startswith("transformer."):
                    unet_peft_state_dict[key.replace("transformer.", "")] = value
                elif key.startswith("text_encoder."):
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[1].isdigit():
                        te_idx = int(parts[1])
                        peft_te_key = ".".join(parts[2:])
                        if te_idx not in text_encoder_peft_state_dict_map:
                            text_encoder_peft_state_dict_map[te_idx] = OrderedDict()
                        text_encoder_peft_state_dict_map[te_idx][peft_te_key] = value
                    else:
                        if 0 not in text_encoder_peft_state_dict_map:
                            text_encoder_peft_state_dict_map[0] = OrderedDict()
                        text_encoder_peft_state_dict_map[0][key.replace("text_encoder.", "")] = value
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
                        set_peft_model_state_dict(te_model, text_encoder_peft_state_dict_map[idx], adapter_name=f"default_adalora_adapter_te_{idx}")
            
            print(f"Loaded AdaLoRA weights from {file}")
            return {}

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

    def _update_torch_multiplier(self: Network):
        if self.network_type.lower() == "adalora":
            device = torch.device("cpu")
            dtype = torch.float32
            if hasattr(self, 'peft_adapted_unet') and self.peft_adapted_unet is not None and isinstance(self.peft_adapted_unet, peft.PeftModel):
                device = self.peft_adapted_unet.device
                dtype = self.peft_adapted_unet.dtype
            elif hasattr(self, 'peft_adapted_text_encoders') and self.peft_adapted_text_encoders is not None:
                if isinstance(self.peft_adapted_text_encoders, torch.nn.ModuleList) and len(self.peft_adapted_text_encoders) > 0 and isinstance(self.peft_adapted_text_encoders[0], peft.PeftModel):
                    device = self.peft_adapted_text_encoders[0].device
                    dtype = self.peft_adapted_text_encoders[0].dtype
                elif isinstance(self.peft_adapted_text_encoders, peft.PeftModel):
                    device = self.peft_adapted_text_encoders.device
                    dtype = self.peft_adapted_text_encoders.dtype

            self.torch_multiplier = torch.tensor(1.0).to(device, dtype=dtype)
            return

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
            return 1.0
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float], List[List[float]]]):
        if self.network_type.lower() == "adalora":
            self._multiplier = value
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
            return []
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        return loras

    def _update_checkpointing(self: Network):
        if self.network_type.lower() == "adalora":
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
            return
        for module in tqdm(self.get_all_modules(), desc="Extracting LoRM"):
            if hasattr(module, 'setup_lorm'):
                module.setup_lorm(state_dict=state_dict)

    def calculate_lorem_parameter_reduction(self):
        if self.network_type.lower() == "adalora":
            return 0
        params_reduced = 0
        for module in self.get_all_modules():
            if hasattr(module, 'org_module') and hasattr(module, 'lora_down') and hasattr(module, 'lora_up'):
                num_orig_module_params = count_parameters(module.org_module[0])
                num_lorem_params = count_parameters(module.lora_down) + count_parameters(module.lora_up)
                params_reduced += (num_orig_module_params - num_lorem_params)

        return params_reduced