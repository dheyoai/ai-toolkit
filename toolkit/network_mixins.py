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
from peft import get_peft_model_state_dict, set_peft_model_state_dict

if TYPE_CHECKING:
    from toolkit.lycoris_special import LycorisSpecialNetwork, LoConSpecialModule
    from toolkit.lora_special import LoRASpecialNetwork, LoRAModule
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.models.DoRA import DoRAModule
    from peft import PeftModel

# Define Network and Module type hints here
Network = Union['LycorisSpecialNetwork', 'LoRASpecialNetwork']
Module = Union['LoConSpecialModule', 'LoRAModule', 'DoRAModule']

LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
    'QLinear'
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

ExtractMode = Literal[
    'existing',
    'fixed',
    'threshold',
    'ratio',
    'quantile',
    'percentage'
]

def broadcast_and_multiply(tensor, multiplier):
    # Determine the number of dimensions required
    num_extra_dims = tensor.dim() - multiplier.dim()

    # Unsqueezing the tensor to match the dimensionality
    for _ in range(num_extra_dims):
        multiplier = multiplier.unsqueeze(-1)

    try:
        # Multiplying the broadcasted tensor with the output tensor
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
    # add batch dim
    bias = bias.unsqueeze(0)
    bias = torch.cat([bias] * tensor.size(0), dim=0)
    # Determine the number of dimensions required
    num_extra_dims = tensor.dim() - bias.dim()

    # Unsqueezing the tensor to match the dimensionality
    for _ in range(num_extra_dims):
        bias = bias.unsqueeze(-1)

    # we may need to swap -1 for -2
    if bias.size(1) != tensor.size(1):
        if len(bias.size()) == 3:
            bias = bias.permute(0, 2, 1)
        elif len(bias.size()) == 4:
            bias = bias.permute(0, 3, 1, 2)

    # Multiplying the broadcasted tensor with the output tensor
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
            # do conv extraction
            down_weight, up_weight, new_dim, diff = extract_conv(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device
            )

        elif self.org_module[0].__class__.__name__ in LINEAR_MODULES:
            # do linear extraction
            down_weight, up_weight, new_dim, diff = extract_linear(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device,
            )
        else:
            raise ValueError(f"Unknown module type: {self.org_module[0].__class__.__name__}")

        self.lora_dim = new_dim

        # inject weights into the param
        self.lora_down.weight.data = down_weight.to(self.lora_down.weight.dtype).clone().detach()
        self.lora_up.weight.data = up_weight.to(self.lora_up.weight.dtype).clone().detach()

        # copy bias if we have one and are using them
        if self.org_module[0].bias is not None and self.lora_up.bias is not None:
            self.lora_up.bias.data = self.org_module[0].bias.data.clone().detach()

        # set up alphas
        self.alpha = (self.alpha * 0) + down_weight.shape[0]
        self.scale = self.alpha / self.lora_dim

        # assign them

        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            # scaler is a parameter update the value with 1.0
            self.scalar.data = torch.tensor(1.0).to(self.scalar.device, self.scalar.dtype)


class ToolkitModuleMixin:
    def __init__(
            self,
            *args,
            network: Network,
            **kwargs
    ):
        self.network_ref: weakref.ref = weakref.ref(network)
        self.is_checkpointing = False
        self._multiplier: Union[float, list, torch.Tensor] = None

    def _call_forward(self: Module, x):
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return 0.0  # added to original forward

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
        # normal dropout
        elif self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.rank_dropout > 0 and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            # scaler is a parameter update the value with 1.0
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
            # we are going to predict input with both and do a loss on them
            inputs = x.detach()
            with torch.no_grad():
                # get the local prediction
                target_pred = self.org_forward(inputs, *args, **kwargs).detach()
            with torch.set_grad_enabled(True):
                # make a prediction with the lorm
                lorm_pred = self.lora_up(self.lora_down(inputs.requires_grad_(True)))

                local_loss = torch.nn.functional.mse_loss(target_pred.float(), lorm_pred.float())
                # backpropr
                local_loss.backward()

            network.module_losses.append(local_loss.detach())
            # return the original as we dont want our trainer to affect ones down the line
            return target_pred

        else:
            x = self.lora_up(self.lora_down(x))
            if x.dtype != orig_dtype:
                x = x.to(orig_dtype)

    def forward(self: Module, x, *args, **kwargs):
        skip = False
        network: Network = self.network_ref()
        if network.is_lorm:
            # we are doing lorm
            return self.lorm_forward(x, *args, **kwargs)

        # The conditional forward for AdaLoRA is handled in LoRASpecialNetwork.apply_to
        # and it replaces the UNet/TextEncoder references in the base_model.
        # Individual LoRAModules don't need to check network.network_type here.

        # skip if not active
        if not network.is_active:
            skip = True

        # skip if is merged in
        if network.is_merged_in:
            skip = True

        # skip if multiplier is 0
        if network._multiplier == 0:
            skip = True

        if skip:
            # network is not active, avoid doing anything
            return self.org_forward(x, *args, **kwargs)

        # if self.__class__.__name__ == "DoRAModule":
        #     # return dora forward
        #     return self.dora_forward(x, *args, **kwargs)
        
        if self.__class__.__name__ == "LokrModule":
            return self._call_forward(x)

        org_forwarded = self.org_forward(x, *args, **kwargs)

        if isinstance(x, QTensor):
            x = x.dequantize()
        # always cast to float32
        lora_input = x.to(self.lora_down.weight.dtype)
        lora_output = self._call_forward(lora_input)
        multiplier = self.network_ref().torch_multiplier

        lora_output_batch_size = lora_output.size(0)
        multiplier_batch_size = multiplier.size(0)
        if lora_output_batch_size != multiplier_batch_size:
            num_interleaves = lora_output_batch_size // multiplier_batch_size
            # todo check if this is correct, do we just concat when doing cfg?
            multiplier = multiplier.repeat_interleave(num_interleaves)

        scaled_lora_output = broadcast_and_multiply(lora_output, multiplier)
        scaled_lora_output = scaled_lora_output.to(org_forwarded.dtype)

        if self.__class__.__name__ == "DoRAModule":
            # ref https://github.com/huggingface/peft/blob/1e6d1d73a0850223b0916052fd8d2382a90eae5a/src/peft/tuners/lora/layer.py#L417
            # x = dropout(x)
            # todo this wont match the dropout applied to the lora
            if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
                lx = self.dropout(x)
            # normal dropout
            elif self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)
            else:
                lx = x
            lora_weight = self.lora_up.weight @ self.lora_down.weight
            # scale it here
            # todo handle our batch split scalers for slider training. For now take the mean of them
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
        # make sure it is positive
        merge_out_weight = abs(merge_out_weight)
        # merging out is just merging in the negative of the weight
        self.merge_in(merge_weight=-merge_out_weight)

    @torch.no_grad()
    def merge_in(self: Module, merge_weight=1.0):
        if not self.can_merge_in:
            return
        # get up/down weight
        up_weight = self.lora_up.weight.clone().float()
        down_weight = self.lora_down.weight.clone().float()

        # extract weight from org_module
        org_sd = self.org_module[0].state_dict()
        # todo find a way to merge in weights when doing quantized model
        if 'weight._data' in org_sd:
            # quantized weight
            return

        weight_key = "weight"
        if 'weight._data' in org_sd:
            # quantized weight
            weight_key = "weight._data"

        orig_dtype = org_sd[weight_key].dtype
        weight = org_sd[weight_key].float()

        multiplier = merge_weight
        scale = self.scale
        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            # scaler is a parameter update the value with 1.0
            scale = scale * self.scalar

        # merge weight
        if len(weight.size()) == 2:
            # linear
            weight = weight + multiplier * (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                    weight
                    + multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # print(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + multiplier * conved * scale

        # set weight to org_module
        org_sd[weight_key] = weight.to(orig_dtype)
        self.org_module[0].load_state_dict(org_sd)

    def setup_lorm(self: Module, state_dict: Optional[Dict[str, Any]] = None):
        # LoRM (Low Rank Middle) is a method reduce the number of parameters in a module while keeping the inputs and
        # outputs the same. It is basically a LoRA but with the original module removed

        # if a state dict is passed, use those weights instead of extracting
        # todo load from state dict
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
        # For AdaLoRA, store PeftModel instances directly
        self.peft_adapted_unet: Optional['PeftModel'] = None
        self.peft_adapted_text_encoders: List['PeftModel'] = []
        self.peft_adapter_name: str = "default_adalora_adapter"


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
                        unwrapped_te_model = te_model.base_model # unwrap if it was wrapped by Accelerator
                        te_state_dict = get_peft_model_state_dict(unwrapped_te_model, adapter_name=f"default_adalora_adapter_te_{i}")
                        for k, v in te_state_dict.items():
                            adapters_state_dict[f"text_encoder.{i}.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            if hasattr(base_model_instance, 'peft_adapted_unet') and base_model_instance.peft_adapted_unet:
                if isinstance(base_model_instance.peft_adapted_unet, peft.PeftModel) and \
                   hasattr(base_model_instance.peft_adapted_unet, 'peft_config') and "default_adalora_adapter_unet" in base_model_instance.peft_adapted_unet.peft_config:
                    unwrapped_unet_model = base_model_instance.peft_adapted_unet.base_model # unwrap if it was wrapped by Accelerator
                    unet_state_dict = get_peft_model_state_dict(unwrapped_unet_model, adapter_name="default_adalora_adapter_unet")
                    for k, v in unet_state_dict.items():
                        adapters_state_dict[f"transformer.{k}"] = v.detach().clone().to("cpu").to(dtype)
            
            # Handle full_train_in_out modules
            if self.full_train_in_out:
                if hasattr(base_model_instance, 'transformer_pos_embed') and base_model_instance.transformer_pos_embed is not None:
                    adapters_state_dict.update({"transformer_pos_embed." + k: v.detach().clone().to("cpu").to(dtype) for k, v in base_model_instance.transformer_pos_embed.state_dict().items()})
                if hasattr(base_model_instance, 'transformer_proj_out') and base_model_instance.transformer_proj_out is not None:
                    adapters_state_dict.update({"transformer_proj_out." + k: v.detach().clone().to("cpu").to(dtype) for k, v in base_model_instance.transformer_proj_out.state_dict().items()})
                if hasattr(base_model_instance, 'unet_conv_in') and base_model_instance.unet_conv_in is not None:
                    adapters_state_dict.update({"unet_conv_in." + k: v.detach().clone().to("cpu").to(dtype) for k, v in base_model_instance.unet_conv_in.state_dict().items()})
                if hasattr(base_model_instance, 'unet_conv_out') and base_model_instance.unet_conv_out is not None:
                    adapters_state_dict.update({"unet_conv_out." + k: v.detach().clone().to("cpu").to(dtype) for k, v in base_model_instance.unet_conv_out.state_dict().items()})

            if extra_state_dict:
                adapters_state_dict.update(extra_state_dict)
            
            return adapters_state_dict

        # --- Original LoRA/LoCon/LoRM state_dict generation ---
        state_dict = self.state_dict()
        save_dict_formatted = OrderedDict()
        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(dtype)
            
            keymap_for_lora = self.get_keymap()
            save_key = key
            if keymap_for_lora is not None and key in keymap_for_lora:
                save_key = keymap_for_lora[key] # This might be the remapping
            
            if self.peft_format: # This refers to internal PEFT format for non-AdaLoRA
                new_key = save_key
                if new_key.endswith('.alpha'):
                    continue
                new_key = new_key.replace('lora_down', 'lora_A')
                new_key = new_key.replace('lora_up', 'lora_B')
                new_key = new_key.replace('$$', '.') # This is for internal toolkit representation
                save_key = new_key
            
            if self.network_type.lower() == "lokr":
                new_key = save_key
                new_key = new_key.replace('lora_transformer_', 'lycoris_')
                save_key = new_key
            
            save_dict_formatted[save_key] = v
            del state_dict[key]

        if extra_state_dict is not None:
            for key in list(extra_state_dict.keys()):
                v = extra_state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_dict_formatted[key] = v
        # --- Removed the base_model_ref().convert_lora_weights_before_save call from here ---
        return save_dict_formatted

    def save_weights(
            self: Network,
            file, dtype=torch.float16,
            metadata=None,
            extra_state_dict: Optional[OrderedDict] = None
    ):
        # For all network types, we now save to a single .safetensors file.
        # get_state_dict is responsible for collecting the correct format.
        consolidated_sd = self.get_state_dict(extra_state_dict=extra_state_dict, dtype=dtype)
        
        if metadata is not None and len(metadata) == 0:
            metadata = None

        if metadata is None:
            metadata = OrderedDict()
        metadata = add_model_hash_to_meta(consolidated_sd, metadata)
        
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(consolidated_sd, file, metadata)
        else:
            torch.save(consolidated_sd, file)

    def load_weights(self: Network, file, force_weight_mapping=False):
        from safetensors.torch import load_file
        full_state_dict = load_file(file, device="cpu")
        base_model_instance = self.base_model_ref()
        if base_model_instance is None:
            # If base_model_ref is None, we can't load full_train_in_out or PEFT adapters correctly
            # This should have been caught earlier during network init.
            raise ValueError("Base model reference is missing, cannot load network weights.")

        unet_peft_state_dict = OrderedDict()
        text_encoder_peft_state_dict_map = {}
        extra_dict = OrderedDict() # Will capture anything not directly handled (e.g., 'emb_params' for TI)

        # Distribute keys from the loaded .safetensors
        for key, value in full_state_dict.items():
            if self.network_type.lower() == "adalora":
                if key.startswith("transformer."): # UNet AdaLoRA
                    unet_peft_state_dict[key.replace("transformer.", "")] = value
                elif key.startswith("text_encoder."): # Text Encoder AdaLoRA
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
                # Handle full_train_in_out layers directly to base_model_instance
                elif key.startswith("transformer_pos_embed."):
                    if hasattr(base_model_instance, 'transformer_pos_embed') and base_model_instance.transformer_pos_embed is not None:
                        base_model_instance.transformer_pos_embed.load_state_dict({key.replace("transformer_pos_embed.", ""): value}, strict=False)
                    else: extra_dict[key] = value # Fallback if layer somehow doesn't exist
                elif key.startswith("transformer_proj_out."):
                    if hasattr(base_model_instance, 'transformer_proj_out') and base_model_instance.transformer_proj_out is not None:
                        base_model_instance.transformer_proj_out.load_state_dict({key.replace("transformer_proj_out.", ""): value}, strict=False)
                    else: extra_dict[key] = value
                elif key.startswith("unet_conv_in."):
                    if hasattr(base_model_instance, 'unet_conv_in') and base_model_instance.unet_conv_in is not None:
                        base_model_instance.unet_conv_in.load_state_dict({key.replace("unet_conv_in.", ""): value}, strict=False)
                    else: extra_dict[key] = value
                elif key.startswith("unet_conv_out."):
                    if hasattr(base_model_instance, 'unet_conv_out') and base_model_instance.unet_conv_out is not None:
                        base_model_instance.unet_conv_out.load_state_dict({key.replace("unet_conv_out.", ""): value}, strict=False)
                    else: extra_dict[key] = value
                else: # Any other keys (e.g., 'emb_params' for TI)
                    extra_dict[key] = value
            else: # Original LoRA/LoCon/LoRM load logic
                keymap = self.get_keymap(force_weight_mapping)
                keymap = {} if keymap is None else keymap
                load_key = keymap[key] if key in keymap else key

                if self.peft_format: # Internal PEFT-like format for non-AdaLoRA
                    if load_key.endswith('.alpha'): continue
                    load_key = load_key.replace('lora_A', 'lora_down').replace('lora_B', 'lora_up').replace('.', '$$')
                    load_key = load_key.replace('$$lora_down$$', '.lora_down.').replace('$$lora_up$$', '.lora_up.')
                
                if self.network_type.lower() == "lokr":
                    load_key = load_key.replace('lycoris_', 'lora_transformer_')

                # For traditional LoRA, keys might match directly, or be in extra_dict
                if load_key not in self.state_dict():
                    extra_dict[load_key] = value
                else:
                    self.state_dict()[load_key].copy_(value) # Directly copy value
        
        # Apply PEFT state dicts for AdaLoRA
        if self.network_type.lower() == "adalora":
            if self.peft_adapted_unet and unet_peft_state_dict:
                set_peft_model_state_dict(self.peft_adapted_unet, unet_peft_state_dict, adapter_name=f"{self.peft_adapter_name}_unet")
            if self.peft_adapted_text_encoders:
                for idx, te_model in enumerate(self.peft_adapted_text_encoders):
                    if idx in text_encoder_peft_state_dict_map:
                        set_peft_model_state_dict(te_model, text_encoder_peft_state_dict_map[idx], adapter_name=f"default_adalora_adapter_te_{idx}")
            print(f"Loaded AdaLoRA weights from {file}")
        else: # For other network types, apply the main state dict
            # The direct copy above might be enough, but to be robust for missing/unexpected keys
            load_sd_for_strict_loading = OrderedDict()
            for key, value in full_state_dict.items():
                keymap = self.get_keymap(force_weight_mapping)
                keymap = {} if keymap is None else keymap
                load_key = keymap[key] if key in keymap else key
                # Apply peft_format/lokr remapping to load_key if applicable
                if self.peft_format:
                    if load_key.endswith('.alpha'): continue
                    load_key = load_key.replace('lora_A', 'lora_down').replace('lora_B', 'lora_up').replace('.', '$$')
                    load_key = load_key.replace('$$lora_down$$', '.lora_down.').replace('$$lora_up$$', '.lora_up.')
                if self.network_type.lower() == "lokr":
                    load_key = load_key.replace('lycoris_', 'lora_transformer_')

                if load_key in self.state_dict():
                    load_sd_for_strict_loading[load_key] = value
                else:
                    extra_dict[load_key] = value # Add to extra if not directly loadable by network.

            info = self.load_state_dict(load_sd_for_strict_loading, strict=False)
            if info.missing_keys: print(f"Warning: LoRA loading missing keys: {info.missing_keys}")
            if info.unexpected_keys: print(f"Warning: LoRA loading unexpected keys: {info.unexpected_keys}")

        if len(extra_dict.keys()) == 0:
            extra_dict = None
        return extra_dict

    def _update_torch_multiplier(self: Network):
        if self.network_type.lower() == "adalora":
            # For AdaLoRA, multiplier is handled by PEFT's scaling, typically 1.0 here for the wrapper.
            device = torch.device("cpu") # Default, will be updated if PEFT model is on GPU
            dtype = torch.float32 # Default, will be updated if PEFT model is on GPU
            if hasattr(self, 'peft_adapted_unet') and self.peft_adapted_unet is not None and isinstance(self.peft_adapted_unet, peft.PeftModel):
                device = self.peft_adapted_unet.device
                dtype = self.peft_adapted_unet.dtype
            elif self.peft_adapted_text_encoders and len(self.peft_adapted_text_encoders) > 0 and isinstance(self.peft_adapted_text_encoders[0], peft.PeftModel):
                device = self.peft_adapted_text_encoders[0].device
                dtype = self.peft_adapted_text_encoders[0].dtype

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
            # For AdaLoRA, the effective multiplier is handled by PEFT's internal scaling.
            # This property reflects the conceptual multiplier for the overall network if needed.
            return self._multiplier # It's better to return the internal _multiplier value
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float], List[List[float]]]):
        if self.network_type.lower() == "adalora":
            self._multiplier = value # Store the value, but PEFT is in charge
            return

        if self._multiplier == value:
            return
        self._multiplier = value
        self._update_torch_multiplier()

    # called when the context manager is entered
    # ie: with network:
    def __enter__(self: Network):
        self.is_active = True

    def __exit__(self: Network, exc_type, exc_value, tb):
        self.is_active = False

    def force_to(self: Network, device, dtype):
        if self.network_type.lower() == "adalora":
            # PEFT models handle their own device/dtype.
            # The LoRASpecialNetwork itself is mostly a container.
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
            # For AdaLoRA, the trainable modules are the PEFT adapters themselves,
            # not individual LoRAModule instances managed by the network here.
            return []
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        return loras

    def _update_checkpointing(self: Network):
        if self.network_type.lower() == "adalora":
            # Checkpointing for AdaLoRA is handled by PEFT
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
            # Call super().enable_gradient_checkpointing if it exists and is applicable,
            # otherwise just use _update_checkpointing
            if hasattr(super(), 'enable_gradient_checkpointing'):
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
            if hasattr(super(), 'disable_gradient_checkpointing'):
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