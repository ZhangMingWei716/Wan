from __future__ import annotations

import os
import json
import functools
from typing import Dict, List, Type, Tuple

import torch
from torch import Tensor, nn

try:
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import LinearQuantizer
    from msmodelslim.pytorch.lowbit.quant_modules import (
        LinearQuantizer as LowBitLinearQuantizer,
    )
except ImportError:
    LinearQuantizer = nn.Linear
    LowBitLinearQuantizer = nn.Linear


LoaderT = List[torch.Tensor]
ModelCacheT = Dict[str, List[Tensor]]


def get_linear_layers_classes() -> Tuple[Type[nn.Module], ...]:
    return nn.Linear, LinearQuantizer, LowBitLinearQuantizer


def find_layers(
    module: nn.Module,
    layers: Tuple[Type[nn.Module], ...] = get_linear_layers_classes(),
    name: str = "",
) -> dict[str, nn.Module]:
    if type(module) in layers:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def catch_model_cache(
    model: nn.Module,
    layers: List[nn.Module],
    calibration_samples: Tensor,
) -> tuple[List[Tensor], ModelCacheT]:
    layers[0] = layers[0].npu()
    inps: List[Tensor] = []
    attention_mask: List[Tensor] = []
    position_ids: List[Tensor] = []
    cache_position: List[Tensor] = []
    position_embeddings: List[Tensor] = []
    mask: List[Tensor] = []
    start_pos: List[int] = []
    freqs_cis: List[Tensor] = []

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module
            # the attr below appeared in transformers >= 4.53 for qwen DecoderLayer
            # and is accessed directly, so we need to make a link to it
            # TODO: find general solution for any transformers models
            self.attention_type = getattr(self.module, "attention_type", None)

        def forward(self, *args, **kwargs):  # noqa
            # kwargs: ['attention_mask', 'position_ids', 'past_key_value',
            # 'output_attentions', 'use_cache', 'cache_position',
            # 'position_embeddings']
            inps.append(args)
            if "attention_mask" in kwargs:
                attention_mask.append(kwargs["attention_mask"])
            if "position_ids" in kwargs:
                if not position_ids:
                    position_ids.append(kwargs["position_ids"])
            if "cache_position" in kwargs:
                if not cache_position:
                    cache_position.append(kwargs["cache_position"])
            if "position_embeddings" in kwargs:
                if not position_embeddings:
                    position_embeddings.append(kwargs["position_embeddings"])
            if "mask" in kwargs:
                if not mask:
                    mask.append(kwargs["mask"])
            if "start_pos" in kwargs:
                if not start_pos:
                    start_pos.append(kwargs["start_pos"])
            if "freqs_cis" in kwargs:
                if not freqs_cis:
                    freqs_cis.append(kwargs["freqs_cis"].npu())
            raise ValueError

    layers[0] = Catcher(layers[0])
    for tokens in calibration_samples:
        try:
            model.model(tokens.npu())
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model_cache = {}  # TODO: add optional offload to CPU?
    if attention_mask:
        model_cache["attention_mask"] = attention_mask[0]
    if position_ids:
        model_cache["position_ids"] = position_ids[0]
    if cache_position:
        model_cache["cache_position"] = cache_position[0]
    if position_embeddings:
        model_cache["position_embeddings"] = position_embeddings[0]
    if mask:
        model_cache["mask"] = mask[0]
    if start_pos:
        model_cache["start_pos"] = start_pos[0]
    if freqs_cis:
        model_cache["freqs_cis"] = freqs_cis[0]

    return inps, model_cache

def check_if_deepseek_v3_2(model_path):
    try:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return False
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        model_type = config.get("model_type", "")
        
        if model_type.lower() == "deepseek_v32":
            return True

        return False
    except Exception as e:
        print(f"Error checking model type.: {e}")
        return False
