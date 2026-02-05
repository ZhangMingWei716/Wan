# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def shard_model(
    model, # 需要进行 FSDP 分片的原始 PyTorch 模型
    device_id, # 当前进程对应的GPU 设备 ID
    param_dtype=torch.bfloat16, # 模型参数的数据类型
    reduce_dtype=torch.float32, # 梯度规约阶段的数据类型
    buffer_dtype=torch.float32, # 模型缓冲区的数据类型
    process_group=None, # 分布式进程组，默认None表示使用 PyTorch 的全局默认进程组
    sharding_strategy=ShardingStrategy.FULL_SHARD, # FSDP分片策略，默认用FULL_SHARD(全分片)
    sync_module_states=True, # 是否同步多卡的模型参数状态
    use_lora=False # 是否为LoRA 微调场景，默认False
):
    model = FSDP(
        module=model, # 指定需要被 FSDP 封装的原始模型
        process_group=process_group, # 传入分布式进程组
        sharding_strategy=sharding_strategy, # 传入分片策略
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        # 自动分片策略，指定 FSDP按模型的哪些子模块进行分片，假设原始模型（如 WanModel、T5）中存在blocks属性
        # 配置 FSDP 的混合精度策略
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id, # 将 FSDP 模型绑定到当前进程的指定 GPU
        sync_module_states=sync_module_states, # 开启多卡模型参数状态同步，分布式初始化时保证所有 GPU 的模型参数完全一致。
        use_orig_params=True if use_lora else False) # LoRA 微调的关键适配配置
    return model


def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
