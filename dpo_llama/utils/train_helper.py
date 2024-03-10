# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, List, Mapping, Text, Any
import logging
import numpy as np
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp.api import ShardingStrategy

import bitsandbytes as bnb

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

logger = logging.getLogger(__name__)


def create_trace_profiler(tb_trace_dir: str) -> torch.profiler.profile:
    torch_profiler = torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
        profile_memory=True,
        with_stack=False,
        record_shapes=False,
    )

    return torch_profiler


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    eps: float,
    weight_decay: float,
    betas: Tuple[float],
    fused: bool = False,
    paged_adamw: bool = False,
) -> torch.optim.AdamW:
    """
    Returns the PyTorch AdamW optimizer for the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    # Create empty lists to store parameters for weight decay and no weight decay.
    decay = []
    no_decay = []

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad

        if is_trainable:
            # Check for parameters corresponding to torch.nn.LayerNorm or torch.nn.Embedding.
            # Note we use hard-coded names where 'ln' is for LayerNorm, and 'embed' is for Embedding, this works better with FSDP
            if (
                p_name.endswith('bias')
                or p_name.endswith('attention_norm.weight')
                or p_name.endswith('ffn_norm.weight')
                or p_name.endswith('post_norm.weight')
                or p_name.endswith('token_embeddings.weight')
            ):
                no_decay.append(params)
            else:
                decay.append(params)

    if weight_decay > 0:
        num_decay_params = sum(p.numel() for p in decay)
        num_nodecay_params = sum(p.numel() for p in no_decay)
        logger.info(f'Number of decayed parameters: {num_decay_params:,}')
        logger.info(f'Number of non-decayed parameters: {num_nodecay_params:,}')

    # create the pytorch optimizer object
    optim_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    kwargs = {
        'lr': lr,
        'eps': eps,
        'betas': betas,
    }

    if paged_adamw:
        optimizer = bnb.optim.PagedAdamW(optim_groups, **kwargs)
    else:
        kwargs['fused'] = fused
        optimizer = torch.optim.AdamW(optim_groups, **kwargs)

    return optimizer


def compute_num_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    num_trainable_params = 0
    num_frozen_params = 0

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad
        is_quantized = hasattr(params, 'quant_state')

        # quantized layer is not trainable
        if not is_trainable and is_quantized:
            num_params = math.prod(params.quant_state.shape)
        else:
            num_params = params.numel()

        num_trainable_params += num_params if is_trainable else 0
        num_frozen_params += num_params if not is_trainable else 0

    return num_trainable_params, num_frozen_params


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor_sum = masked_sum(tensor, mask, dim)
    mask_sum = mask.sum(dim=dim)

    # Avoid division by zero
    mask_sum = torch.where(mask_sum <= 0, 1e-8, mask_sum)
    mean = tensor_sum / mask_sum
    return mean


def masked_sum(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    masked_tensor = tensor * mask
    tensor_sum = masked_tensor.sum(dim=dim)
    return tensor_sum


def masked_whiten(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8, shift_mean: bool = True) -> torch.Tensor:
    masked_tensor = tensor * mask
    mean = masked_mean(masked_tensor, mask, dim=dim)

    if len(tensor.shape) > len(mean.shape) and len(mean.shape) == 1:
        mean = mean.unsqueeze(1)

    mean_centered = masked_tensor - mean

    var = masked_mean(mean_centered**2, mask, dim=dim)
    if len(tensor.shape) > len(var.shape) and len(var.shape) == 1:
        var = var.unsqueeze(1)

    # Avoid potential division by zero
    var = torch.where(var <= 0, eps, var)

    whitened = mean_centered * var.clamp(min=eps).rsqrt()

    if not shift_mean:
        whitened += mean

    return whitened


def get_grad_norm_local(model) -> torch.Tensor:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return total_norm**0.5


def get_grad_norm_fsdp(model, rank, world_size, sharding_strategy=ShardingStrategy.FULL_SHARD) -> torch.Tensor:
    local_norm = get_grad_norm_local(model)
    op = torch.distributed.ReduceOp.SUM
    return_norm = local_norm.clone().detach().requires_grad_(False).to(rank) ** 2
    dist.all_reduce(return_norm, op=op)
    if sharding_strategy == ShardingStrategy.NO_SHARD:
        return_norm = return_norm / world_size
    return return_norm**0.5


def optimizer_to(optim: torch.optim.Optimizer, device: str):
    """Move pytorch optimizer to some device

    Code copied from
    https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
