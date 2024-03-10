# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Run direct preference optimization (DPO) using QLoRA, starting with a SFT model."""
import os
import functools
from typing import Tuple, Mapping, Text, Any, Dict
import tqdm
import random

import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from dpo_llama.models.model_lora import Transformer, LoraModelArgs
from dpo_llama.models.tokenizer import Tokenizer
from dpo_llama.models.lora import mark_only_lora_as_trainable

from dpo_llama.configs.dpo_lora import config as cfg
from dpo_llama.utils.custom_dataset import PreferenceDataset
from dpo_llama.utils.schedule import CosineDecayWithWarmupLRScheduler
from dpo_llama.utils.train_helper import (
    create_trace_profiler,
    create_optimizer,
    compute_num_trainable_params,
    get_grad_norm_local,
    optimizer_to,
)
from dpo_llama.utils.logger import create_logger, log_statistics
from dpo_llama.utils.tracker import DPOStatsTracker
from dpo_llama.utils.checkpoint import create_lora_checkpoint

logger = create_logger()


def clear_gpu_cache():
    torch.cuda.empty_cache()


def compute_preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Code copied from: https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L45

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size, )
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size, )
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size, )
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size, )
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    assert 0 < beta <= 1

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def extract_chosen_and_reject_logprobs(
    pi_logits: torch.Tensor, ref_logprobs: torch.Tensor, token_target: torch.Tensor, loss_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(pi_logits.shape) == 3
    assert len(ref_logprobs.shape) == 2
    assert len(token_target.shape) == 2
    assert len(loss_mask.shape) == 2

    pi_logprobs = torch.log_softmax(pi_logits, dim=2)  # [batch_size, seq_len, vocab_size]
    pi_logprobs = torch.gather(pi_logprobs, dim=2, index=token_target.unsqueeze(2)).squeeze(2)  # [batch_size, seq_len]

    pi_logprobs = (pi_logprobs * loss_mask.detach()).sum(1)  # [batch_size]
    ref_logprobs = (ref_logprobs * loss_mask.detach()).sum(1)  # [batch_size]

    half_size = token_target.shape[0] // 2

    pi_chosen_logprobs = pi_logprobs[:half_size]
    ref_chosen_logprobs = ref_logprobs[:half_size]

    pi_rejected_logprobs = pi_logprobs[half_size:]
    ref_rejected_logprobs = ref_logprobs[half_size:]

    return pi_chosen_logprobs, pi_rejected_logprobs, ref_chosen_logprobs, ref_rejected_logprobs


def train_step(
    model: Transformer,
    batch: Dict[Text, torch.Tensor],
    scaler: torch.cuda.amp.GradScaler,
    gradient_accum_steps: int,
    tracker: DPOStatsTracker,
    dpo_beta: float,
    dpo_label_smoothing: float,
    dpo_reference_free: bool,
    use_ipo_loss: bool,
) -> None:
    """Run a single training step, where we do a forward + backward passes, but do no update parameters"""

    assert gradient_accum_steps >= 1

    token_input = batch['token_input'].to('cuda', non_blocking=True)
    token_target = batch['token_target'].to('cuda', non_blocking=True)
    loss_mask = batch['loss_mask'].to('cuda', non_blocking=True)
    ref_logprobs = batch['ref_logprobs'].to('cuda', non_blocking=True)
    pi_logits = model(token_input)

    pi_chosen_logprobs, pi_rejected_logprobs, ref_chosen_logprobs, ref_rejected_logprobs = extract_chosen_and_reject_logprobs(pi_logits, ref_logprobs, token_target, loss_mask)

    (losses, chosen_rewards, rejected_rewards) = compute_preference_loss(
        pi_chosen_logprobs,
        pi_rejected_logprobs,
        ref_chosen_logprobs,
        ref_rejected_logprobs,
        beta=dpo_beta,
        label_smoothing=dpo_label_smoothing,
        ipo=use_ipo_loss,
        reference_free=dpo_reference_free,
    )

    loss = losses.mean()
    # scale the loss to account for gradient accumulation
    scaled_loss = loss / gradient_accum_steps

    if scaler is not None:  # when using float16
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    tracker.update(losses.detach(), chosen_rewards, rejected_rewards)


def update_step(
    model: Transformer,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler = None,
) -> torch.Tensor:
    """Run a single parameter update step"""
    grad_norm = get_grad_norm_local(model)

    if grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if scaler is not None:  # when using float16
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        optimizer.step()

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    return grad_norm


@torch.no_grad()
def run_validation_steps(
    model: Transformer,
    loader: DataLoader,
    steps: int,
    tracker: DPOStatsTracker,
    dpo_beta: float,
    dpo_label_smoothing: float,
    dpo_reference_free: bool,
    use_ipo_loss: bool,
) -> None:
    """Run M validation steps"""

    val_pbar = tqdm.tqdm(range(steps), colour='green', desc='Validation steps')
    for i, (batch) in enumerate(loader):
        token_input = batch['token_input'].to('cuda', non_blocking=True)
        token_target = batch['token_target'].to('cuda', non_blocking=True)
        loss_mask = batch['loss_mask'].to('cuda', non_blocking=True)
        ref_logprobs = batch['ref_logprobs'].to('cuda', non_blocking=True)
        pi_logits = model(token_input)

        pi_chosen_logprobs, pi_rejected_logprobs, ref_chosen_logprobs, ref_rejected_logprobs = extract_chosen_and_reject_logprobs(pi_logits, ref_logprobs, token_target, loss_mask)

        (losses, chosen_rewards, rejected_rewards) = compute_preference_loss(
            pi_chosen_logprobs,
            pi_rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
            beta=dpo_beta,
            label_smoothing=dpo_label_smoothing,
            ipo=use_ipo_loss,
            reference_free=dpo_reference_free,
        )

        tracker.update(losses.detach(), chosen_rewards, rejected_rewards)
        val_pbar.update(1)

        if i >= steps:
            break

    val_pbar.close()


def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor]:
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    and compute the loss mask for the batch.
    """

    chosen_seqlen = [len(item['chosen_tokens']) for item in batch]
    rejected_seqlen = [len(item['rejected_tokens']) for item in batch]

    max_batch_seqlen = max(chosen_seqlen + rejected_seqlen)
    assert max_batch_seqlen <= max_seq_len
    if full_pad:
        max_batch_seqlen = max_seq_len

    # concatenate chosen and rejected completions together, where the first half are chosen sequences, and last half are rejected sequences
    batch_size = len(batch) * 2
    half_size = len(batch)

    batch_input = torch.full((batch_size, max_batch_seqlen), pad_id, dtype=torch.long)
    batch_ref_logprobs = torch.full((batch_size, max_batch_seqlen), 0.0, dtype=torch.float)

    # loss mask where 0s at the beginning are prompt tokens, 1s are completion tokens, and 0s at the ending are padding tokens
    batch_loss_mask = torch.full((batch_size, max_batch_seqlen), 0, dtype=torch.long)

    for i, item in enumerate(batch):
        chosen_tokens = item['chosen_tokens'].type(torch.long)
        rejected_tokens = item['rejected_tokens'].type(torch.long)
        chosen_ref_logprobs = item['chosen_ref_logprobs'].type(torch.float)
        rejected_ref_logprobs = item['rejected_ref_logprobs'].type(torch.float)

        len_prompt = item['len_prompt']
        len_chosen = len(chosen_tokens)
        len_rejected = len(rejected_tokens)

        assert len(chosen_ref_logprobs) == len_chosen
        assert len(rejected_ref_logprobs) == len_rejected

        # Chosen sequences
        batch_input[i, :len_chosen] = chosen_tokens
        batch_ref_logprobs[i, :len_chosen] = chosen_ref_logprobs
        batch_loss_mask[i, len_prompt : len_chosen - 1] = 1  # -1 because our target is shifted one step to the left

        # Rejected sequences
        batch_input[i + half_size, :len_rejected] = rejected_tokens
        batch_ref_logprobs[i + half_size, :len_rejected] = rejected_ref_logprobs
        batch_loss_mask[i + half_size, len_prompt : len_rejected - 1] = 1

    # shift one step to get target for computing log probabilities
    batch_target = torch.full((batch_size, max_batch_seqlen), pad_id, dtype=torch.long)
    batch_target[:, :-1] = batch_input[:, 1:].clone()

    return {
        'token_input': batch_input,
        'token_target': batch_target,
        'loss_mask': batch_loss_mask,
        'ref_logprobs': batch_ref_logprobs,
    }


def main():
    assert cfg.num_epochs >= 1
    assert cfg.train_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 0
    assert cfg.val_steps >= 1

    assert 0 < cfg.dpo_beta <= 1
    assert 0 <= cfg.dpo_label_smoothing <= 1

    if not torch.version.cuda:
        raise RuntimeError('This script requires Pytorch with CUDA.')

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint {cfg.pretrain_ckpt_file!r}, aborting ...')

    # --------------- Load datasets ---------------

    logger.info('Loading datasets ...')

    tokenizer = Tokenizer(cfg.tokenizer_file)

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=cfg.max_seq_len,
        full_pad=cfg.full_pad,
    )

    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': cfg.dataloader_workers,
        'pin_memory': False,
        'shuffle': True,
    }

    train_dataset = PreferenceDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)
    train_kwargs = {'batch_size': cfg.train_batch_size, 'sampler': None}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = PreferenceDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)
        val_kwargs = {'batch_size': cfg.val_batch_size, 'sampler': None}
        val_kwargs.update(cuda_kwargs)
        val_loader = DataLoader(val_dataset, **val_kwargs)
        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    batch_size = int(cfg.train_batch_size * cfg.gradient_accum_steps)
    steps_per_epoch = len(train_loader) // cfg.gradient_accum_steps
    max_train_steps = steps_per_epoch * cfg.num_epochs

    # --------------- Setup model and optimizer ---------------

    logger.info('Initializing model and optimizer ...')

    torch.cuda.set_device('cuda:0')
    clear_gpu_cache()

    compute_dtype = torch.float32
    scaler = None
    if cfg.mixed_precision:
        if torch.version.cuda and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    else:
        logger.info('Training in float32 mode, make sure you have enough GPU RAM')

    model_args = LoraModelArgs.from_model_type(
        model_type=cfg.model_type,
        # LoRA configurations
        lora_r=cfg.lora_r,
        lora_scaling=cfg.lora_scaling,
        lora_dropout=cfg.lora_dropout,
        # LoRA trainable layers
        lora_attn_query=cfg.lora_attn_query,
        lora_attn_key=cfg.lora_attn_key,
        lora_attn_value=cfg.lora_attn_value,
        lora_attn_proj=cfg.lora_attn_proj,
        lora_attn_mlp=cfg.lora_attn_mlp,
        lora_head=cfg.lora_head,
        # Quantization configurations
        quant_4bit=cfg.quant_4bit,
        quant_lora_4bit=cfg.quant_lora_4bit,
        quant_4bit_double=cfg.quant_4bit_double,
        quant_4bit_type=cfg.quant_4bit_type,
        quant_compute_dtype=compute_dtype,
        # Regular configurations
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.max_seq_len,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
        head_dropout=cfg.head_dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    model = Transformer(model_args)

    # Load model checkpoint using strict=False,
    # because there are missing keys due to LoRA weights not contained in checkpoint state
    logger.info(f'Loading pretrained checkpoint {cfg.pretrain_ckpt_file!r} ...')
    model_state = torch.load(cfg.pretrain_ckpt_file)
    model.load_state_dict(model_state, strict=False)
    del model_state

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        module = module.to(dtype=compute_dtype)

    mark_only_lora_as_trainable(model, train_bias=cfg.train_bias)

    # This is where the weights quantization happens
    # when we move the model to cuda, the bnb.nn.Params4bit.cuda() method is called,
    # and the weights is quantized using bnb.functional.quantize_4bit
    model = model.to('cuda')

    torch.cuda.empty_cache()

    logger.info('Initializing optimizer ...')
    num_trainable, num_frozen = compute_num_trainable_params(model)
    logger.info(f'Number of trainable parameters: {num_trainable:,}')
    logger.info(f'Number of frozen parameters: {num_frozen:,}')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        paged_adamw=cfg.use_paged_adamw,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=cfg.init_lr,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=int(cfg.warmup_ratio * max_train_steps),
        max_decay_steps=max_train_steps,
    )

    # --------------- Start Training ---------------

    create_ckpt_func = functools.partial(create_lora_checkpoint, train_bias=cfg.train_bias)

    torch_profiler = None
    tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))
    train_pbar = tqdm.tqdm(range(max_train_steps), colour='blue', desc='Training steps')
    best_val_loss = np.inf
    train_steps = 0

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    train_tracker = DPOStatsTracker()
    val_tracker = DPOStatsTracker()

    logger.info(f'Starting to run {cfg.num_epochs} training epochs, total of {max_train_steps} steps, with batch size {batch_size}')

    for epoch in range(1, cfg.num_epochs + 1):  # for each epoch
        logger.info(f'Start epoch {epoch}')
        model.train()
        train_tracker.reset()
        val_tracker.reset()

        for i, batch in enumerate(train_loader):  # for each batch in current epoch
            train_step(
                model=model,
                batch=batch,
                scaler=scaler,
                gradient_accum_steps=cfg.gradient_accum_steps,
                tracker=train_tracker,
                dpo_beta=cfg.dpo_beta,
                dpo_label_smoothing=cfg.dpo_label_smoothing,
                dpo_reference_free=cfg.dpo_reference_free,
                use_ipo_loss=cfg.use_ipo_loss,
            )

            if i % cfg.gradient_accum_steps == 0:
                grad_norm = update_step(model, optimizer, scheduler, cfg.grad_clip, scaler)
                train_pbar.update(1)
                train_steps += 1

                if torch_profiler is not None:
                    torch_profiler.step()

                train_stats = train_tracker.get_dict(reset=True)
                train_stats['learning_rate'] = optimizer.param_groups[0]['lr']
                train_stats['grad_norm'] = grad_norm.item()

                # logging training statistics
                if train_steps % cfg.log_interval == 0:
                    log_statistics(tb_writer, train_steps, train_stats, True)

                # regular checkpointing
                if cfg.ckpt_interval > 0 and (train_steps % cfg.ckpt_interval == 0 or train_steps == max_train_steps):
                    create_ckpt_func(model=model, full_path=os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-steps-{train_steps}.pth'))

                # validation steps
                if cfg.val_steps > 0 and (cfg.val_interval > 0 and train_steps % cfg.val_interval == 0 or train_steps == max_train_steps):
                    model.eval()
                    optimizer_to(optimizer, 'cpu')  # move optimizer to cpu so we can use larger batch size for validation
                    run_validation_steps(
                        model=model,
                        loader=val_loader,
                        steps=cfg.val_steps,
                        tracker=val_tracker,
                        dpo_beta=cfg.dpo_beta,
                        dpo_label_smoothing=cfg.dpo_label_smoothing,
                        dpo_reference_free=cfg.dpo_reference_free,
                        use_ipo_loss=cfg.use_ipo_loss,
                    )
                    model.train()
                    optimizer_to(optimizer, 'cuda')

                    val_stats = val_tracker.get_dict(reset=True)
                    log_statistics(tb_writer, train_steps, val_stats, False)

                    # save best model
                    if val_stats['loss'] < best_val_loss:
                        best_val_loss = val_stats['loss']
                        logger.info(f'New best validation loss: {best_val_loss:.4f}')
                        create_ckpt_func(model=model, full_path=os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-best.pth'))

    # final checkpoint after training is finished
    create_ckpt_func(model=model, full_path=os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-steps-{train_steps}.pth'))

    # show some training stats
    logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
