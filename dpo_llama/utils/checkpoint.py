# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

import os
import json
import torch
import logging
from typing import Union

from dpo_llama.models.model import Transformer
from dpo_llama.models.lora import lora_state_dict
from dpo_llama.utils.fsdp_checkpoint import save_full_state_model_checkpoint


logger = logging.getLogger(__name__)


def _check_file_path(file_path) -> None:
    if not file_path or file_path == '':
        raise ValueError(f'Invalid checkpoint file path {file_path!r}')
    if os.path.exists(file_path):
        logger.warning(f'Existing checkpoint file at {file_path!r} will be overwritten')


def _save_model_meta(model: Transformer, save_dir: str) -> None:
    meta_file = os.path.join(save_dir, 'params.json')
    if not os.path.exists(meta_file):
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(model.params.dict(), f, indent=2)
        logger.info(f'Model meta params saved at {meta_file!r}')


def create_checkpoint(model: Transformer, full_path: str) -> None:
    _check_file_path(full_path)
    ckpt_state = model.state_dict()
    torch.save(ckpt_state, full_path)
    logger.info(f'Model checkpoint saved at {full_path!r}')

    _save_model_meta(model, os.path.dirname(full_path))


def create_lora_checkpoint(model: Transformer, full_path: str, train_bias: bool) -> None:
    _check_file_path(full_path)

    ckpt_state = lora_state_dict(model, train_bias=train_bias)
    torch.save(ckpt_state, full_path)
    logger.info(f'LoRA model checkpoint saved at {full_path!r}')

    _save_model_meta(model, os.path.dirname(full_path))


def create_fsdp_full_checkpoint(model: Transformer, rank: int, full_path: str) -> None:
    _check_file_path(full_path)

    save_full_state_model_checkpoint(model, rank, full_path, True)
    logger.info(f'FSDP model full state checkpoint saved at {full_path!r}')

    _save_model_meta(model, os.path.dirname(full_path))
