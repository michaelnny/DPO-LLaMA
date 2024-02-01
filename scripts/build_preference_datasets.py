# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Module to build preference comparison dataset"""
from typing import Tuple, List, Mapping, Text, Any, Dict
import functools
import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
import shutil
import json
import random
import pickle
import re
import torch
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import bs4

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from dpo_llama.models.tokenizer import Tokenizer
from dpo_llama.utils.logger import create_logger
from dpo_llama.utils.file_helper import (
    find_certain_files_under_dir,
    read_json_file,
    read_jsonl_file,
    read_zipped_jsonl_file,
    count_words,
)
from dpo_llama.utils.prompt_builder import build_prompt_completion, Dialog
from dpo_llama.models.model import Transformer, ModelArgs

logger = create_logger()

Metadata = Mapping[Text, Text]

DEFAULT_SYSTEM_PROMPT = {
    'role': 'system',
    'content': '',
}

# this will be inserted into the training data as the first system prompt
DEFAULT_DIALOG = [DEFAULT_SYSTEM_PROMPT]


# ----------------------------------- helper functions -----------------------------------

KEYWORDS_TO_SKIP = ['photo', 'video', 'movie', 'youtube', 'YouTube']


Answers = List[Mapping[Text, Any]]


def _split_and_save_datasets(
    datasets: List[dict],
    validation_ratio: float,
    train_output_file: str,
    val_output_file: str,
    meta_output_file: str,
    meta: dict,
) -> None:
    # split into train and validation datasets as dolly only have one single .json file
    random.shuffle(datasets)

    val_size = int(len(datasets) * validation_ratio)
    train_size = len(datasets) - val_size

    train_set, val_set = torch.utils.data.random_split(datasets, [train_size, val_size])

    for data, out_file in zip((train_set, val_set), (train_output_file, val_output_file)):
        if len(data) > 0:
            logger.info(f'Saving {len(data)} processed data to {out_file!r} ...')
            pickle.dump(data, open(out_file, 'wb'))

    meta = {
        **meta,
        'num_train_samples': len(train_set),
        'num_validation_samples': len(val_set),
    }

    logger.info(f'Saving metadata to {meta_output_file!r} ...')

    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _string_found(string1: str, string2: str) -> bool:
    if re.search(r'\b' + re.escape(string1) + r'\b', string2):
        return True
    return False


def _sort_answers_by_score_desc(answers: Answers) -> Answers:
    out = sorted(answers, key=lambda d: d['pm_score'], reverse=True)
    return out


def _deduplicate_answers_by_score(answers: Answers, shuffle: bool = True) -> Answers:
    if shuffle:
        # add some randomness so we are not always using the first occurrence of some score
        random.shuffle(answers)

    scores = [a['pm_score'] for a in answers]

    if len(answers) == len(set(scores)):
        return answers

    _, unique_indices = np.unique(scores, return_index=True)

    out = [answers[i] for i in unique_indices]

    assert len(out) == len(set(scores))

    return out


def _question_contains_skip_words(question: str) -> bool:
    if any(_string_found(question, k) for k in KEYWORDS_TO_SKIP):
        return True
    return False


def _filter_answers(answers: Answers, max_responses: int) -> Answers:
    assert max_responses >= 2

    answers = _deduplicate_answers_by_score(answers)
    answers = _sort_answers_by_score_desc(answers)

    if len(answers) > max_responses:
        answers = answers[:max_responses]

    return answers


def _extract_text(input_string: str) -> str:
    """Extract raw text from the given string, since it often contains lots of HTML tags."""
    soup = bs4.BeautifulSoup(input_string, features='html.parser')

    out = soup.text.replace('\n', ' ').replace('  ', ' ')  # .replace("\'", "'")
    out = out.strip()
    return out


def _process_single_stackexchange_file(
    file_path: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_responses: int,
) -> List[Tuple[int]]:
    """
    Read one single .parquet file and go over each row to build the dataset samples.

    For each row, we apply these:
        * Check if question contains some key words which we should skip (like pictures, movies etc)
        * Filter answers by apply the following rules:
            - Remove (semi-randomly) answers with duplicate scores
            - Sort answer by score in descending order
        * Build and tokenize prompt with standard chat format
        * Tokenize each answer, skip the answer if the length of prompt tokens + answer tokens are greater than max_seq_len

    """
    df = pq.read_table(file_path).to_pandas()

    samples = []

    for index, row in df.iterrows():
        question = row['question']
        answers = row['answers']

        if _question_contains_skip_words(question):
            continue

        # filter and sort answers from best (index 0) to worse
        answers = _filter_answers(answers, max_responses=max_responses)
        if len(answers) < 2:
            continue

        # build prompt tokens once
        dialog_prompt = DEFAULT_DIALOG + [
            {'role': 'user', 'content': question},
        ]

        for i in range(0, len(answers) - 1):  # break multiple responses into subsets, each subset contains a pair of chose:reject responses
            chosen_answer = _extract_text(answers[i]['text'])
            rejected_answer = _extract_text(answers[i + 1]['text'])

            chosen_dialog = dialog_prompt + [
                {'role': 'assistant', 'content': chosen_answer.strip()},
            ]
            rejected_dialog = dialog_prompt + [
                {'role': 'assistant', 'content': rejected_answer.strip()},
            ]

            prompt_tokens, chosen_tokens = build_prompt_completion(chosen_dialog, tokenizer)
            prompt_tokens, rejected_tokens = build_prompt_completion(rejected_dialog, tokenizer)

            if len(prompt_tokens) + len(chosen_tokens) > max_seq_len or len(prompt_tokens) + len(rejected_tokens) > max_seq_len:
                continue

            item = {}
            item['chosen_tokens'] = prompt_tokens + chosen_tokens  # chosen prompt + completion tokens
            item['rejected_tokens'] = prompt_tokens + rejected_tokens  # rejected prompt + completion tokens

            item['len_prompt'] = len(prompt_tokens)
            item['len_chosen_completion'] = len(chosen_tokens)
            item['len_rejected_completion'] = len(rejected_tokens)

            samples.append(item)

    return samples


def _convert_to_llama_chat_format(raw_text) -> Dialog:
    dialog = []
    conversations = raw_text.split('\n\nHuman: ')[1:]

    for pair in conversations:
        # standardize some punctuation
        pair = pair.replace(',  ', ', ').replace('.  ', '. ').replace('?  ', '? ').replace('!  ', '! ')
        contents = pair.split('\n\nAssistant: ')
        # skip some bad samples
        if len(contents) != 2:
            return dialog

        dialog.append({'role': 'user', 'content': contents[0]})
        dialog.append({'role': 'assistant', 'content': contents[1]})

    return dialog


def _process_single_hh_rlhf_jsonl_file(
    file_path: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> List[Tuple[int]]:
    """
    Read one single .jsonl.gz file and go over each row to build the dataset sample
    """

    samples = []

    for row in read_zipped_jsonl_file(file_path):
        chosen_dialog = _convert_to_llama_chat_format(row['chosen'])
        rejected_dialog = _convert_to_llama_chat_format(row['rejected'])

        if len(chosen_dialog) == 0 or len(rejected_dialog) == 0:
            continue

        chosen_dialog = DEFAULT_DIALOG + chosen_dialog
        prompt_tokens, chosen_tokens = build_prompt_completion(chosen_dialog, tokenizer)

        if len(prompt_tokens) + len(chosen_tokens) > max_seq_len:
            continue

        rejected_dialog = DEFAULT_DIALOG + rejected_dialog
        _, rejected_tokens = build_prompt_completion(rejected_dialog, tokenizer)

        if len(prompt_tokens) + len(rejected_tokens) > max_seq_len:
            continue

        item = {}
        item['chosen_tokens'] = prompt_tokens + chosen_tokens  # chosen prompt + completion tokens
        item['len_prompt'] = len(prompt_tokens)
        item['len_chosen_completion'] = len(chosen_tokens)

        item['rejected_tokens'] = prompt_tokens + rejected_tokens  # rejected prompt + completion tokens
        item['len_rejected_completion'] = len(rejected_tokens)

        samples.append(item)

    return samples


def compute_reference_logprobs(datasets: List[Dict], reference_model: Transformer, batch_size: int) -> List[Dict]:
    logger.info(f'Computing logprobs using reference model for {len(datasets)} samples ...')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    reference_model.eval()
    reference_model.to(device)

    num_batches = math.ceil(len(datasets) / batch_size)
    pbar = tqdm.tqdm(range(num_batches), colour='green', desc='Processing batches')

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        if end_idx > len(datasets):
            end_idx = len(datasets)

        current_batch = datasets[start_idx:end_idx]
        current_batch_size = len(current_batch)

        max_seqlen_chosen = max([len(item['chosen_tokens']) for item in current_batch])
        max_seqlen_rejected = max([len(item['rejected_tokens']) for item in current_batch])

        # concatenate prompt, chosen completion together
        batch_chosen_input = torch.full((current_batch_size, max_seqlen_chosen), tokenizer.eos_id, dtype=torch.long, device=device)
        batch_rejected_input = torch.full((current_batch_size, max_seqlen_rejected), tokenizer.eos_id, dtype=torch.long, device=device)

        for i, item in enumerate(current_batch):
            chosen_tokens, rejected_tokens = (item['chosen_tokens'], item['rejected_tokens'])

            chosen_seq = torch.tensor(chosen_tokens).type(torch.long)
            rejected_seq = torch.tensor(rejected_tokens).type(torch.long)

            # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
            batch_chosen_input[i, : len(chosen_seq)] = chosen_seq
            batch_rejected_input[i, : len(rejected_seq)] = rejected_seq

        # shift one step to get targets for computing log probabilities
        batch_chosen_target = torch.full((current_batch_size, max_seqlen_chosen), tokenizer.eos_id, dtype=torch.long, device=device)
        batch_rejected_target = torch.full((current_batch_size, max_seqlen_rejected), tokenizer.eos_id, dtype=torch.long, device=device)
        batch_chosen_target[:, :-1] = batch_chosen_input[:, 1:].clone()
        batch_rejected_target[:, :-1] = batch_rejected_input[:, 1:].clone()

        batch_chosen_logits = reference_model(batch_chosen_input)  # [batch_size, seq_len, vocab_size]
        batch_chosen_logprobs = torch.log_softmax(batch_chosen_logits, dim=2)  # [batch_size, seq_len, vocab_size]
        batch_chosen_logprobs = torch.gather(batch_chosen_logprobs, dim=2, index=batch_chosen_target.unsqueeze(2)).squeeze(2).cpu()  # [batch_size, seq_len]

        batch_rejected_logits = reference_model(batch_rejected_input)  # [batch_size, seq_len, vocab_size]
        batch_rejected_logprobs = torch.log_softmax(batch_rejected_logits, dim=2)  # [batch_size, seq_len, vocab_size]
        batch_rejected_logprobs = torch.gather(batch_rejected_logprobs, dim=2, index=batch_rejected_target.unsqueeze(2)).squeeze(2).cpu()  # [batch_size, seq_len]

        # save logprobs from reference model
        for j, (logprobs_chosen, logprobs_rejected) in enumerate(zip(batch_chosen_logprobs.tolist(), batch_rejected_logprobs.tolist())):
            idx = start_idx + j
            item = datasets[idx]
            seqlen_chosen = item['len_prompt'] + item['len_chosen_completion']
            datasets[idx]['chosen_ref_logprobs'] = logprobs_chosen[:seqlen_chosen]  # [seq_len]

            seqlen_rejected = item['len_prompt'] + item['len_rejected_completion']
            datasets[idx]['rejected_ref_logprobs'] = logprobs_rejected[:seqlen_rejected]  # [seq_len]

        pbar.update(1)

    pbar.close()

    return datasets


def process_hh_rlhf_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    reference_model: Transformer,
    batch_size: int = 32,
    num_workers: int = 8,
    validation_ratio: float = 0.05,
    max_seq_length: int = 2048,  # prompt lengths greater than this are discarded
    max_samples: int = 0,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Human preference data',
        'language': 'English',
        'home_page': 'https://github.com/anthropics/hh-rlhf',
    },
) -> None:
    """Process Human preference dataset in .jsonl.gz format and save the tokenized prompt:completion pairs to .pkl format."""

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...')
        return

    # Create the output directory if necessary
    if os.path.exists(output_dir) and overwrite_output:
        logger.info(f'Cleanup output folder {output_dir!r}')
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    if metadata is None:
        metadata = {}

    working_files = find_certain_files_under_dir(src_dir, '.jsonl.gz')

    num_files = len(working_files)

    if num_files == 0:
        logger.warning('Found no .jsonl.gz file')
        return

    if num_files < num_workers:
        num_workers = num_files

    logger.info(f'Processing {num_files} .jsonl.gz files using {num_workers} workers ...')

    process_file_func = functools.partial(
        _process_single_hh_rlhf_jsonl_file,
        max_seq_len=max_seq_length,
        tokenizer=tokenizer,
    )

    with mp.Pool(num_workers) as pool:
        result_list = list(tqdm.tqdm(pool.imap(process_file_func, working_files), total=len(working_files), desc='Processing files'))

    datasets = []
    for result in result_list:
        datasets.extend(result)

    if max_samples > 0 and len(datasets) > max_samples:
        random.shuffle(datasets)
        datasets = datasets[:max_samples]

    datasets = compute_reference_logprobs(datasets, reference_model, batch_size)

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of dictionary object, check the \'data_structure_details\' for details'
    metadata['data_structure_details'] = {
        'chosen_tokens': 'tokenized prompt text + chosen completion text',
        'rejected_tokens': 'tokenized prompt text + rejected completion text',
        'chosen_ref_logprobs': 'the log probabilities for the chosen tokenized prompt text + chosen completion text, computed with the reference model',
        'rejected_ref_logprobs': 'the log probabilities for the chosen tokenized prompt text + rejected completion text, computed with the reference model',
        'len_prompt': 'length of tokenized prompt text',
        'len_chosen_completion': 'length of tokenized chosen completion text',
        'len_rejected_completion': 'length of tokenized rejected completion text',
    }
    metadata['min_responses'] = 2
    metadata['max_responses'] = 2

    logger.info('Saving processed Human preference dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_stackexchange_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    reference_model: Transformer,
    batch_size: int = 32,
    max_responses: int = 6,
    num_workers: int = 8,
    validation_ratio: float = 0.05,
    max_seq_length: int = 2048,  # prompt + completion lengths greater than this are discarded
    max_samples: int = 0,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Stack exchange preferences',
        'language': 'English',
        'home_page': 'https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences',
    },
) -> None:
    """Process Stack exchange preferences dataset and save the tokenized prompt:completion pairs to .pkl format."""

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...')
        return

    # Create the output directory if necessary
    if os.path.exists(output_dir) and overwrite_output:
        logger.info(f'Cleanup output folder {output_dir!r}')
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    if metadata is None:
        metadata = {}

    working_files = find_certain_files_under_dir(src_dir, '.parquet')

    num_files = len(working_files)

    if num_files == 0:
        logger.warning('Found no .parquet file')
        return

    if num_files < num_workers:
        num_workers = num_files

    logger.info(f'Processing {num_files} .parquet files using {num_workers} workers ...')

    process_file_func = functools.partial(
        _process_single_stackexchange_file,
        max_seq_len=max_seq_length,
        max_responses=max_responses,
        tokenizer=tokenizer,
    )

    with mp.Pool(num_workers) as pool:
        result_list = list(tqdm.tqdm(pool.imap(process_file_func, working_files), total=len(working_files), desc='Processing files'))

    datasets = []
    for result in result_list:
        datasets.extend(result)

    if max_samples > 0 and len(datasets) > max_samples:
        random.shuffle(datasets)
        datasets = datasets[:max_samples]

    datasets = compute_reference_logprobs(datasets, reference_model, batch_size)

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of dictionary object, check the \'data_structure_details\' for details'
    metadata['data_structure_details'] = {
        'chosen_tokens': 'tokenized prompt text + chosen completion text',
        'rejected_tokens': 'tokenized prompt text + rejected completion text',
        'chosen_ref_logprobs': 'the log probabilities for the chosen tokenized prompt text + chosen completion text, computed with the reference model',
        'rejected_ref_logprobs': 'the log probabilities for the chosen tokenized prompt text + rejected completion text, computed with the reference model',
        'len_prompt': 'length of tokenized prompt text',
        'len_chosen_completion': 'length of tokenized chosen completion text',
        'len_rejected_completion': 'length of tokenized rejected completion text',
    }
    metadata['min_responses'] = 2
    metadata['max_responses'] = max_responses

    logger.info('Saving processed stack exchange preferences dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def load_reference_model(ckpt_path: str, vocab_size: int = 32000, max_seq_len: int = 2048, max_batch_size: int = 64, compute_dtype: Any = torch.bfloat16) -> Transformer:
    if not os.path.exists(ckpt_path):
        raise ValueError(f'Checkpoint file {ckpt_path!r} does not exist, aborting ...')
    ckpt_dir = os.path.dirname(ckpt_path)

    params_path = os.path.join(ckpt_dir, 'params.json')
    if not os.path.exists(params_path):
        raise ValueError(f'Can not find model metadata file {params_path!r}, aborting ...')

    print(f'Starting to load model checkpoints {ckpt_path!r} ...')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    with open(params_path, 'r') as f:
        params = json.loads(f.read())

    # remove old keys
    for k in ['max_seq_len', 'max_batch_size', 'use_cache', 'vocab_size']:
        try:
            del params[k]
        except Exception:
            continue

    model_args: ModelArgs = ModelArgs(
        **params,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        use_cache=False,
    )

    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)

    for params in model.parameters():
        params.requires_grad = False

    for name, module in model.named_modules():
        module = module.to(dtype=compute_dtype)

    model = model.eval()
    return model


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Set multiprocessing start mode
    mp.set_start_method('spawn')

    tokenizer = Tokenizer(model_path='/home/michael/models/meta_llama2/tokenizer.model')

    reference_model = load_reference_model(
        ckpt_path='./checkpoints/7b-sft/steps-2200-merged.pth',
        vocab_size=tokenizer.vocab_size,
    )

    process_hh_rlhf_dataset(
        src_dir='/home/michael/datasets/hh-rlhf',
        output_dir='./datasets/hh_rlhf_preference',
        tokenizer=tokenizer,
        reference_model=reference_model,
        num_workers=16,
        batch_size=32,
        max_seq_length=512,
        max_samples=100000,
    )

    process_stackexchange_dataset(
        src_dir='/home/michael/datasets/stack_exchange_preferences',
        output_dir='./datasets/stack_exchange_preferences',
        tokenizer=tokenizer,
        reference_model=reference_model,
        num_workers=16,
        batch_size=32,
        max_seq_length=512,
        max_samples=100000,
    )
