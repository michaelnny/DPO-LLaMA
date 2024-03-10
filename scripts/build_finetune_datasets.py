# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
Module for build instruct fine-tuning datasets.
"""

from typing import Tuple, List, Mapping, Text, Any
import functools
import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import json
import random
import pickle
import pyarrow.parquet as pq
import re
import torch

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from dpo_llama.models.tokenizer import Tokenizer
from dpo_llama.utils.logger import create_logger
from dpo_llama.utils.file_helper import find_certain_files_under_dir, read_zipped_jsonl_file, read_json_file, read_jsonl_file, count_words
from dpo_llama.utils.prompt_builder import build_prompt_completion, Dialog

logger = create_logger()

Metadata = Mapping[Text, Text]

DEFAULT_SYSTEM_PROMPT = {
    'role': 'system',
    'content': '',
}

# this will be inserted into the training data as the first system prompt
DEFAULT_DIALOG = [DEFAULT_SYSTEM_PROMPT]

Answers = List[Mapping[Text, Any]]

# ----------------------------------- helper functions -----------------------------------


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


def _process_single_dm_math_txt_file(txt_file: str, tokenizer: Tokenizer) -> List[dict]:
    pairs = []

    with open(str(txt_file), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')[:-1]

    for i in range(0, len(lines), 2):
        prompt = lines[i].strip()
        completion = lines[i + 1].strip()

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        pairs.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    return pairs


def _raw_texts_to_dialog(dialog_texts: List[str]) -> Dialog:
    """Converts a list of raw text into dialog formation.
    Note it requires the texts follows the correct user/assistant/user/assistant ... order."""

    # requires at least one turn from each role (user, assistant)
    if len(dialog_texts) < 2:
        return None

    # try to trim the last one so we always get a pair of contents from user and assistant
    if len(dialog_texts) % 2 != 0:
        dialog_texts = dialog_texts[:-1]

    assert len(dialog_texts) % 2 == 0, f'dialog length: {len(dialog_texts)}'

    dialog = DEFAULT_DIALOG + [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': raw_text.strip()} for i, raw_text in enumerate(dialog_texts)]

    return dialog


# ----------------------------------- high quality datasets -----------------------------------


def process_dolly_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    max_seq_length: int = 2048,  # prompt + completion lengths greater than this are discarded
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Dolly',
        'language': 'English',
        'home_page': 'https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm',
    },
) -> None:
    """Process dolly dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...')
        return

    if metadata is None:
        metadata = {}

    json_objs = read_jsonl_file(src_file)

    if json_objs is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info('Processing dolly dataset ...')
    datasets = []

    for item in json_objs:
        context = item['context'].strip()
        prompt = item['instruction'].strip()
        completion = item['response'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        if len(prompt_tokens) + len(completion_tokens) > max_seq_length:
            continue

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed dolly dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_alpaca_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Alpaca_cleaned',
        'language': 'English',
        'home_page': 'https://github.com/gururise/AlpacaDataCleaned',
    },
) -> None:
    """Process alpaca dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...')
        return

    if metadata is None:
        metadata = {}

    json_objs = read_json_file(src_file)

    if json_objs is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info('Processing alpaca dataset ...')
    datasets = []

    for item in json_objs:
        context = item['input'].strip()
        prompt = item['instruction'].strip()
        completion = item['output'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed alpaca dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def _convert_to_llama_chat_format(raw_text) -> Dialog:
    dialog = []
    conversations = raw_text.split('\n\nHuman: ')[1:]

    for pair in conversations:
        # standardize some punctuation
        pair = pair.replace(',  ', ', ').replace('.  ', '. ').replace('?  ', '? ').replace('!  ', '! ')
        contents = pair.split('\n\nAssistant: ')
        # skip some bad samples
        if len(contents) != 2 or any(['Assistant:' in t or 'Human:' in t for t in contents]):
            return dialog

        dialog.append({'role': 'user', 'content': contents[0]})
        dialog.append({'role': 'assistant', 'content': contents[1]})

    return dialog


def _sort_answers_by_score_desc(answers: Answers) -> Answers:
    out = sorted(answers, key=lambda d: d['pm_score'], reverse=True)
    return out


def _process_single_stackexchange_file(
    file_path: str,
    tokenizer: Tokenizer,
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

        if len(answers) < 1:
            continue

        answers = _sort_answers_by_score_desc(answers)

        chosen_answer = answers[0]['text'].strip()
        # build prompt tokens once
        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': chosen_answer},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None
        samples.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    return samples


def _process_single_hh_rlhf_jsonl_file(
    file_path: str,
    tokenizer: Tokenizer,
) -> List[Tuple[int]]:
    """
    Read one single .jsonl.gz file and go over each row to build the dataset sample
    """

    samples = []

    for row in read_zipped_jsonl_file(file_path):
        chosen_dialog = _convert_to_llama_chat_format(row['chosen'])

        if len(chosen_dialog) == 0:
            continue

        chosen_dialog = DEFAULT_DIALOG + chosen_dialog
        prompt_tokens, completion_tokens = build_prompt_completion(chosen_dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None
        samples.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    return samples


def process_hh_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    num_workers: int = 8,
    validation_ratio: float = 0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Human preference data',
        'language': 'English',
        'home_page': 'https://github.com/anthropics/hh-rlhf',
    },
) -> None:
    """Process Human preference dataset and save the tokenized prompt:completion pairs to .pkl format.

    Note, we only use the 'chosen' completion as part of the target for supervised fine-tuning.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

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
    # only include training samples from hh-rlhf 'helpful'
    working_files = [f for f in working_files if 'helpful' in f]
    train_files = [f for f in working_files if 'train' in f]
    val_files = [f for f in working_files if 'test' in f]

    def _build_dataset_from_files(files, num_workers):
        num_files = len(files)

        if num_files == 0:
            logger.warning('Found no .jsonl.gz file')
            return

        if num_files < num_workers:
            num_workers = num_files

        logger.info(f'Processing {num_files} .jsonl.gz files using {num_workers} workers ...')

        process_file_func = functools.partial(
            _process_single_hh_rlhf_jsonl_file,
            tokenizer=tokenizer,
        )

        with mp.Pool(num_workers) as pool:
            result_list = list(tqdm.tqdm(pool.imap(process_file_func, files), total=len(files), desc='Processing files'))

        data = []
        for result in result_list:
            data.extend(result)

        return data

    train_ds = _build_dataset_from_files(train_files, num_workers)
    val_ds = _build_dataset_from_files(val_files, num_workers)

    logger.info('Saving processed Human preference dataset ...')

    for data, out_file in zip((train_ds, val_ds), (train_output_file, val_output_file)):
        if len(data) > 0:
            logger.info(f'Saving {len(data)} processed data to {out_file!r} ...')
            pickle.dump(data, open(out_file, 'wb'))

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'
    metadata['num_train_samples'] = len(train_ds)
    metadata['num_validation_samples'] = len(val_ds)

    logger.info(f'Saving metadata to {meta_output_file!r} ...')

    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def process_stackexchange_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    num_workers: int = 8,
    validation_ratio: float = 0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Stack exchange preferences',
        'language': 'English',
        'home_page': 'https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences',
    },
) -> None:
    """Process Stack exchange preference dataset and save the tokenized prompt:completion pairs to .pkl format.

    Note, we only use the first 'chosen' completion as part of the target for supervised fine-tuning.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """
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
        tokenizer=tokenizer,
    )

    with mp.Pool(num_workers) as pool:
        result_list = list(tqdm.tqdm(pool.imap(process_file_func, working_files), total=len(working_files), desc='Processing files'))

    datasets = []
    for result in result_list:
        datasets.extend(result)

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed stack exchange preferences dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    # Set multiprocessing start mode
    mp.set_start_method('spawn')

    tokenizer = Tokenizer(model_path='/home/michael/models/meta_llama2/tokenizer.model')

    process_dolly_dataset(
        src_file='/home/michael/datasets/dolly_15k/databricks-dolly-15k.jsonl',
        output_dir='./datasets/dolly',
        tokenizer=tokenizer,
    )

    process_alpaca_dataset(
        src_file='/home/michael/datasets/alpaca_dataset/alpaca_cleaned.json',
        output_dir='./datasets/alpaca',
        tokenizer=tokenizer,
    )

    process_hh_dataset(
        src_dir='/home/michael/datasets/hh-rlhf/helpful-base',
        output_dir='./datasets/hh_rlhf_finetune',
        tokenizer=tokenizer,
        num_workers=16,
    )

    process_stackexchange_dataset(
        src_dir='/home/michael/datasets/stack_exchange_preferences',
        output_dir='./datasets/stack_exchange_finetune',
        tokenizer=tokenizer,
        num_workers=16,
    )
