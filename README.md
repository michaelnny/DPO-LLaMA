# DPO-LLaMA

A clean implementation of direct preference optimization (DPO) to train the LLaMA 2 model to align with human preferences. This project supports 4-bit QLoRA.

# Main Difference compared to the original DPO code

The code for DPO published by author Eric can be found here: [DPO: Direct Preference Optimization](https://github.com/eric-mitchell/direct-preference-optimization)

Our work mainly differs in these aspects:

- Simplify code by separating supervised fine-tuning (SFT) and direct preference optimization (DPO) into different modules, including training scripts and datasets.
- Use a custom yet very simple LLaMA model and QLoRA implementation, decoupled from Hugging Face tools.
- Pre-compute the log probabilities for preference datasets, so during DPO training, we only need to train the main LLM model. This saves a lot of compute resources and makes training a 7B model on a single RTX 3090 GPU with 24GB VRAM possible (with QLoRA).
- Due to limited support between PyTorch FSDP and QLoRA (Bitsandbytes library), the training scripts in this project only support a single GPU.

# Disclaimer

**Project Purpose:** This project is for research and education only, focusing on the study of individual algorithms rather than creating a standard library. If you're looking for a ready-to-use library for production applications, this project may not be suitable for your needs.

**Bug Reporting and Contributions:** Testing has been conducted in different runs, but we cannot guarantee it's bug-free. Bug reports and pull requests are highly encouraged and welcomed.

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.1.2
- Tensorboard 2.13.0
- Bitsandbytes 0.41.3

# Code Structure

- `dpo_llama` directory contains main source code for the project.
  - `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
  - `utils` directory contains helper modules like custom datasets, logging, checkpoint etc.
  - `models` contains the LLaMA model class, including LoRA module and 4-bit quantized layers.
  - `run_sft_lora.py` run supervised fine-tuning starting from Meta's pre-trained model, supports 4-bit QLoRA parameter efficient fine-tuning method (only supports single GPU).
  - `run_dpo_lora.py` train policy model with DPO, starting from supervised fine-tuning model, supports 4-bit QLoRA parameter efficient fine-tuning method (only supports single GPU).
- `scripts` directory contains all source code for convert the model weights and build datasets.
  - `build_finetune_datasets.py` build fine-tuning datasets (save the dataset to .pkl files), which is used for supervised fine-tuning training stage.
  - `build_preference_datasets.py` build preference comparison datasets (save the dataset to .pkl files), which is used for DPO training stage.
  - `convert_meta_checkpoint.py` convert Meta's pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.
  - `convert_lora_checkpoint.py` convert fine-tunned LoRA weights to a full state_dict checkpoint.
- `logs` directory contains training logs for the different runs.

# Project Setup

```
python -m pip install --upgrade pip setuptools

python -m pip install -r requirements.txt
```

# Project Overview

## Preparation

Here are the steps required to utilize the project:

1. **Download the pre-trained model weights** please refer to https://github.com/facebookresearch/llama on how to download it.
2. **Convert Meta's pre-trained model weights** so it's compatible with our naming convention. Remember to change the file path before running it.

```
python scripts/convert_meta_checkpoint.py
```

3. **Download and build training datasets** For each training stage, we need to use different datasets, we're already prepared some of the common datasets. However, some datasets are too big to upload, so you may need to re-build it. Here's an example of build fine-tuning datasets, remember to change the file path before running it.

```
python scripts/build_finetune_datasets.py
```

Note, we pre-compute the log probabilities from the reference model when building the preference dataset. This can save compute resource and it also makes training with DPO possible on a single GPU, since we only need to run the main LLM model.

To save compute resource, we limit number of samples in the `build_finetune_datasets.py`, you may want to revert that before you run it. Just for reference, it took ~6 hours to pre-compute these log probabilities for the 100,000 samples (maximum length of 512) on a single RTX 3090 GPU.

## Training Stages

1. Run the `run_sft_lora.py` script to use supervised fine-tuning and QLoRA to train the model, this requires a pre-trained model and fine-tune datasets. Check and maintain the configuration inside `dpo_llama/configs/sft_lora.py` if necessary.
2. Run the `run_dpo_lora.py` script to use DPO and QLoRA to train the model, this requires a fine-tuned model and the preference datasets. Check and maintain the configuration inside `dpo_llama/configs/dpo_lora.py` if necessary.

## 4-bit QLoRA

### LoRA parameters

We use a slightly modified LoRALayer class, where we set the scaling directly instead of using an alpha parameter, we found this more consistent and easy to maintain. Since in most case, using a scaling of 1 makes more sense.

```
lora_r: int = 64
lora_scaling: float = 1.0  # set the LoRA scaling, by default 1.0 no scaling
lora_dropout: float = 0.0
```

### LoRA Trainable layers

We can specify which layers in the model should be trainable with LoRA using options like the ones below.

```
lora_attn_query: bool = True  # train Attention query layer
lora_attn_key: bool = False  # train Attention key layer
lora_attn_value: bool = True  # train Attention value layer
lora_attn_proj: bool = False  # train Attention projection layer
lora_attn_mlp: bool = False  # train Attention MLP block
lora_lm_head: bool = False  # train model output head
```

### 4-bit Quantization

We can apply quantization to the frozen linear layers, or both the frozen linear layers and trainable LoRA layers.

When quantizing a LoRA layer, only the pre-trained weights are quantized, while the LoRA parameters remain unchanged.

It's important to mention that quantization is limited to 4-bit quantization, and we utilize `Bitsandbytes` to do so.

```
quant_4bit: bool = False  # quantize frozen linear layer
quant_lora_4bit: bool = False  # quantize LoRA linear layer
quant_4bit_double: bool = False  # double quantize
quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'
```

# Stage 1 - Supervised Fine-Tuning (SFT)

This stage is when we turn a pre-trained language model from predicting next token to answer general questions, in a chat formation.

Once we have a pre-trained model and the fine-tuning datasets are ready, we can start doing supervised fine-tuning using QLoRA.

```
python dpo_llama/run_sft_lora.py
```

Keep in mind we need to merge the LoRA weights after the training, remember to update the file path in the script accordingly.

```
python scripts/convert_lora_checkpoint.py
```

# Stage 2 - Direct Preference Optimization (DPO)

After the fine-tuned phase is done, and the preference comparison datasets are ready, we can start training the model using DPO and QLoRA.

Unlike the original DPO repo from the paper, where it runs two models (the main LLM model, and a fixed reference model). We only need to run the main LLM model, because we've already pre-computed the log probabilities when we build these preference datasets. This saves lots of GPU compute during training, this also makes the training of the 7B model on a single GPU possible (with QLoRA and gradient checkpoint).

This training phase demands more GPU RAM compared to fine-tuning, as it involves maintaining multiple computation graphs/gradients during DPO loss computation.

```
python dpo_llama/run_dpo_lora.py
```

Keep in mind we need to merge the LoRA weights after the training, remember to update the file path in the script accordingly.

```
python scripts/convert_lora_checkpoint.py
```

# Monitoring with tensorboard

We can monitoring the training progress by using Tensorboard:

```
tensorboard --logdir=./logs
```

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

The LLaMA2 model weights are licensed for both researchers and commercial entities. For details, visit: https://github.com/facebookresearch/llama#license.

# Acknowledgments

This project is greatly influenced by the following projects:

- [Llama 2] (https://github.com/facebookresearch/llama)
- [DPO: Direct Preference Optimization] (https://github.com/eric-mitchell/direct-preference-optimization)
- [LoRA] (https://github.com/microsoft/LoRA)
- [QLoRA-LLM] (https://github.com/michaelnny/QLoRA-LLM)
- [InstructLLaMA] (https://github.com/michaelnny/InstructLLaMA)
