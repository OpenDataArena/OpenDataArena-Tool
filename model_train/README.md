# OpenDataArena Model Training with LLaMA-Factory
**[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** is a unified and easy-to-use framework for LLM fine-tuning.
We use LLaMA-Factory to fine-tune the base models, using datasets listed on [OpenDataArena](https://opendataarena.github.io).

## Installation
We use version `v0.9.2` of LLaMA-Factory to conduct supervised fine-tuning (SFT):
```
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Data Preparation

We use the Alpaca format for SFT:
```
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["human instruction in the first round (optional)", "model response in the first round (optional)"],
      ["human instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```
You can also refer to `LLaMA-Factory/data/alpaca_en_demo.json` for an example.

If you want to use your own dataset, you can update `LLaMA-Factory/data/dataset_info.json` accordingly. For more details, please refer to the [README](https://github.com/OpenDataArena/LLaMA-Factory/tree/main/data#supervised-fine-tuning-dataset).

## Supervised Fine-tuning
Use the following commands to run full parameter SFT of [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) and [Qwen-2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), respectively.
Take `Llama-3.1-8B` as an example:
```bash
export SEED=42
export DATASET=your_dataset_name
llamafactory-cli train \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --trust_remote_code \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --seed ${SEED} \
  --deepspeed examples/deepspeed/ds_z2_config.json \
  --dataset ${DATASET} \
  --template default \
  --cutoff_len 4096 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --output_dir saves/llama3.1-8b/full/sft-${DATASET}/seed-${SEED} \
  --save_only_model \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --plot_loss \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2.0e-5 \
  --use_liger_kernel \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --bf16 \
  --ddp_timeout 180000000
```
You can also modify the `SEED` and `DATASET` in the YAML config file, then run the following commands to train the model:
```bash
llamafactory-cli train train_config/llama_config.yaml
llamafactory-cli train train_config/qwen_config.yaml
```

For data whose total length (including system, conversation history, instructions, input, and output) exceeds 4096 tokens, we use the long cot training setting as follows:

```bash
llamafactory-cli train train_config/llama_long_config.yaml
llamafactory-cli train train_config/qwen_long_config.yaml
```
where we adapt the `cutoff_len`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, and `packing`.

## About
For more detailed usage of LLaMA-Factory, please refer to the [LLaMA-Factory documentation](https://llamafactory.readthedocs.io/en/latest/).