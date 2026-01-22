# OpenDataArena Model Training with LLaMA-Factory

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

**[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** is a unified and easy-to-use framework for LLM fine-tuning.
We use LLaMA-Factory to fine-tune the base models, using datasets listed on [OpenDataArena](https://opendataarena.github.io).

## Installation
We use version `v0.9.4` of LLaMA-Factory to conduct supervised fine-tuning (SFT):
```
git clone -b v0.9.4 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
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
Use the following commands to run full parameter SFT of [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [Qwen-2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) and [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base), respectively.
Take `Llama-3.1-8B` base model as an example:

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
llamafactory-cli train train_config/qwen3_config.yaml
```

### Long CoT SFT
For data whose total length (including system, conversation history, instructions, input, and output) exceeds 4096 tokens, we use the long CoT (Chain-of-Thought) setting as follows:

```bash
llamafactory-cli train train_config/llama_long_config.yaml
llamafactory-cli train train_config/qwen_long_config.yaml
llamafactory-cli train train_config/qwen3_config.yaml
```

The difference from the setting above is that we adapt the `cutoff_len`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, and `packing`.

The datasets in the following table are trained using above long CoT setting.

| Dataset | Affiliation | HF Link |
|---|---|---|
| OpenR1-Math-220k | Huggingface | [Link](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) |
| QwQ-LongCoT-130K | Guijin Son | [Link](https://huggingface.co/datasets/amphora/QwQ-LongCoT-130K) |
| QwQ-LongCoT-130K-2 | Guijin Son | [Link](https://huggingface.co/datasets/amphora/QwQ-LongCoT-130K-2) |
| Bespoke-Stratos-17k | Bespoke Labs | [Link](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |
| OpenThoughts-114k | Stanford University | [Link](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| R1-Distill-SFT | ServiceNow-AI | [Link](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) |
| OpenO1-SFT | GAIR | [Link](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT) |
| DeepMath-103K | Tencent & SJTU | [Link](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |
| Raiden-DeepSeek-R1 | sequelbox | [Link](https://huggingface.co/datasets/sequelbox/Raiden-DeepSeek-R1) |
| Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B | Allen AI | [Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B) |
| Magpie-Reasoning-V2-250K-CoT-QwQ | Allen AI | [Link](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-QwQ) |
| AM-Thinking-v1-Distilled-code | a-m-team | [Link](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled/blob/main/code.jsonl) |
| AM-Thinking-v1-Distilled-math | a-m-team | [Link](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled/blob/main/math.jsonl) |
| Light-R1-SFTData | Qiyuan Tech | [Link](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) |
| Fast-Math-R1-SFT | University of Tokyo | [Link](https://huggingface.co/datasets/RabotniKuma/Fast-Math-R1-SFT) |
| OpenThoughts3-1.2M | Stanford University | [Link](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) |
| OpenMathReasoning-cot | Nvidia | [Link](https://huggingface.co/datasets/nvidia/OpenMathReasoning) |


## About
For more detailed usage of LLaMA-Factory, please refer to the [LLaMA-Factory documentation](https://llamafactory.readthedocs.io/en/latest/).