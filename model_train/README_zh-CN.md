# 使用 LLaMA-Factory 训练模型

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

**[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** 是一个统一且易于使用的框架，用于 LLM 微调。
我们使用 LLaMA-Factory，在 [OpenDataArena](https://opendataarena.github.io) 提供的数据集上进行微调。

## 安装
我们使用 LLaMA-Factory 的 `v0.9.4` 版本进行监督微调 (SFT):
```
git clone -b v0.9.4 https://github.com/hiyouga/LlamaFactory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## 数据准备

我们使用 Alpaca 格式进行 SFT:
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
您也可以参考 `LLaMA-Factory/data/alpaca_en_demo.json` 的示例。

如果您想使用自己的数据集，您可以相应地更新 `LLaMA-Factory/data/dataset_info.json`。更多详细信息，请参阅 [README](https://github.com/OpenDataArena/LLaMA-Factory/tree/main/data#supervised-fine-tuning-dataset)。

## 监督微调
使用以下命令分别运行 [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [Qwen-2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) 和 [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) 的全参数 SFT。
以 `Llama-3.1-8B` 基础模型为例:

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
您也可以修改 YAML 配置文件中的 `SEED` 和 `DATASET`，然后运行以下命令来训练模型:
```bash
llamafactory-cli train train_config/llama_config.yaml
llamafactory-cli train train_config/qwen_config.yaml
llamafactory-cli train train_config/qwen3_config.yaml
```

### 长 CoT SFT
对于总长度（包括系统、对话历史、指令、输入和输出）超过 4096 个 token 的数据，我们使用长 CoT（Chain-of-Thought）SFT，设置如下:

```bash
llamafactory-cli train train_config/llama_long_config.yaml
llamafactory-cli train train_config/qwen_long_config.yaml
llamafactory-cli train train_config/qwen3_config.yaml
```

与上述 SFT 设置的不同之处在于，我们调整了 `cutoff_len`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`learning_rate` 和 `packing`。

以下数据集使用上述长 CoT 设置进行训练。

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


## 关于
更多关于 LLaMA-Factory 的详细使用方法，请参阅 [LLaMA-Factory 文档](https://llamafactory.readthedocs.io/en/latest/)。