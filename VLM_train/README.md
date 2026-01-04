# Training Models with LLaMA-Factory

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

**[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** is a unified and easy-to-use framework for LLM fine-tuning.
We use LLaMA-Factory to fine-tune models on datasets provided by [OpenDataArena](https://opendataarena.github.io).

## Installation
We use version `0.9.4.dev0` of LLaMA-Factory for Supervised Fine-Tuning (SFT):
```bash
git clone https://github.com/OpenDataArena/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 0.9.4.dev0 
pip install -e ".[torch,metrics]"
```

## Data Preparation

We use the Alpaca format for SFT:
```json
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "images": ["/path/to/images.jpg"]
  }
]
```

If you want to use your own dataset, you can update `LLaMA-Factory/data/dataset_info.json` accordingly. For more details, please refer to the [README](https://github.com/OpenDataArena/LLaMA-Factory/tree/main/data#supervised-fine-tuning-dataset).

## Supervised Fine-Tuning
Run full-parameter SFT for [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) using the following command:

```bash
export SEED=42
export DATASET=your_dataset_name

llamafactory-cli train \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --trust_remote_code \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --freeze_vision_tower \
  --freeze_multi_modal_projector \
  --image_max_pixels 262144 \
  --video_max_pixels 16384 \
  --seed ${SEED} \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --dataset ${DATASET} \
  --template qwen3_vl \
  --packing \
  --cutoff_len 12288 \
  --max_samples 10000000 \
  --overwrite_cache \
  --preprocessing_num_workers 64 \
  --dataloader_num_workers 4 \
  --output_dir saves/qwen3vl-8b/full/sft-${DATASET}/seed-${SEED} \
  --save_only_model \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss \
  --overwrite_output_dir \
  --report_to none \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --ddp_timeout 180000000
```
You can also modify `SEED` and `DATASET` in the YAML configuration file, and then run the following command to train the model:
```bash
llamafactory-cli train train_config/qwen3_config.yaml
```

## About
For more detailed usage of LLaMA-Factory, please refer to the [LLaMA-Factory Documentation](https://llamafactory.readthedocs.io/en/latest/).