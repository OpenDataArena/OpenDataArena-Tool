# 使用 LLaMA-Factory 训练模型

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

**[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** 是一个统一且易于使用的框架，用于 LLM 微调。
我们使用 LLaMA-Factory，在 [OpenDataArena](https://opendataarena.github.io) 提供的数据集上进行微调。

## 安装
我们使用 LLaMA-Factory 的 `0.9.4.dev0` 版本进行监督微调 (SFT):
```
git clone https://github.com/OpenDataArena/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 0.9.4.dev0 
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
    "images": ["/path/to/images.jpg"]
  }
]
```

如果您想使用自己的数据集，您可以相应地更新 `LLaMA-Factory/data/dataset_info.json`。更多详细信息，请参阅 [README](https://github.com/OpenDataArena/LLaMA-Factory/tree/main/data#supervised-fine-tuning-dataset)。

## 监督微调
使用以下命令运行 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 的全参数 SFT：

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
您也可以修改 YAML 配置文件中的 `SEED` 和 `DATASET`，然后运行以下命令来训练模型:
```bash
llamafactory-cli train train_config/qwen3_config.yaml
```



## 关于
更多关于 LLaMA-Factory 的详细使用方法，请参阅 [LLaMA-Factory 文档](https://llamafactory.readthedocs.io/en/latest/)。