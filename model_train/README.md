# OpenDataArena Model Training with LLaMA-Factory
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune the base models, using datasets listed on [OpenDataArena](https://opendataarena.github.io).

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

```bash
llamafactory-cli train train_config/llama_config.yaml
llamafactory-cli train train_config/qwen_config.yaml
```

For data whose total length (including system, conversation history, instructions, input, and output) exceeds 4096 tokens, we use the long cot training setting as follows:

```bash
llamafactory-cli train train_config/llama_long_config.yaml
llamafactory-cli train train_config/qwen_long_config.yaml
```

## About
For more detailed usage of LLaMA-Factory, please refer to the [LLaMA-Factory documentation](https://llamafactory.readthedocs.io/en/latest/).