# 使用 OpenCompass 评估模型

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

**[OpenCompass](https://github.com/OpenCompass/OpenCompass)** 是一个一站式平台，用于全面评估 LLM，提供公平、透明和可重复的基准测试框架。
我们使用 OpenCompass 评估 SFT 模型在流行基准上的性能。

## 安装
我们使用 OpenCompass 的版本 `v0.4.2` 来评估 SFT 模型的性能。

```bash
# 创建虚拟环境
conda create --name opencompass python=3.10 -y
conda activate opencompass
# 从源码安装 OpenCompass 及其依赖
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge pyarrow
conda install -c conda-forge pyzmq
git clone --recurse-submodules https://github.com/OpenDataArena/OpenDataArena-Tool.git
cd model_eval/opencompass
pip install -e .
cd ..
git clone https://github.com/openai/human-eval
pip install -e human-eval
cd vllm
pip install -e .
cd ../evalplus
pip install -e . 
pip install lmdeploy
pip install tree-sitter==0.21.3
```

## 评估
要评估模型在基准上的性能，请运行以下命令。
您需要将模型路径、模型名称和基准文件（如表所示）传递给评估脚本。

* `your_model_path`: 模型检查点的路径。
* `dataset_name`: 数据集的名称。它用于区分您的评估结果，仅用于标识目的，不影响评估本身。
* `benchmark_file`: 基准文件的名称。更多详细信息，请参阅 **基准** 部分。

### 在单个基准上评估
```bash
# 基于 Llama-3.1-8B 评估 SFT 模型
bash eval_script/test_llama.sh your_model_path dataset_name benchmark_file

# 基于 Qwen-2.5-7B 评估 SFT 模型
bash eval_script/test_qwen.sh your_model_path dataset_name benchmark_file

# 基于 Qwen-3-8B 评估 SFT 模型
bash eval_script/test_qwen3.sh your_model_path dataset_name benchmark_file
```

### 在特定领域基准上评估
```bash
# 基于 Llama-3.1-8B 在数学领域基准上评估 SFT 模型
bash eval_script/test_llama_math.sh your_model_path dataset_name

# 基于 Qwen-2.5-7B 在代码领域基准上评估 SFT 模型
bash eval_script/test_qwen_code.sh your_model_path dataset_name

# 基于 Qwen-3-8B 在代码领域基准上评估 SFT 模型
bash eval_script/test_qwen3_code.sh your_model_path dataset_name
```

### 在所有基准上评估

```bash
# 基于 Llama-3.1-8B 评估 SFT 模型
bash eval_script/test_llama_all_benchmarks.sh your_model_path dataset_name

# 基于 Qwen-2.5-7B 评估 SFT 模型
bash eval_script/test_qwen_all_benchmarks.sh your_model_path dataset_name

# 基于 Qwen-3-8B 评估 SFT 模型
bash eval_script/test_qwen3_all_benchmarks.sh your_model_path dataset_name
```

更多评估脚本，请参阅 `eval_script` 文件夹。

## 基准
我们评估 SFT 模型在以下基准上的性能。

| Domain  | Benchmark  | Benchmark File | Evaluator |
| :--- | :--- | :--- | :--- |
| general | DROP | drop_gen_a2697c | IAAR-Shanghai/xVerify-9B-C |
| general | IFEval | IFEval_gen | IFEvaluator |
| general | AGIEval | agieval_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| general | MMLU-Pro | mmlu_pro_few_shot_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| math | OmniMath | omni_math_gen | KbsdJames/Omni-Judge |
| math | OlympiadBenchMath | OlympiadBenchMath_0shot_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| math | GSM8K | gsm8k_0shot_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| math | MATH | math_0shot_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| math | MATH-500 | math_prm800k_500_0shot_cot_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| math | AIME_2024 | aime2024_repeat8_cver_gen | CompassVerifier-7B |
| math | AIME_2025 | aime2025_repeat8_cver_gen | CompassVerifier-7B |
| math | HMMT_Feb_2025 | hmmt2025_repeat8_cver_gen | CompassVerifier-7B |
| math | CMIMC_2025 | cmimc2025_repeat8_cver_gen | CompassVerifier-7B |
| math | BRUMO_2025 | brumo2025_repeat8_cver_gen | CompassVerifier-7B |
| code | HumanEval | humaneval_gen_8e312c | HumanEvalEvaluator |
| code | HumanEval+ | humaneval_plus_gen_8e312c | HumanEvalPlusEvaluator |
| code | MBPP | sanitized_mbpp_mdblock_gen_a447ff | MBPPEvaluator |
| code | LiveCodeBench | livecodebench_gen | LCBCodeGenerationEvaluator |
| reasoning | ARC-c | ARC_c_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| reasoning | BBH | bbh_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| reasoning | Kor-Bench | korbench_single_0_shot_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| reasoning | CaLM | calm | CaLMEvaluator |
| reasoning | GPQA | gpqa_xver_gen | IAAR-Shanghai/xVerify-9B-C |


## 结果总结
运行以下命令来总结模型的评估分数。

```bash
# 总结 Llama-3.1-8B 评估分数
python summary_scores/run_summary.py -s opencompass/outputs -d res_llama -m llama-3_1-8b-instruct-vllm
# 总结 Qwen-2.5-7B 评估分数
python summary_scores/run_summary.py -s opencompass/outputs -d res_qwen -m qwen2.5-7b-instruct-vllm
# 总结 Qwen-3-8B 评估分数
python summary_scores/run_summary.py -s opencompass/outputs -d res_qwen3 -m qwen3-8b-instruct-vllm
```

* `opencompass/outputs`: 原始测试结果和日志保存的目录。

* `res_llama` , `res_qwen` 和 `res_qwen3`: 最终汇总结果文件的保存路径。

* `llama-3_1-8b-instruct-vllm` , `qwen-2.5-7b-instruct-vllm`和 `qwen3-8b-instruct-vllm`: 使用 `vllm` 加速的模型标识符。这是一个必需的参数。

## 关于
更多关于 OpenCompass 的详细使用方法，请参阅 [OpenCompass 文档](https://opencompass.readthedocs.io/en/latest/)。
