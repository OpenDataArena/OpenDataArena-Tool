# OpenDataArena Model Evaluation with OpenCompass

**[OpenCompass](https://github.com/OpenCompass/OpenCompass)** is an all-in-one platform designed for the comprehensive evaluation of LLMs, which provides a fair, transparent, and reproducible benchmarking framework.
We use OpenCompass to evaluate the performance of our SFT models on the popular benchmarks.

## Installation

```bash
# Create a virtual environment
conda create --name opencompass python=3.10 -y
conda activate opencompass
# Install OpenCompass and its dependencies from source
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

## Evaluation
To evaluate the performance of the model on the benchmarks, you can pass the model path, model name, and benchmark file (as shown in the table below) to the evaluation script as follows:

### Evaluate on a Single Benchmark
```bash
# evaluate sft model based on Llama-3.1-8B
bash eval_script/test_llama.sh your_model_path dataset_name benchmark_file

# evaluate sft model based on Qwen-2.5-7B
bash eval_script/test_qwen.sh your_model_path dataset_name benchmark_file
```
The `dataset_name` parameter is used as a label to distinguish your test results. Please note that this parameter is for identification purposes only and does not affect the test execution itself.

The `benchmark_file` parameter specifies which benchmark you want to run. For more details on this, please refer to the **Benchmarks** section below.

### Evaluate on Benchmarks in a Specific Domain
```bash
# evaluate sft model based on Llama-3.1-8B in math domain benchmarks
bash eval_script/test_llama_math.sh your_model_path dataset_name

# evaluate sft model based on Qwen-2.5-7B in code domain benchmarks 
bash eval_script/test_llama_code.sh your_model_path dataset_name
```
For more evaluation scripts, please refer to the `eval_script` folder.

### Evaluate on All Benchmarks

```bash
# evaluate sft model based on Llama-3.1-8B
bash eval_script/test_llama_all_benchmarks.sh your_model_path dataset_name

# evaluate sft model based on Qwen-2.5-7B
bash eval_script/test_qwen_all_benchmarks.sh your_model_path dataset_name
```

## Benchmarks

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
| math | AIME 2024 | aime2024_repeat8_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| code | HumanEval | humaneval_gen_8e312c | HumanEvalEvaluator |
| code | HumanEval+ | humaneval_plus_gen_8e312c | HumanEvalPlusEvaluator |
| code | MBPP | sanitized_mbpp_mdblock_gen_a447ff | MBPPEvaluator |
| code | LiveCodeBench | livecodebench_gen | LCBCodeGenerationEvaluator |
| reasoning | ARC-c | ARC_c_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| reasoning | BBH | bbh_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| reasoning | Kor-Bench | korbench_single_0_shot_xver_gen | IAAR-Shanghai/xVerify-9B-C |
| reasoning | CaLM | calm | CaLMEvaluator |
| reasoning | GPQA | gpqa_xver_gen | IAAR-Shanghai/xVerify-9B-C |


## Results Summary
Run the following commands to summarize the evaluation scores of the model.
```bash
# summarize Llama-3.1-8B evaluation scores
python summary_scores/run_summary.py -s opencompass/outputs -d res_llama -m llama-3_1-8b-instruct-vllm
# summarize Qwen-2.5-7B evaluation scores
python summary_scores/run_summary.py -s opencompass/outputs -d res_qwen -m qwen2.5-7b-instruct-vllm
```
`opencompass/outputs`: The directory where the raw test results and logs are saved.

`res_llama` and `res_qwen`: The destination path for the final, summarized result file.

`llama-3_1-8b-instruct-vllm` and `qwen-2.5-7b-instruct-vllm`: The identifier for the model is accelerated by `vllm`. This is a required argument.
