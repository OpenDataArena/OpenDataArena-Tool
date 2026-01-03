# Model Evaluation using VLMEvalKit

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

**[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** is a one-stop platform for the comprehensive evaluation of VLMs (Vision-Language Models), providing a fair, transparent, and reproducible benchmarking framework.

We use VLMEvalKit to evaluate the performance of VLM models on popular benchmarks.

## Installation

We use VLMEvalKit version `95568f5` to evaluate VLM model performance.

```bash
# Clone ODA evaluation tool
git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git
cd VLM_eval

# Create virtual environment
conda create -n vlmeval python=3.10 -y
conda activate vlmeval

# Clone and checkout specific version
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
git checkout 8573857
pip install -e .

# Install inference backend vllm
pip install vllm>=0.11.0
cd ..
```

## Quick Start

### Launch vLLM Model Service
Use vLLM to launch an API service compatible with OpenAI interfaces.

```bash
# Set environment variables
export MODEL_PATH="/path/to/your/model"
export MODEL_NAME=$(basename "$MODEL_PATH")

# Start API service
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --trust-remote-code \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --port 11455 \
    --enable-prefix-caching
```

### Generate Evaluation Configuration File
Run the following command in the project directory to generate `config.json`. Evaluation data can be added or removed as appropriate. The default configuration below tests all datasets in `ODA`:

```bash
cat << EOF > config.json
{
    "model": {
        "${MODEL_NAME}": {
            "class": "GPT4V",
            "model": "${MODEL_NAME}",
            "key": "sk-123",
            "api_base": "${API_BASE}",
            "temperature": ${TEMPERATURE},
            "img_detail": "high",
            "retry": 3,
            "timeout": 180000,
            "max_tokens": 30000
        }
    },
    "data": { 
        "CV-Bench-2D": {
            "class": "CVBench",
            "dataset": "CV-Bench-2D"
        },
        "CV-Bench-3D": {
            "class": "CVBench",
            "dataset": "CV-Bench-3D"
        },
        "MathVista_MINI": {
            "class": "MathVista",
            "dataset": "MathVista_MINI"
        },
        "MathVision_MINI": {
            "class": "MathVision",
            "dataset": "MathVision_MINI"
        },
        "MathVerse_MINI": {
            "class": "MathVerse",
            "dataset": "MathVerse_MINI"
        },
        "Dynamath": {
            "class": "Dynamath",
            "dataset": "Dynamath"
        },
        "LogicVista": {
            "class": "LogicVista",
            "dataset": "LogicVista"
        },
        "VisuLogic": {
            "class": "VisuLogic",
            "dataset": "VisuLogic"
        },
        "AI2D_TEST": {
            "class": "ImageMCQDataset",
            "dataset": "AI2D_TEST"
        },
        "ScienceQA_TEST": {
            "class": "ImageMCQDataset",
            "dataset": "ScienceQA_TEST"
        },
        "MMMU_DEV_VAL": {
            "class": "MMMUDataset",
            "dataset": "MMMU_DEV_VAL"
        },
        "SEEDBench2": {
            "class": "ImageMCQDataset",
            "dataset": "SEEDBench2"
        },
        "MMBench_DEV_EN_V11": {
            "class": "ImageMCQDataset",
            "dataset": "MMBench_DEV_EN_V11"
        },
        "RealWorldQA": {
            "class": "ImageMCQDataset",
            "dataset": "RealWorldQA"
        },
        "MMStar": {
            "class": "ImageMCQDataset",
            "dataset": "MMStar"
        },
        "ChartQA_TEST": {
            "class": "ImageVQADataset",
            "dataset": "ChartQA_TEST"
        },
        "OCRBench": {
            "class": "OCRBench",
            "dataset": "OCRBench"
        },
        "InfoVQA_TEST": {
            "class": "ImageVQADataset",
            "dataset": "InfoVQA_TEST"
        },
        "CharXiv_descriptive_val":{
            "class": "CharXiv",
            "dataset": "CharXiv_descriptive_val"
        },
        "CharXiv_reasoning_val": {
            "class": "CharXiv",
            "dataset": "CharXiv_reasoning_val"
        } 
    }
}
EOF
```

### Execute Model Inference

```bash
# Specify dataset storage path
export LMUData="/path/to/your/LMUData"

python VLMEvalKit/run.py \
    --config config.json \
    --work-dir ./eval_results \
    --mode infer \
    --api-nproc 64 \
    --reuse
```

### Results Statistics and Summary (Evaluation)

```bash
# Automatically retrieve the latest timestamp folder in the model directory
export INFER_PATH=$(ls -td ./eval_results/${MODEL_NAME}/T*_G | head -1)
echo "Detected inference path: ${INFER_PATH}"

# 1. Run evaluation to calculate scores
python vlm_eval.py ${INFER_PATH}

# 2. Aggregate accuracy results to Excel (saved to ${INFER_PATH}/results.xlsx)
python merge.py ${INFER_PATH} --model_prefix ${MODEL_NAME}

# 3. Convert to standard JSON report (for official leaderboard submission) Submission URL: [https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=3]
python convert_to_json.py ${INFER_PATH}
```

## About
For more detailed usage of VLMEvalKit, please refer to the [VLMEvalKit Official Repository](https://github.com/open-compass/VLMEvalKit).