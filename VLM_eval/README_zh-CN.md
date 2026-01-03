# 使用 VLMEvalKit 评估模型

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

**[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** 是一个一站式平台，用于全面评估 VLM，提供公平、透明和可重复的基准测试框架。
我们使用 VLMEvalKit 评估 VLM 模型在流行基准上的性能。



## 安装
我们使用 VLMEvalKit 的版本 `95568f5` 来评估 VLM 模型的性能。

```bash
#克隆ODA评测工具
 git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git
 cd VLM_eval

# 创建虚拟环境
conda create -n vlmeval python=3.10 -y
conda activate vlmeval

# 克隆并切换到指定版本
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
git checkout 95568f5
pip install -e .

#安装推理后端 vLLM
pip install vllm>=0.11.0
cd ..
```

## 快速运行
### 启动 vLLM 模型服务
使用 vLLM 启动兼容 OpenAI 接口的 API 服务。
```bash
# 设置环境变量
export MODEL_PATH="/path/to/your/model"
export MODEL_NAME=$(basename "$MODEL_PATH")

# 启动 API 服务
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

### 生成评测配置文件
在项目目录运行以下命令，生成 `config.json`，评测数据可作适当增减，以下默认配置为`ODA`中测试全部数据集:
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

### 执行模型推理 (Inference)
```bash
# 指定数据集存放路径
export LMUData="/path/to/your/LMUData"

python VLMEvalKit/run.py \
    --config config.json \
    --work-dir ./eval_results \
    --mode infer \
    --api-nproc 64 \
    --reuse
```

### 结果统计与汇总 (Evaluation)
```bash
# 自动获取该模型目录下最新的时间戳文件夹
export INFER_PATH=$(ls -td ./eval_results/${MODEL_NAME}/T*_G | head -1)
echo "检测到推理路径: ${INFER_PATH}"

# 1. 运行评测计算得分
python vlm_eval.py ${INFER_PATH}

# 2. 汇总准确率结果至 Excel (保存至 ${INFER_PATH}/results.xlsx)
python merge.py ${INFER_PATH} --model_prefix ${MODEL_NAME}

# 3. 转换为标准 JSON 报告 (用于提交官方榜单) 提交url: https://rrc.cvc.uab.es/?ch=17&com=mymethods&task=3
python convert_to_json.py ${INFER_PATH}
```
## 关于
更多关于 VLMEvalKit 的详细使用方法，请参阅 [VLMEvalKit 官方代码库](https://github.com/open-compass/VLMEvalKit)。