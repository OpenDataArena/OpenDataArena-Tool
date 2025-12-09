# Model-based Data Evaluation Framework

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

This framework provides a complete data quality evaluation system that performs multi-dimensional assessment of datasets using deep learning models.

## ✨ Core Features

- **🚀 Automatic Data Parallelism**: Automatically calculates data parallelism based on global GPU count and per-scorer GPU requirements
- **🎯 Intelligent GPU Allocation**: Supports multi-GPU tasks with automatic dedicated GPU resource allocation for each parallel task
- **📊 Dual Scoring Modes**: Distinguishes between **Pointwise** (per-sample scoring) and **Setwise** (whole-dataset scoring) modes
- **🔧 Configuration-Driven**: Easily manage models, evaluation metrics, and runtime parameters through YAML configuration files
- **💾 Structured Output**: Automatically merges and categorizes scoring results with support for viewing intermediate results
- **🔄 Parallel Execution**: Multiple scorers execute sequentially, with each scorer using data parallelism for acceleration

## 📦 Supported Scorers

This framework integrates nearly 40 model-based scorers covering quality, complexity, gradient analysis, and more:

### 🎯 Quality

Evaluate data quality, accuracy, readability, and related dimensions:

- **SkyworkRewardScorer**: Skywork reward model scoring
- **AtheneScorer**: Athene reward model scoring  
- **RMDeBERTaScorer**: DeBERTa reward model scoring
- **Gpt2HarmlessScorer**: GPT-2 harmlessness reward model
- **Gpt2HelpfulScorer**: GPT-2 helpfulness reward model
- **InfOrmScorer**: INF-ORM reward model scoring
- **DeitaQScorer**: Deita quality scorer
- **DebertaScorer**: DeBERTa quality classifier
- **FinewebEduScorer**: FineWeb educational quality classifier
- **TextbookScorer**: Textbook quality classifier
- **QuRateScorer**: QuRater multi-dimensional quality scoring
- **CleanlinessScorer**: Data cleanliness scoring
- **ProfessionalismScorer**: Professionalism scoring
- **ReadabilityScorer**: Readability scoring
- **ReasoningScorer**: Reasoning capability scoring
- **UniEvalD2tScorer**: Data-to-Text quality evaluation
- **UniEvalDialogScorer**: Dialog quality evaluation
- **UniEvalFactScorer**: Factual accuracy evaluation
- **UniEvalSumScorer**: Summarization quality evaluation

### 🧠 Complexity

Evaluate data difficulty, complexity, perplexity, and related dimensions:

- **DeitaCScorer**: Deita complexity scoring
- **IFDScorer**: Instruction Following Difficulty scoring
- **ThinkingProbScorer**: Thinking probability scoring
- **PPLScorer**: Perplexity scoring
- **NormLossScorer**: Normalized loss scoring
- **UPDScorer**: Uncertainty and Predictive Difficulty scoring

### 🔧 Others

Including gradient analysis, data selection, specific tasks, etc.:

- **GraNdScorer**: Gradient norm difference scoring
- **NuclearNormScorer**: Nuclear norm scoring
- **EffectiveRankScorer**: Effective rank scoring
- **Task2VecScorer**: Task2Vec embedding scoring
- **MIWVScorer**: Maximum Influence Weighted Value scoring
- **SelectitTokenScorer**: SelectIT token-level scoring
- **SelectitSentenceScorer**: SelectIT sentence-level scoring
- **SelectitModelScorer**: SelectIT model ensemble scoring
- **HESScorer**: High Entropy Sample scoring
- **AnswerProbScorer**: Answer probability scoring
- **AskLlmScorer**: LLM-based quality inquiry
- **FailRateScorer**: Failure rate evaluation
- **InstagScorer**: Instruction tag classification

## 🚀 Quick Start

### 1. Configure YAML File

Create or modify a configuration file in the `configs/` directory, e.g., `configs/my_scorer.yaml`:

```yaml
# Data path configuration
input_path: /path/to/your/data.jsonl
output_path: results/my_experiment

# Global GPU configuration
num_gpu: 8                    # Total number of available GPUs
num_gpu_per_job: 1            # Default number of GPUs per task (can be overridden by scorers)

# Scorer configuration
scorers:
  # Example 1: Quality scorer (using 1 GPU)
  - name: DeitaQScorer
    model: /path/to/deita-quality-scorer
    max_length: 2048
    batch_size: 8
    num_gpu_per_job: 1
  
  # Example 2: Multi-GPU task (using 4 GPUs)
  - name: InfOrmScorer
    model: /path/to/INF-ORM-Llama3.1-70B
    batch_size: 8
    max_length: 2048
    num_gpu_per_job: 4
  
  # Example 3: Multiple scorers
  - name: PPLScorer
    model: /path/to/language-model
    batch_size: 16
    max_length: 2048
    num_gpu_per_job: 1
```

**Configuration Details**:
- **`input_path`**: Input data file path (JSONL format)
- **`output_path`**: Output results directory
- **`num_gpu`**: Total number of globally available GPUs (required)
- **`num_gpu_per_job`**: Global default GPUs per task (optional, default 1)
- **`scorers`**: List of scorers, each can specify its own `num_gpu_per_job` to override the global setting

### 2. Prepare Data

Ensure input data is in JSONL format, with one JSON object per line:

```json
{"instruction": "What is machine learning?", "output": "Machine learning is...",...}
{"instruction": "Explain neural networks", "output": "Neural networks are...",...}
```

**Field Requirements**:
- `instruction`: Question or instruction (required)
- `output`: Answer or output (required for QA-type scorers)
- `input`: Additional input field (optional)
- Other fields: Some scorers may require additional specific fields; please refer to the corresponding scorer documentation

### 3. Run Evaluation

```bash
python main_para.py --config configs/my_scorer.yaml
```

**Parameter Description**:
- `--config`: Path to YAML configuration file

## 🔧 Data Parallelism Mechanism

### Automatic Parallelism Calculation

The framework automatically calculates data parallelism for each scorer:

```
data_parallel = num_gpu ÷ num_gpu_per_job
```

**Example**:
- 8 GPUs available globally
- Scorer A needs 1 GPU → data_parallel = 8 (data split into 8 parts for parallel processing)
- Scorer B needs 4 GPUs → data_parallel = 2 (data split into 2 parts for parallel processing)
- Scorer C needs 2 GPUs → data_parallel = 4 (data split into 4 parts for parallel processing)

### GPU Allocation Strategy

1. **Data Splitting**: Dataset is split into `data_parallel` parts
2. **Process Launch**: An independent process is launched for each data part
3. **GPU Assignment**: Each process sees its assigned GPUs through `CUDA_VISIBLE_DEVICES`
4. **Parallel Execution**: All processes run simultaneously without interference
5. **Result Merging**: Results are automatically merged after all processes complete

**Example: 8 GPUs, scorer requires 2 GPUs per job**
```
Job 0: GPU [0, 1] → Processes data shard 0
Job 1: GPU [2, 3] → Processes data shard 1
Job 2: GPU [4, 5] → Processes data shard 2
Job 3: GPU [6, 7] → Processes data shard 3
```

## 📤 Output Results

After completion, the following files will be generated in the `output_path` directory:

### Pointwise Scoring Results (`pointwise_scores.jsonl`)

Per-sample scoring results, each line corresponds to one record in the input data:

```json
{
  "id": 0,
  "scores": {
    "DeitaQScorer": {
      "score": 4.5
    },
    "PPLScorer": {
      "score": 12.34
    },
    "UPDScorer": {
      "score": 0.79
    }
  }
}
```

### Setwise Scoring Results (`setwise_scores.jsonl`)

Scoring results for the entire dataset (if any scorer returns overall scores):

```json
{
  "Task2VecScorer": {
    "score": 0.024327838269528,
    "num_samples": 32,
    "num_anomalous": 0,
    "num_truncated": 31,
    "truncation_rate": 0.96875,
    "last_layer_only": true,
    "embedding_dim": 768
  }
}
```

### Intermediate Results (`master_temp/`)

```
master_temp/
├── processed_data.jsonl              # Preprocessed data (with id added)
├── scorer_DeitaQScorer/              # Temporary directory for each scorer
│   ├── job_0/                        # Results from each parallel task
│   │   └── DeitaQScorer.jsonl
│   ├── job_1/
│   │   └── DeitaQScorer.jsonl
│   └── DeitaQScorer_merged.jsonl    # Merged results
└── scorer_PPLScorer/
    └── PPLScorer.jsonl
```

## 📝 Scorer Configuration Details

### Common Parameters

All scorers support the following parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | string | Scorer name (required) | - |
| `num_gpu_per_job` | int | Number of GPUs this scorer requires | 1 |

### Model-based Scorer Parameters

Most deep learning model-based scorers support:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | string | Model path or HuggingFace model name | - |
| `batch_size` | int | Batch size | 8 |
| `max_length` | int | Maximum sequence length | 2048 |
| `num_gpu_per_job` | int | Number of GPUs this scorer requires | 1 |

### Specific Scorer Parameters

For detailed scorer configurations, please refer to:
- **Configuration Examples**: `configs/MultiScorer.yaml` (contains complete configurations for all model-based scorers)
- **Online Documentation**: [Wiki Page](https://opendataarena-tool.readthedocs.io/en/latest/model-based-evaluation/)

## 🎯 Usage Scenario Examples

### Scenario 1: Quick Quality Assessment

Evaluate basic quality metrics of a dataset:

```yaml
num_gpu: 4
scorers:
  - name: DeitaQScorer    # Quality
  - name: DeitaCScorer    # Complexity  
  - name: PPLScorer       # Perplexity
```

### Scenario 2: Comprehensive Data Analysis

Perform in-depth analysis using scorers from multiple dimensions:

```yaml
num_gpu: 8
scorers:
  - name: SkyworkRewardScorer      # Quality
  - name: IFDScorer                # Difficulty
  - name: PPLScorer                # Perplexity
  - name: UPDScorer                # Uncertainty difficulty
```

### Scenario 3: Large Model Evaluation

Evaluation using large-scale models:

```yaml
num_gpu: 16
scorers:
  - name: InfOrmScorer
    model: /path/to/INF-ORM-Llama3.1-70B
    num_gpu_per_job: 8              # Use 8 GPUs → data_parallel=2
    batch_size: 4
```

### Scenario 4: Data Selection Optimization

For data selection and filtering:

```yaml
num_gpu: 8
scorers:
  - name: HESScorer                # High entropy samples
  - name: SelectitTokenScorer      # SelectIT token scoring
  - name: SelectitSentenceScorer   # SelectIT sentence scoring
  - name: GraNdScorer              # Gradient norm difference
```

## ⚙️ Advanced Features

### Custom Scorers

1. Inherit from the `BaseScorer` class:

```python
from scorers.base_scorer import BaseScorer
from typing import Dict, List, Any

class MyCustomScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        required = ["model", "batch_size"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def _setup(self):
        """Initialize model and resources"""
        self.model = load_model(self.config["model"])
    
    def score_item(self, data_item: Dict) -> Dict:
        """Score a single sample"""
        score = self.model.score(data_item["instruction"])
        return {"custom_score": score}
    
    def evaluate(self, dataset_path: str) -> List[Dict]:
        """Evaluate the entire dataset"""
        results = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                score = self.score_item(item)
                score["id"] = item["id"]
                results.append(score)
        return results
```

2. Register in `scorers/scores_info.json`:

```json
{
  "name": "MyCustomScorer",
  "module": "scorers.MyCustomScorer"
}
```

3. Use in configuration file:

```yaml
scorers:
  - name: MyCustomScorer
    model: /path/to/model
    batch_size: 8
```


## 📚 References

- **Configuration Examples**: `configs/MultiScorer.yaml` - Complete configurations for all model-based scorers
- **Online Documentation**: [https://opendataarena-tool.readthedocs.io](https://opendataarena-tool.readthedocs.io)
- **Scorer Details**: Visit the Wiki page to learn detailed descriptions and paper references for each scorer


## 🤝 Contributing

Issues and Pull Requests are welcome!
