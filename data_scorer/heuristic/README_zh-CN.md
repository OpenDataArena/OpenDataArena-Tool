# 基于启发式的数据评估框架

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

本框架提供了一套完整的数据质量评估系统，利用统计方法和启发式算法对数据集进行多维度评估。

## ✨ 核心特性

- **🚀 CPU 多进程并行**: 使用 `ProcessPoolExecutor` 实现高效的 CPU 并行计算
- **📊 双模式评分**: 区分 **Pointwise**（逐样本评分）和 **Setwise**（全集评分）两种评分模式
- **🔧 配置驱动**: 通过 YAML 配置文件轻松管理评估指标和运行参数
- **💾 结构化输出**: 统一保存评分结果，无数据分片，便于结果查看和处理
- **🔄 顺序执行**: 多个评分器按顺序执行，每个评分器内部使用多进程并行加速
- **⚡ 高效计算**: 基于启发式算法，无需 GPU，计算效率高

## 📦 支持的评分器

本框架集成了 **23 种启发式评分器**，涵盖多样性、统计特征、内容检测等多个维度：

### 📈 多样性类

评估数据集的多样性、覆盖度、独特性等维度：

- **VendiScorer**: Vendi Score 多样性度量
- **KNNScorer**: K 近邻多样性评分
- **ApsScorer**: 平均成对相似度
- **ApjsScorer**: 平均 Jaccard 相似度
- **RadiusScorer**: 数据半径评分
- **ClusterInertiaScorer**: 聚类惯性评分
- **PartitionEntropyScorer**: 分区熵评分
- **NovelSumScorer**: 新颖性与代表性评分
- **FacilityLocationScorer**: 设施位置函数评分
- **UniqueNgramScorer**: N-gram 唯一性评分
- **UniqueNtokenScorer**: N-token 唯一性评分
- **MtldScorer**: 词汇多样性度量
- **VocdDScorer**: 词汇密度 D 值
- **TokenEntropyScorer**: Token 熵评分
- **GramEntropyScorer**: N-gram 熵评分
- **HddScorer**: HD-D 多样性评分

### 📊 统计特征类

评估数据的基础统计特征：

- **TokenLengthScorer**: Token 长度统计
- **StrLengthScorer**: 字符串长度统计
- **TreeInstructScorer**: 语法树统计
- **LogDetDistanceScorer**: 对数行列式距离评分

### 🔍 内容检测类

检测数据的特定内容特征：

- **ThinkOrNotScorer**: 是否包含思考检测
- **PureThinkScorer**: 纯思考内容检测
- **TsPythonScorer**: Python 代码检测

## 🚀 快速开始

### 1. 配置 YAML 文件

在 `configs/` 目录下创建或修改配置文件，例如 `configs/my_scorer.yaml`:

```yaml
# 数据路径配置
input_path: /path/to/your/data.jsonl
output_path: results/my_experiment

# 全局 GPU 配置
num_gpu: 0
num_gpu_per_job: 0

# 评分器配置
scorers:
  # 示例 1: Token 长度统计
  - name: TokenLengthScorer
    encoder: o200k_base
    fields: ["instruction", "input", "output"]
    max_workers: 128
  
  # 示例 2: 多样性评估
  - name: VendiScorer
    embedding_path: ../data_process/mock_embedding_128x512.npy
    similarity_metric: euclidean
    max_workers: 128
  
  # 示例 3: KNN 多样性
  - name: KNNScorer
    k: 10
    distance_metric: cosine
    max_workers: 128
    embedding_path: ../data_process/mock_embedding_128x512.npy
```

**配置说明**:
- **`input_path`**: 输入数据文件路径（JSONL 格式）
- **`output_path`**: 输出结果目录
- **`num_gpu`**: 全局可用 GPU 总数（可选，默认 0，启发式评分通常不需要 GPU）
- **`num_gpu_per_job`**: 全局默认的每任务 GPU 数量（可选，默认 0，启发式评分通常不需要 GPU）
- **`scorers`**: 评分器列表，每个评分器可指定 `max_workers` 控制并行度

### 2. 准备数据

确保输入数据为 JSONL 格式，每行一个 JSON 对象：

```json
{"instruction": "What is machine learning?", "output": "Machine learning is...",...}
{"instruction": "Explain neural networks", "output": "Neural networks are...",...}
```

**字段要求**:
- `instruction`: 问题或指令（必需）
- `output`: 回答或输出（对于 QA 类评分器必需）
- `input`: 额外的输入字段（可选）
- 其他字段: 某些评分器可能还要求其它特定字段，具体请参考对应评分器的说明或配置文档

### 3. 运行评估

```bash
python main_para.py --config configs/my_scorer.yaml
```

**参数说明**:
- `--config`: YAML 配置文件路径

## 🔧 CPU 并行机制

### 并行处理说明

启发式评分器使用 **CPU 并行处理**，通过 `ProcessPoolExecutor` 实现多进程并行

### 并行执行流程

1. **初始化**: 创建大小为 `max_workers` 的进程池
2. **任务分配**: 将数据项动态分配给空闲的工作进程
3. **并行计算**: 多个进程同时处理不同的数据项
4. **结果收集**: 收集所有进程的计算结果并统一保存

**注意**: 与基于模型的评分器不同，启发式评分器不会将数据分片到不同的 job 目录中。

## 📤 输出结果

运行完成后，在 `output_path` 目录下会生成以下文件：

### Pointwise 评分结果 (`pointwise_scores.jsonl`)

逐样本的评分结果，每行对应输入数据的一条记录：

```json
{
  "id": 0,
  "scores": {
    "TokenLengthScorer": {
      "score": 120
    },
    "StrLengthScorer": {
      "score": 518
    },
    "ThinkOrNotScorer": {
      "score": 1
    }
  }
}
```

### Setwise 评分结果 (`setwise_scores.jsonl`)

对整个数据集的评分结果：


```json
{
  "ApjsScorer": {
    "score": 0.16716303774426786,
    "num_samples": 30,
    "num_pairs": 435,
    "total_possible_pairs": 435,
    "is_sampled": false,
    "tokenization_method": "gram",
    "n": 1,
    "similarity_method": "direct",
    "max_workers": 128
  },
  "ApsScorer": {
    "score": 0.45395749064657004,
    "num_samples": 30,
    "num_pairs": 435,
    "total_possible_pairs": 435,
    "is_sampled": false,
    "similarity_metric": "euclidean",
    "max_workers": 128
  }
}
```

### 中间结果 (`master_temp/`)

```
master_temp/
├── processed_data.jsonl              # 预处理后的数据（添加了 id）
├── scorer_TokenLengthScorer/         # 每个评分器的临时目录
│   └── TokenLengthScorer.jsonl       # 评分结果
└── scorer_VendiScorer/
    └── VendiScorer.jsonl              # 评分结果
```


## 📝 评分器配置详解

### 通用参数

所有评分器都支持以下参数：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `name` | string | 评分器名称（必需） | - |
| `max_workers` | int | 最大并行工作进程数 | 128 |
| `num_gpu` | int | 总共所需GPU 数量（启发式评分器通常设为 0） | 0 |
| `num_gpu_per_job` | int | 单数据分片任务所需GPU 数量（启发式评分器通常设为 0） | 0 |



### 特定评分器参数

详细的评分器配置请参考：
- **配置示例**: `configs/` 目录下的配置文件
- **在线文档**: [Wiki 页面](https://opendataarena-tool.readthedocs.io/en/latest/heuristic-evaluation/)

## 🎯 使用场景示例

### 场景 1: 基础统计分析

评估数据集的基础统计特征：

```yaml
scorers:
  - name: TokenLengthScorer        # Token 长度统计
    encoder: o200k_base
    fields: ["instruction", "output"]
  - name: StrLengthScorer          # 字符串长度统计
  - name: TreeInstructScorer       # 语法树统计
```

### 场景 2: 多样性评估

使用多个维度的评分器评估数据多样性：

```yaml
scorers:
  - name: VendiScorer              # Vendi 多样性
  - name: KNNScorer                # KNN 多样性
    k: 10
  - name: ApsScorer                # 平均成对相似度
  - name: UniqueNgramScorer        # N-gram 唯一性
```

### 场景 3: 词汇多样性分析

评估数据的词汇丰富度和多样性：

```yaml
scorers:
  - name: MtldScorer               # 词汇多样性度量
  - name: VocdDScorer              # 词汇密度 D 值
  - name: TokenEntropyScorer       # Token 熵评分
  - name: GramEntropyScorer        # N-gram 熵评分
  - name: HddScorer                # HD-D 多样性评分
```

### 场景 4: 内容检测

检测数据中的特定内容特征：

```yaml
scorers:
  - name: ThinkOrNotScorer         # 是否包含思考
  - name: PureThinkScorer          # 纯思考内容检测
  - name: TsPythonScorer           # Python 代码检测
```

### 场景 5: 全面数据分析

综合使用多种评分器进行深度分析：

```yaml
scorers:
  - name: VendiScorer              # 多样性
  - name: KNNScorer                # KNN 多样性
  - name: TokenLengthScorer        # 长度统计
  - name: MtldScorer               # 词汇多样性
  - name: ThinkOrNotScorer         # 内容检测
```

## ⚙️ 高级功能

### 自定义评分器

1. 继承 `BaseScorer` 类：

```python
from scorers.base_scorer import BaseScorer
from typing import Dict, List, Any
import json

class MyCustomScorer(BaseScorer):
    def _validate_config(self):
        """验证配置参数"""
        required = ["max_workers"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def _setup(self):
        """初始化资源"""
        self.max_workers = self.config.get("max_workers", 128)
    
    def score_item(self, data_item: Dict) -> Dict:
        """评分单个样本"""
        # 实现自定义的启发式评分逻辑
        text = data_item["instruction"]
        score = len(text.split())  # 示例：统计单词数
        return {"custom_score": score}
    
    def evaluate(self, dataset_path: str) -> List[Dict]:
        """评估整个数据集"""
        results = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                score = self.score_item(item)
                score["id"] = item["id"]
                results.append(score)
        return results
```

2. 在 `scorers/scores_info.json` 中注册：

```json
{
  "name": "MyCustomScorer",
  "module": "scorers.MyCustomScorer"
}
```

3. 在配置文件中使用：

```yaml
scorers:
  - name: MyCustomScorer
    max_workers: 128
```


## 📚 参考资料

- **配置示例**: `configs/` 目录 - 包含各种启发式评分器的配置示例
- **在线文档**: [https://opendataarena-tool.readthedocs.io](https://opendataarena-tool.readthedocs.io)
- **评分器详解**: 访问 Wiki 页面了解每个评分器的详细说明


## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
