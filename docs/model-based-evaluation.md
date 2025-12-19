# Model-based Scorers

Model-based scorers leverage pre-trained neural networks, language models, and specialized machine learning models to evaluate data quality, difficulty, and diversity. Unlike heuristic methods, these scorers capture semantic understanding, contextual relationships, and learned patterns from large-scale training, providing more nuanced and task-specific assessments.

---

## AnswerProbScorer

### Overview

The **Answer Probability Scorer** is a model-based evaluation tool designed to assess the quality and difficulty of instruction-answer pairs by computing the conditional probability of answers given instructions. This scorer leverages causal language models to measure how well an answer aligns with its corresponding instruction through probabilistic analysis.

**Important Design Choice:** This scorer **focuses exclusively on the final answer tokens** (e.g., extracted from `\boxed{...}` notation or the `answer` field), rather than evaluating the entire output including reasoning steps and explanations. This design choice ensures:
- **Precision**: Direct measurement of how well the instruction guides to the correct answer
- **Fairness**: Avoids bias from varying output lengths and writing styles
- **Clarity**: Scores directly reflect instruction-answer alignment, not intermediate reasoning quality

Unlike simple perplexity-based metrics, the Answer Probability Scorer implements a **normalized scoring mechanism** that accounts for the intrinsic probability of the answer itself. By comparing the conditional probability P(Answer|Instruction) against the baseline probability P(Answer), this method provides a more robust measure of instruction-answer alignment that is less biased by answer length or common phrase frequencies.

### Metric Definition:

* **Definition:** 

  Given an instruction Q and answer A (note: **only the final answer is evaluated, not the entire output**), the scorer computes:
  
  1. **Conditional Probability Score (P_A):** The average log probability of **answer tokens only** given the full instruction-answer context
  2. **Baseline Probability Score (P_B):** The average log probability of **answer tokens only** without any instruction context
  3. **Normalized Score:** `score = log(P_A) - log(P_B) = log(P_A / P_B)`

* **Why Focus Only on Answer Tokens?**
  
  This scorer deliberately excludes intermediate reasoning steps and explanations from the output, focusing solely on the final answer because:
  - The goal is to measure **how effectively the instruction guides to the correct answer**
  - Including reasoning steps would introduce noise from varying writing styles and output lengths
  - Answer-focused evaluation provides fairer comparison across samples with different output structures
  - This aligns with practical scenarios where the correctness of the final answer is the primary concern

* **Explanation:** This metric measures the **relative probability gain** when the instruction is provided:
  
  * A **higher normalized score** (positive value) indicates that the instruction **significantly increases** the likelihood of the answer, suggesting strong instruction-answer alignment and higher quality.
  * A **lower normalized score** (negative value) indicates that the instruction provides **little guidance** or even **contradicts** the natural answer generation, suggesting poor alignment or lower quality.
  * A **score close to zero** suggests that the instruction provides **marginal information** beyond what the model already knows.

* **Key Advantages:**
  
  * **Length-invariant:** By using average log probabilities, the metric is not biased by answer length
  * **Baseline normalization:** Subtracting the unconditional answer probability removes bias toward common phrases
  * **Log-space computation:** Prevents numerical underflow and provides interpretable probability ratios

### YAML Configuration

```yaml
name: AnswerProbScorer
model: Qwen/Qwen2.5-7B
case_sensitive: true
batch_size: 16
max_length: 2048
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"AnswerProbScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path for the causal language model used to compute probabilities |
| `case_sensitive` | boolean | `true` | Whether to perform case-sensitive answer extraction and matching |
| `batch_size` | integer | `1` | Number of samples to process in parallel per forward pass |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |


### Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute token-level probabilities. By default, it uses **Qwen/Qwen3-8B**, but can be configured to use any autoregressive language model.

### Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` and optional `input` fields)
   - Answer (from `answer` field if present, otherwise extracted from `output` using `\boxed{...}` notation)

2. **Tokenization**: The concatenated instruction-answer text is tokenized with offset mapping to track character-to-token alignment

3. **Forward Pass A (Conditional)**: Compute log probabilities for all tokens in the instruction+answer sequence

4. **Answer Token Identification**: Use offset mapping to identify which tokens correspond to the answer segment

5. **Forward Pass B (Baseline)**: Compute log probabilities for the answer-only sequence without instruction

6. **Score Computation**: Calculate normalized score as `log(P_A / P_B)` where:
   - P_A = average log probability of answer tokens in full context
   - P_B = average log probability of answer tokens without instruction

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "mean_prob": -2.3456,
  "token_count": 15,
  "answers": ["42"],
  "answer_str": "42",
  "mean_prob_answer_only": -3.1234,
  "score": 0.7778,
  "answer_only_token_count": 15
}
```

- `mean_prob`: Average log probability of answer tokens given instruction (P_A)
- `token_count`: Number of answer tokens used in conditional probability calculation
- `answers`: Extracted answer(s) from the output
- `answer_str`: Comma-separated string of all answers
- `mean_prob_answer_only`: Average log probability of answer tokens without instruction (P_B)
- `score`: Normalized score = log(P_A / P_B)
- `answer_only_token_count`: Number of tokens in answer-only sequence

### Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```



---

## AskLlmScorer

### Overview

The **AskLlmScorer** is a sample-wise data quality evaluation metric that leverages Large Language Models (LLMs) to assess the quality of individual training samples. It computes the **average log probability** of a specified positive token (e.g., "yes") given a prompt and the data sample, effectively measuring how likely the model believes the sample is high quality.

### Metric Definition:

* **Definition:**

  Given a prompt P, data sample D, and target positive token(s) Y, the scorer computes:
  
  For a single-token yes_token:
  ```
  Score = log P(yes_token | prompt, data)
  ```
  
  For a multi-token yes_token:
  ```
  Score = (1/T) × Σ log P(token_i | prompt, data, token_1...token_{i-1})
  ```
  
  where T is the number of tokens in the yes_token.

* **Explanation:** The metric quantifies data quality through conditional probability:
  
  * A **higher score** (closer to 0) indicates the LLM assigns **high probability** to the positive response, suggesting the sample is likely high quality.
  * A **lower score** (more negative) indicates the LLM assigns **low probability** to the positive response, suggesting the sample may be low quality.
  * Scores typically range from approximately -10 to 0, with scores above -2 generally indicating good quality.

* **Key Advantages:**
  
  * **Customizable prompts:** Allows flexible quality criteria through prompt engineering
  * **Multi-token support:** Handles both single and multi-token positive responses
  * **Probabilistic interpretation:** Provides interpretable quality scores based on model confidence

### YAML Configuration

```yaml
name: AskLlmScorer
model: Qwen/Qwen2.5-7B
prompt: "Is the following data high quality? Please answer yes or no.\n\n"
yes_token: "yes"
batch_size: 8
max_length: 2048
model_dtype: bfloat16
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"AskLlmScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen2.5-7B"` | HuggingFace model identifier or local path to the evaluation LLM |
| `prompt` | string | `"Is the following data high quality? Please answer yes or no.\n\n"` | Prompt template that precedes the data sample |
| `yes_token` | string | `"yes"` | Target token(s) representing positive/high-quality response |
| `batch_size` | integer | `8` | Number of samples to process in parallel per batch |
| `max_length` | integer | `2048` | Maximum sequence length in tokens; longer sequences are truncated |
| `model_dtype` | string | `bfloat16` | Precision for model loading: `"float32"`, `"bfloat16"`, or `"float16"` |

### Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute token-level probabilities. By default, it uses **Qwen/Qwen2.5-7B**, but can be configured to use any autoregressive language model with strong instruction-following capabilities. Larger models generally provide more accurate quality judgments but require more computational resources.

### Scoring Process

1. **Data Preparation**: For each sample, construct the full text by concatenating instruction, input (if present), and output fields.

2. **Prompt Construction**: Build the evaluation prompt by appending the data sample to the configured prompt template, followed by the yes_token.

3. **Batch Tokenization**: Tokenize samples in batches with padding and truncation. Track prompt lengths to identify yes_token positions.

4. **Model Inference**: Run forward pass through the LLM to obtain logits for all token positions in batch mode.

5. **Log Probability Computation**: For each sample, extract log probabilities of yes_token tokens given the prompt and data context. Compute average log probability across all yes_token tokens. Critical computations use float32 precision for numerical stability.

6. **Score Assignment**: Return the average log probability as the quality score. Handle edge cases (truncation, empty tokens) by assigning a score of -100.0.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": -0.523
}
```

- `id`: Unique identifier for the sample (from input data's `id` field, or empty string if not present)
- `score`: Average log probability of yes_token given prompt and data. Higher (less negative) scores indicate higher quality. Typical range: -10 to 0, with scores above -2 generally indicating good quality

### Citation

```bibtex
@article{sachdeva2024train,
  title={How to train data-efficient llms},
  author={Sachdeva, Noveen and Coleman, Benjamin and Kang, Wang-Cheng and Ni, Jianmo and Hong, Lichan and Chi, Ed H and Caverlee, James and McAuley, Julian and Cheng, Derek Zhiyuan},
  journal={arXiv preprint arXiv:2402.09668},
  year={2024}
}
```


---

## AtheneScorer

### Overview

The **AtheneScorer** is a sample-wise quality evaluation metric that leverages the Athene reward model to assess the quality of instruction-response pairs in supervised fine-tuning (SFT) datasets. Unlike traditional metrics that focus on surface-level features, AtheneScorer provides a learned quality signal by using a reward model trained to distinguish between high-quality and low-quality responses.

This metric is particularly useful for:
- **Response quality assessment**: Evaluating the helpfulness and appropriateness of model outputs
- **Dataset curation**: Filtering high-quality instruction-response pairs for training
- **RLHF data preparation**: Identifying superior responses for reinforcement learning from human feedback
- **Model evaluation**: Comparing the quality of responses from different models

The AtheneScorer supports batch processing for efficient evaluation and is based on the Llama-3-8B-Instruct architecture fine-tuned specifically as a reward model.

### Metric Definition:

* **Definition:**

  The reward score is a scalar value computed by the Athene reward model that represents the quality of a given instruction-response pair. The model processes the conversation in chat format and extracts a reward signal from the final hidden states at the CLS token position.

  ```
  Reward(instruction, input, output) = RewardModel(conversation)
  ```

  where:
  - `instruction` is the task description or question
  - `input` is additional context (optional)
  - `output` is the model's response
  - `conversation` is formatted as a chat template with user and assistant roles

* **Explanation:** The Athene reward model provides a quality assessment based on learned preferences:

  * A **higher reward score** indicates **better quality**, meaning the response is more helpful, accurate, appropriate, and well-aligned with the instruction
  * A **lower reward score** indicates **lower quality**, suggesting the response may be unhelpful, inaccurate, inappropriate, or poorly aligned with the instruction
  * The score is unbounded but typically ranges from negative to positive values, with the relative ranking being more meaningful than absolute values

### YAML Configuration

```yaml
name: AtheneScorer
model: Nexusflow/Athene-RM-8B
batch_size: 32
max_length: 4096
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"AtheneScorer"` | Identifier for the scorer |
| `model` | string | `"Nexusflow/Athene-RM-8B"` | HuggingFace model path or local checkpoint for the reward model |
| `batch_size` | integer | `32` | Number of samples to process in parallel per forward pass |
| `max_length` | integer | `4096` | Maximum sequence length for tokenization; longer sequences are truncated from the left |


### Underlying Model

The scorer uses the **Nexusflow/Athene-RM-8B** reward model, which is based on the **Llama-3-8B-Instruct** architecture fine-tuned specifically for reward modeling. The model consists of a Llama-3-8B transformer backbone with a linear reward head that projects hidden states to a scalar reward score. A special CLS token is appended to conversations for reward extraction from the final hidden states.

### Scoring Process

1. **Input Formatting**: For each sample, construct a chat conversation with user role (instruction + input) and assistant role (output)

2. **Chat Template Application**: Apply the Llama-3 chat template to format the conversation and append the CLS token

3. **Tokenization**: Tokenize the formatted text; truncate from the left if exceeding `max_length`

4. **Batch Processing**: Process samples in batches with padding for efficient GPU computation

5. **Reward Computation**: Forward pass through the model to extract reward value at the CLS token position from the final hidden states

6. **Score Extraction**: Convert tensor outputs to float values and return scores for each sample

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.456
}
```

- `id`: Unique identifier of the sample from the input dataset
- `score`: Reward score computed by the Athene model (unbounded float value; higher is better)

### Citation

```bibtex
@misc{Athene2024,
    title = {Athene-70B: Redefining the Boundaries of Post-Training for Open Models},
    url = {https://nexusflow.ai/blogs/athene},
    author = {Frick, Evan and Jin, Peter and Li, Tianle and Ganesan, Karthik and Zhang, Jian and Jiao, Jiantao and Zhu, Banghua},    
    month = {July},
    year = {2024}
}
```



---

## CleanlinessScorer

### Overview

The **Cleanliness Scorer** is a model-based evaluation tool designed to assess the **format quality** and **noise-free content** of SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates how well-formatted, complete, and structurally sound a text is, focusing on presentation quality rather than semantic content. The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable cleanliness assessments on a continuous 0-5 scale.

### Metric Definition:

* **Definition:** 

  A continuous score ranging from 0 to 5 that quantifies the format quality and structural integrity of text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** 

  The Cleanliness metric evaluates text across three primary dimensions:
  
  1. **Correct Formatting** - Text appears human-edited with proper structure and no corrupted characters
  2. **Appropriate Content** - No irrelevant links, advertisements, or spam; sufficient content length
  3. **Completeness** - Complete sentences with coherent structure and natural flow

* **Score Interpretation:**
  
  * **4.0-5.0**: High-quality content with perfect or near-perfect formatting
  * **3.0-3.9**: Acceptable content with minor issues that don't seriously impact readability
  * **2.0-2.9**: Obvious problems that noticeably affect reading fluency
  * **1.0-1.9**: Serious formatting or structural issues
  * **0.0-0.9**: Absolute noisy content unsuitable for training

### YAML Configuration

```yaml
name: CleanlinessScorer
model: opendatalab/meta-rater-cleanliness-rating
batch_size: 16
max_model_len: 8192
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"CleanlinessScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-cleanliness-rating"` | HuggingFace model path or local path to the cleanliness rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_model_len` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated) |

### Underlying Model

The scorer uses [opendatalab/meta-rater-cleanliness-rating](https://huggingface.co/opendatalab/meta-rater-cleanliness-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 87.88% F1 score, 92.25% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

### Scoring Process

The Cleanliness Scorer follows a systematic evaluation pipeline:

1. **Text Concatenation**: For each data sample, the scorer concatenates the fields in the following order:
   ```
   content = instruction + "\n" + input + "\n" + output
   ```
   If the `input` field is empty, it uses:
   ```
   content = instruction + "\n" + output
   ```

2. **Tokenization**: The concatenated text is tokenized using the ModernBERT tokenizer with:
   * Left-side padding for batch processing
   * Truncation at `max_model_len` (default 8,192 tokens)
   * Automatic padding for batch inference

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5).

4. **Score Calculation**: Instead of using the argmax prediction, the scorer computes a **continuous score** using weighted probability averaging:
   ```
   score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]
   ```
   where `P(class_i)` is the softmax probability of class `i`.

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.85
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the cleanliness score (higher values indicate better formatting and structural quality)

### Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```



---

## DebertaScorer

### Overview

The **Deberta Quality Classifier** is a model-based evaluation tool designed to assess the overall quality of supervised fine-tuning (SFT) data. Built upon the DeBERTa-v3 architecture, this classifier assigns each instruction-response pair a quality score (0, 1, or 2), representing three distinct quality levels: **Low**, **Medium**, and **High**. The model was trained on 22,828 Common Crawl text samples annotated by human evaluators who assessed quality based on factors such as content accuracy, clarity, coherence, grammar, depth of information, and overall usefulness.

### Metric Definition:

* **Definition:** The model assigns each SFT sample a quality score (0, 1, or 2) based on the concatenated text of instruction, input (if present), and output:
  * **Score = 2 (High Quality):** The content demonstrates excellent accuracy, clarity, coherence, proper grammar, substantial depth of information, and high overall usefulness.
  * **Score = 1 (Medium Quality):** The content shows acceptable quality with reasonable clarity and coherence, but may have minor issues in grammar, depth, or organization.
  * **Score = 0 (Low Quality):** The content exhibits poor quality, with significant issues in accuracy, clarity, coherence, grammar, or lacks meaningful information.

* **Explanation:** The quality score reflects the overall suitability of the data for supervised fine-tuning:
  * A **score of 2** indicates that the sample is well-suited for SFT and likely to improve model performance.
  * A **score of 1** suggests the sample has acceptable quality but may benefit from additional filtering or refinement.
  * A **score of 0** indicates the sample should likely be filtered out as it may negatively impact model training.

### YAML Configuration

```yaml
name: DebertaScorer
model: nvidia/quality-classifier-deberta
max_length: 2048
batch_size: 32
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"DebertaScorer"` | Identifier for the scorer |
| `model` | string | `"nvidia/quality-classifier-deberta"` | Path to the model, either a local directory or a HuggingFace model ID |
| `max_length` | integer | `2048` | Maximum token length for input text. Texts exceeding this length will be truncated |
| `batch_size` | integer | `32` | Number of samples to process in each batch |

### Underlying Model

The scorer uses [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta), a text classification model based on the **DeBERTa-v3 Base** architecture with a context length of 1024 tokens. The model consists of:

* A DeBERTa-v3 Base encoder for contextual text representation
* A dropout layer for regularization
* A linear classification head that outputs probabilities for three quality classes

The model was trained on human-annotated Common Crawl data and achieves an accuracy of **0.8252** on the evaluation set where all three annotators agreed on the label.

If the specified local model fails to load, the scorer automatically falls back to the default remote`nvidia/quality-classifier-deberta` model.

### Scoring Process

1. **Text Concatenation**: For each data sample, the scorer concatenates the following fields:
   - `instruction`: The instruction text
   - `input`: Additional input context (if present and non-empty)
   - `output`: The response or completion text
   - The final text format is: `instruction\n[input\n]output`

2. **Batch Tokenization**: Text samples are tokenized in batches using the DeBERTa tokenizer with padding, truncation to `max_length` tokens, and automatic addition of special tokens

3. **Truncation Warning**: If any text exceeds `max_length`, a warning is displayed indicating the number of truncated samples

4. **Model Inference**: The tokenized inputs are passed through the quality classification model:
   - The DeBERTa encoder generates contextual embeddings
   - Dropout is applied for regularization
   - The classification head outputs a probability distribution over the three quality classes
   - The class with the highest probability is selected via argmax and returned as an integer score (0, 1, or 2)

5. **Batch Processing**: All samples in the dataset are processed in batches of size `batch_size`, with a progress bar displaying the current status

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1
}
```

- `id`: The unique identifier of the sample, extracted from the input data's `id` field. If no `id` is present in the input, this will be an empty string
- `score`: An integer quality score where **0 = Low quality**, **1 = Medium quality**, **2 = High quality**

### Citation

```bibtex
@article{he2021debertav3,
  title={Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing},
  author={He, Pengcheng and Gao, Jianfeng and Chen, Weizhu},
  journal={arXiv preprint arXiv:2111.09543},
  year={2021}
}
```



---

## DeitaCScorer

### Overview

The **Deita Complexity Scorer** is a model-based evaluation tool designed to estimate the *instruction complexity* of SFT data. Proposed in the paper [Liu et al., 2024](https://arxiv.org/abs/2312.15685), this method aims to measure how cognitively demanding an instruction is for a model to execute. Rather than relying on shallow heuristics, the Deita Complexity Scorer provides a learning-based, instruction-only metric that correlates with downstream performance and instruction-following capabilities.

### Metric Definition:

* **Definition:**
  
    1. First generate variations of each instruction with increasing difficulty using the In-Depth Evolving Prompt method.
    2. Collect these data-score pairs to train a LLM as a complexity scorer.
    3. The trained scorer is used to predict complexity scores (1-6) for new instructions.

* **Explanation:** Intuitively, the complexity score estimates how *unexpected or difficult* an instruction is to follow for the SFT model.

  * A **higher Deita Complexity score** imply that the SFT model struggles with the instruction relative to the reference model, indicating **greater complexity**.
  * A **lower Deita Complexity score** suggest that the instruction is easy to complete and consistent with the SFT model's learned behaviors.

### YAML Configuration
```yaml
name: DeitaCScorer
model: hkust-nlp/deita-complexity-scorer
max_length: 2048
batch_size: 32
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"DeitaCScorer"` | Identifier for the scorer |
| `model` | string | `"hkust-nlp/deita-complexity-scorer"` | HuggingFace model path for the Deita complexity scorer |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `32` | Number of samples to process in parallel per forward pass |


### Underlying Model

The scorer uses [hkust-nlp/deita-complexity-scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer), which is introduced in [Liu et al., 2024](https://arxiv.org/abs/2312.15685).

### Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts the instruction from `instruction` field (combined with `input` field if present)

2. **Tokenization**: The instruction text is tokenized according to the model's tokenizer specifications

3. **Forward Pass**: The instruction is fed into the Deita complexity scorer model

4. **Score Prediction**: The model predicts a complexity score ranging from 1 (simplest) to 6 (most complex)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.5,
}
```

- `id`: Unique identifier for the input sample
- `score`: Complexity score ranging from 1-6, where higher values indicate more complex instructions

### Citation

```bibtex
@article{liu2023makes,
  title={What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning},
  author={Liu, Wei and Zeng, Weihao and He, Keqing and Jiang, Yong and He, Junxian},
  journal={arXiv preprint arXiv:2312.15685},
  year={2023}
}
```


---

## DeitaQScorer

### Overview

The **Deita Quality Scorer** is a model-based evaluation tool designed to estimate the data quality of instruction-tuning (SFT) data. This scorer was proposed in the paper [Liu et al., 2024](https://arxiv.org/abs/2312.15685) as part of the DEITA (Data-Efficient Instruction Tuning for Alignment) framework.

The scorer is trained to predict quality scores for instruction-answer pairs by learning from data variants with different quality levels. It provides an automated way to assess the overall quality of supervised fine-tuning samples, helping practitioners select high-quality data for efficient model alignment.

### Metric Definition:

* **Definition:**
  
  1. First generate different quality variants of the same data using the In-Depth Evolving Prompt method.
  2. Collect these data-score pairs to train a LLM as a quality scorer.
  3. The trained scorer is used to predict quality scores (1-6) for other SFT data samples.

* **Explanation:** Intuitively, the Deita Quality score estimates the overall quality of an SFT sample.
  
  * A **higher Deita Quality score** implies that the response presents data in a clear, accurate, and meaningful way.
  * A **lower Deita Quality score** suggests that the response is vague, misleading, or poorly organized in terms of data content.

### YAML Configuration

```yaml
name: DeitaQScorer
model: hkust-nlp/deita-quality-scorer
max_length: 2048
batch_size: 32
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"DeitaQScorer"` | Identifier for the scorer |
| `model` | string | `"hkust-nlp/deita-quality-scorer"` | HuggingFace model path for the quality scoring model |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `32` | Number of samples to process in parallel per forward pass |


### Underlying Model

The scorer uses [hkust-nlp/deita-quality-scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer), which is introduced in [Liu et al., 2024](https://arxiv.org/abs/2312.15685).

### Scoring Process

### Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts the instruction from `instruction` field (combined with `input` field if present) and `output` field

2. **Tokenization**: The combined text is tokenized according to the model's tokenizer specifications

3. **Forward Pass**: The combined is fed into the Deita quality scorer model

4. **Score Prediction**: The model predicts a quality score ranging from 1 (lowest quality) to 6 (highest quality)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.523
}
```

- `id`: Unique identifier of the data sample (inherited from input data)
- `score`: Predicted quality score as a float value, typically in the range [1.0, 6.0]

### Citation

```bibtex
@article{liu2023makes,
  title={What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning},
  author={Liu, Wei and Zeng, Weihao and He, Keqing and Jiang, Yong and He, Junxian},
  journal={arXiv preprint arXiv:2312.15685},
  year={2023}
}
```


---

## EffectiveRankScorer

### Overview

The **Effective Rank Scorer** is a gradient-based evaluation tool designed to assess the quality of instruction-following and reasoning data through spectral analysis of layer-wise gradients. This scorer computes the effective rank of gradient matrices derived from attention layer parameters (Q, K, V, O) during model fine-tuning, providing insights into the richness and complexity of gradient structures induced by training data.

Based on the research paper ["How Instruction and Reasoning Data shape Post-Training: Data Quality through the Lens of Layer-wise Gradients"](https://arxiv.org/abs/2504.10766), this method reveals that higher-quality data typically exhibits higher effective ranks, indicating richer gradient structures and more complex learning patterns. The effective rank metric demonstrates better robustness and resolution than nuclear norm in capturing subtle quality differences between instruction and reasoning data.

### Metric Definition:

* **Definition:** 
  
  The Effective Rank is computed through singular value decomposition (SVD) of gradient matrices:
  
  ```
  Effective_Rank = exp(H)
  
  where H = -Σ(p_i * ln(p_i))  (Shannon entropy)
  
  and p_i = σ_i / Σ(σ_j)  (normalized singular values)
  ```
  
  Where:
  - `σ_i` represents the i-th singular value from SVD of the gradient matrix
  - `p_i` forms a probability distribution from normalized singular values
  - `H` is the Shannon entropy computed using natural logarithm
  - `Effective_Rank` is the exponential of the entropy

* **Explanation:** 
  
  Effective Rank quantifies the dimensionality and richness of the gradient space:
  
  * A **higher Effective Rank** indicates that the gradients span a more diverse set of directions in parameter space, suggesting that the training sample induces **richer gradient structures** and potentially **higher data quality**.
  * A **lower Effective Rank** suggests that gradients are concentrated in fewer dimensions, indicating **simpler learning patterns** and potentially **lower data complexity**.
  
  The metric is computed separately for Query (Q), Key (K), Value (V), and Output (O) projection matrices in attention layers, providing fine-grained insights into how different attention components respond to training data.

### YAML Configuration

```yaml
name: EffectiveRankScorer
model: Qwen/Qwen3-8B
max_length: 2048
start_layer_index: 16
num_layers: 4
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"EffectiveRankScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path or local directory for the causal language model used to compute gradients |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization and gradient computation |
| `start_layer_index` | integer | `None` | Starting index of transformer layers to analyze (0-indexed). If `None`, only the last layer is analyzed |
| `num_layers` | integer | `1` | Number of consecutive layers to analyze starting from `start_layer_index`. Scores are averaged across all specified layers |

### Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute gradients through backpropagation. By default, it uses **Qwen/Qwen3-8B**, but can be configured to use any autoregressive transformer model. The model is used in training mode to compute gradients but is not updated—gradients are computed solely for analysis purposes.

### Scoring Process

1. **Input Processing**: For each data sample, concatenate the `instruction`, `input` (if present), and `output` fields into a single text sequence

2. **Tokenization**: Tokenize the concatenated text using the model's tokenizer with padding and truncation to `max_length`

3. **Forward Pass**: Set model to training mode and compute the language modeling loss through forward propagation

4. **Backward Pass**: Compute gradients via backpropagation using `loss.backward()` to accumulate gradients in parameter `.grad` attributes

5. **Layer Selection**: Determine target layers based on `start_layer_index` and `num_layers` parameters (defaults to last layer only)

6. **Gradient Extraction**: For each target layer, extract gradient matrices from attention projection parameters (Q, K, V, O)

7. **Effective Rank Computation**: For each gradient matrix:
   - Perform Singular Value Decomposition (SVD)
   - Normalize singular values to form probability distribution
   - Compute Shannon entropy: `H = -Σ(p_i * ln(p_i))`
   - Calculate Effective Rank: `exp(H)`

8. **Aggregation**: Average effective ranks across all specified layers for each projection type (Q, K, V, O)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "Q_EffectiveRank": 128.45,
  "K_EffectiveRank": 115.32,
  "V_EffectiveRank": 142.67,
  "O_EffectiveRank": 135.89
}
```

- `id`: Unique identifier of the sample
- `Q_EffectiveRank`: Effective rank of Query projection gradients, averaged across specified layers
- `K_EffectiveRank`: Effective rank of Key projection gradients, averaged across specified layers
- `V_EffectiveRank`: Effective rank of Value projection gradients, averaged across specified layers
- `O_EffectiveRank`: Effective rank of Output projection gradients, averaged across specified layers

### Citation

```bibtex
@article{li2025instruction,
  title={How instruction and reasoning data shape post-training: Data quality through the lens of layer-wise gradients},
  author={Li, Ming and Li, Yanhong and Li, Ziyue and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2504.10766},
  year={2025}
}
```


---

## FailRateScorer

### Overview

The **FailRateScorer** is a comprehensive evaluation framework designed to quantify the **failure rate** of **mathematical problems** by leveraging strong language models to estimate problem difficulty through multi-sample inference. This pipeline calculates the probability that a model will fail to solve a specific mathematical problem, providing an objective measure of problem complexity.

### Metric Definition:

* **Definition:** `Fail_Rate = 1 - sample_n_pass@1`

* **Explanation:** This metric estimates the **difficulty** of a problem by measuring the probability that a model gives the correct answer across multiple attempts.
  
  * A **higher value** (closer to 1) indicates the model is **more likely to fail**, suggesting the problem is **difficult**.
  * A **lower value** (closer to 0) indicates the model can **consistently provide correct answers**, suggesting the problem is **simple**.

### YAML Configuration

```yaml
name: FailRateScorer
model: Qwen/Qwen3-8B
metrics_sample_size: 4
generation_size: 4096
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"FailRateScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path or local directory for the language model used in evaluation |
| `metrics_sample_size` | integer | `4` | Number of sampling attempts per problem (options: 1, 4, 8, 16, 32, 64) |
| `generation_size` | integer | `4096` | Maximum generation length for model outputs |

**Note:** Currently, `metrics_sample_size` can only be set to 1, 4, 8, 16, 32, or 64, as these are the only preset options provided by the LightEval framework. If you need other metrics or custom evaluation methods, you can refer to the LightEval official documentation: [Adding a New Metric](https://huggingface.co/docs/lighteval/main/en/adding-a-new-metric) to add them yourself.

### Underlying Framework

The evaluation pipeline is implemented using Hugging Face's [**LightEval**](https://github.com/huggingface/lighteval) framework, which provides a robust and scalable evaluation infrastructure. The pipeline uses configurable language models (e.g., `Qwen/Qwen3-8B`) and operates through the following process:

1. **Task Generation:** Custom evaluation tasks are dynamically created for each split
2. **Parallel Evaluation:** Each split is evaluated on separate GPUs using the LightEval framework
3. **Result Aggregation:** Results are collected and merged back into the original dataset format

### Scoring Process

1. **Input Processing**: For each mathematical problem, the scorer extracts:
   - Problem statement (from `instruction` and optional `input` fields)
   - Ground truth answer (from `answer`)

2. **Multi-Sample Generation**: The model generates `metrics_sample_size` solutions for each problem using the specified language model

3. **Answer Extraction**: Extracts answers from generated outputs (typically from `\boxed{...}` notation or structured answer format)

4. **Correctness Verification**: Each generated answer is compared against the ground truth to determine correctness

5. **Fail Rate Calculation**: Computes the failure rate as `1 - (number of correct answers / total attempts)`

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.75,
}
```

- `id`: Unique identifier for the input sample
- `score`: Fail rate score ranging from 0-1, where higher values indicate more difficult problems (0 = always correct, 1 = always failed)

### Citation

```bibtex
@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.8.0},
  url = {https://github.com/huggingface/lighteval}
}
```


---

## FinewebEduScorer

### Overview

The **FineWeb-Edu Scorer** is a model-based evaluation tool designed to assess the **educational value** of text data. Originally developed to filter and curate educational content from web datasets, this classifier was trained on 450,000 annotations generated by LLama3-70B-Instruct for web samples from the FineWeb dataset. It provides a regression-based score indicating how educationally valuable a given text is, ranging from non-educational content to highly educational material suitable for learning purposes.

### Metric Definition:

* **Definition:** 

  A regression score ranging from 0 to 5 that quantifies the educational value of text content.

* **Explanation:** 

  The educational score estimates how suitable and valuable the content is for educational purposes, particularly for primary and grade school levels.
  
  * **Score 0-1:** Content has minimal to no educational value; may be commercial, entertainment-focused, or lacks substantive learning content.
  * **Score 2-3:** Content has moderate educational value with some informative elements but may lack depth or clarity.
  * **Score 4-5:** Content demonstrates high educational value with clear explanations, well-structured information, and strong pedagogical qualities.

* **Note:** 

  The model outputs continuous scores, but they are often rounded to integer values (0-5) for practical data curation. A threshold of `int_score >= 3` is commonly recommended for filtering educational content.

### YAML Configuration

```yaml
name: FinewebEduScorer
model: HuggingFaceFW/fineweb-edu-classifier
max_length: 2048
batch_size: 32
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"FinewebEduScorer"` | Identifier for the scorer |
| `model` | string | `"HuggingFaceFW/fineweb-edu-classifier"` | Path to the model checkpoint or Hugging Face model identifier. Can specify a local model path if you have a fine-tuned version |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization (valid range: 1-2048 tokens) |
| `batch_size` | integer | `32` | Number of samples to process in each batch. Adjust based on available GPU memory |

### Underlying Model

The scorer uses [HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier), a regression model built upon the **Snowflake-arctic-embed** architecture with an added classification head for single regression output.

#### Model Training Details

* **Base Architecture:** Snowflake-arctic-embed with a regression classification head
* **Training Data:** 450,000 web samples annotated by LLama3-70B-Instruct
* **Training Configuration:**
  * 20 epochs with learning rate 3e-4
  * Embedding and encoder layers frozen during training
  * Focus on training the classification head only
* **Performance:** Achieves 82% F1 score when converted to a binary classifier (threshold = 3)

The model was specifically trained to replicate the educational quality judgments of LLama3-70B-Instruct, making it a lightweight and efficient alternative for large-scale data filtering.

### Scoring Process

1. **Text Extraction:** For each data item, the scorer extracts and concatenates:
   - `instruction`: The task description or question
   - `input`: Additional context (if present)
   - `output`: The response or answer
   
   These fields are joined with newlines to form a complete text sample.

2. **Tokenization:** Texts are batch-tokenized using the model's tokenizer with:
   - Padding to the longest sequence in the batch
   - Truncation at `max_length` tokens
   - Conversion to PyTorch tensors

3. **Inference:** The model processes the tokenized inputs and outputs logits from the regression head. The logits are converted to float scores representing the educational value.

4. **Score Extraction:** Raw regression scores are returned as floating-point values. These can be:
   - Used directly as continuous scores for ranking
   - Rounded to integers (0-5) for classification purposes
   - Thresholded (e.g., `int_score >= 3`) for binary filtering

5. **Batch Processing:** The scorer processes data in configurable batch sizes for efficiency, with progress tracking via tqdm progress bars.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.847
}
```

- `id`: The unique identifier for the data sample, extracted from the input data's `id` field
- `score`: The continuous educational value score predicted by the model (range: 0-5, higher scores indicate greater educational value)

### Citation

```bibtex
@article{penedo2024fineweb,
  title={The fineweb datasets: Decanting the web for the finest text data at scale},
  author={Penedo, Guilherme and Kydl{\'\i}{\v{c}}ek, Hynek and Lozhkov, Anton and Mitchell, Margaret and Raffel, Colin A and Von Werra, Leandro and Wolf, Thomas and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={30811--30849},
  year={2024}
}
```



---

## Gpt2HarmlessScorer

### Overview

The **GPT2 Harmless Scorer** is a model-based evaluation tool designed to assess the harmlessness of instruction-response pairs in supervised fine-tuning (SFT) data. This scorer leverages a GPT2-large reward model specifically trained on the Anthropic/hh-rlhf harmless dataset to detect potentially harmful responses and evaluate alignment safety. The model achieves an accuracy of **0.73698** on the test set, matching the performance of larger models while maintaining computational efficiency.

### Metric Definition:

* **Definition:** 

  Given an instruction-response pair `(Q, A)`, the harmlessness reward model assigns a scalar reward score representing the safety and harmlessness of the response in the context of the instruction.

* **Explanation:** The reward score quantifies how safe and harmless a response is relative to the given instruction:
  
  * A **higher reward score** indicates the response is **more harmless and safer**, suggesting good alignment with safety preferences.
  * A **lower reward score** suggests the response may contain **harmful, toxic, or unsafe content**, indicating potential alignment issues.

* **Formulation:** Following the Anthropic/hh-rlhf dataset format:
  
  ```
  Input: "\n\nHuman: {instruction}\n\nAssistant:"
  Output: {response}
  ```

### YAML Configuration

```yaml
name: Gpt2HarmlessScorer
model: Ray2333/gpt2-large-harmless-reward_model
batch_size: 8
max_length: 1024
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"Gpt2HarmlessScorer"` | Identifier for the scorer. Must be set to `Gpt2HarmlessScorer` |
| `model` | string | `"Ray2333/gpt2-large-harmless-reward_model"` | Path or name of the harmless reward model from Hugging Face Hub. You can specify alternative GPT2-based harmlessness models trained on similar datasets |
| `batch_size` | integer | `8` | Number of samples to process simultaneously during inference. Adjust based on GPU memory availability |
| `max_length` | integer | `1024` | Maximum sequence length (in tokens) for tokenization. Sequences exceeding this length will be truncated |

### Underlying Model

The scorer uses [**Ray2333/gpt2-large-harmless-reward_model**](https://huggingface.co/Ray2333/gpt2-large-harmless-reward_model), a GPT2-large model fine-tuned as a reward model on the **Anthropic/hh-rlhf harmless dataset**. 

#### Model Characteristics

* **Base Architecture:** GPT2-large (774M parameters)
* **Training Data:** Anthropic/hh-rlhf harmless subset
* **Task:** Binary sequence classification (harmlessness scoring)
* **Precision:** bfloat16 for efficient inference
* **Performance:** 73.7% accuracy on the harmless test set

**Important Note:** This reward model differs from other open-source reward models trained on the full Anthropic/hh-rlhf dataset, as it focuses exclusively on the harmless subset for specialized harmlessness evaluation.

### Scoring Process

The GPT2 Harmless Scorer follows a structured pipeline to evaluate instruction-response pairs:

#### 1. **Input Formatting**

For each data item containing `instruction`, `input` (optional), and `output` fields:

```python
## If input field exists:
Q = "\n\nHuman: {instruction}\n{input}\n\nAssistant:"

## If input field is empty:
Q = "\n\nHuman: {instruction}\n\nAssistant:"

A = {output}
```

This formatting follows the Anthropic/hh-rlhf dataset convention, ensuring compatibility with the reward model's training format.

#### 2. **Tokenization**

The instruction `Q` and response `A` are tokenized together with:
* **Truncation:** Enabled with `max_length` parameter
* **Padding:** Applied to create uniform batch sizes
* **Warning System:** Samples exceeding `max_length` (estimated at ~4 characters per token) trigger warnings before truncation

#### 3. **Model Inference**

The tokenized inputs are passed through the reward model:
```python
with torch.no_grad():
    logits = model(**inputs).logits  # Shape: [batch_size, 1]
    rewards = logits.squeeze(-1)     # Extract scalar rewards
```

#### 4. **Batch Processing**

* Samples are processed in batches of size `batch_size`
* Progress is tracked with a progress bar showing completion status
* Remaining samples in the final incomplete batch are processed separately

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.45
}
```

- `id`: Unique identifier for the data sample, extracted from the input data's `id` field. If the input lacks an `id` field, an empty string is used
- `score`: The harmlessness reward score assigned by the model. Higher values (e.g., > 0) indicate harmless, safe responses. Lower values (e.g., < 0) indicate potentially harmful or unsafe responses. The score range depends on the model's training distribution

### Citation

This reward model was developed and utilized for multi-objective alignment research, particularly focusing on harmlessness and helpfulness alignment objectives:

```bibtex
@article{yang2024rewards,
  title={Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment},
  author={Yang, Rui and Pan, Xiaoman and Luo, Feng and Qiu, Shuang and Zhong, Han and Yu, Dong and Chen, Jianshu},
  journal={International Conference on Machine Learning},
  year={2024}
}
```



---

## Gpt2HelpfulScorer

### Overview

The **GPT2 Helpful Scorer** is a model-based evaluation tool designed to assess the helpfulness of instruction-response pairs in supervised fine-tuning (SFT) data. This scorer leverages a GPT2-large reward model specifically trained on the Anthropic/hh-rlhf helpful dataset to evaluate how useful and informative responses are in addressing user instructions. The model achieves an accuracy of **0.72621** on the test set, matching the performance of larger models while maintaining computational efficiency.

### Metric Definition:

* **Definition:** 

  Given an instruction-response pair `(Q, A)`, the helpfulness reward model assigns a scalar reward score representing the usefulness and informativeness of the response in addressing the instruction.

* **Explanation:** 

  The reward score quantifies how helpful and informative a response is relative to the given instruction:
  
  * A **higher reward score** indicates the response is **more helpful and informative**, providing relevant, accurate, and actionable information that effectively addresses the user's query.
  * A **lower reward score** suggests the response is **less helpful**, potentially vague, irrelevant, or failing to adequately address the instruction.

* **Formulation:** 

  Following the Anthropic/hh-rlhf dataset format:
  ```
  Input: "\n\nHuman: {instruction}\n\nAssistant:"
  Output: {response}
  ```

### YAML Configuration

```yaml
name: Gpt2HelpfulScorer
model: Ray2333/gpt2-large-helpful-reward_model
batch_size: 8
max_length: 1024
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"Gpt2HelpfulScorer"` | Identifier for the scorer. Must be set to `Gpt2HelpfulScorer` |
| `model` | string | `"Ray2333/gpt2-large-helpful-reward_model"` | Path or name of the helpful reward model from Hugging Face Hub. You can specify alternative GPT2-based helpfulness models trained on similar datasets |
| `batch_size` | integer | `8` | Number of samples to process simultaneously during inference. Adjust based on GPU memory availability |
| `max_length` | integer | `1024` | Maximum sequence length (in tokens) for tokenization. Sequences exceeding this length will be truncated |

### Underlying Model

The scorer uses [**Ray2333/gpt2-large-helpful-reward_model**](https://huggingface.co/Ray2333/gpt2-large-helpful-reward_model), a GPT2-large model fine-tuned as a reward model on the **Anthropic/hh-rlhf helpful dataset**. 

#### Model Characteristics

* **Base Architecture:** GPT2-large (774M parameters)
* **Training Data:** Anthropic/hh-rlhf helpful subset
* **Task:** Binary sequence classification (helpfulness scoring)
* **Precision:** bfloat16 for efficient inference
* **Performance:** 72.6% accuracy on the helpful test set

**Important Note:** This reward model differs from other open-source reward models trained on the full Anthropic/hh-rlhf dataset, as it focuses exclusively on the helpful subset for specialized helpfulness evaluation.

### Scoring Process

1. **Input Formatting**: For each data item containing `instruction`, `input` (optional), and `output` fields:
   ```python
   # If input field exists:
   Q = "\n\nHuman: {instruction}\n{input}\n\nAssistant:"
   
   # If input field is empty:
   Q = "\n\nHuman: {instruction}\n\nAssistant:"
   
   A = {output}
   ```
   This formatting follows the Anthropic/hh-rlhf dataset convention, ensuring compatibility with the reward model's training format.

2. **Tokenization**: The instruction `Q` and response `A` are tokenized together with truncation enabled (`max_length` parameter), padding applied to create uniform batch sizes, and warning system for samples exceeding `max_length`

3. **Model Inference**: The tokenized inputs are passed through the reward model to extract scalar reward scores
   ```python
   with torch.no_grad():
       logits = model(**inputs).logits  # Shape: [batch_size, 1]
       rewards = logits.squeeze(-1)     # Extract scalar rewards
   ```

4. **Batch Processing**: Samples are processed in batches of size `batch_size` with progress tracking

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.12
}
```

- `id`: Unique identifier for the data sample, extracted from the input data's `id` field. If the input lacks an `id` field, an empty string is used
- `score`: The helpfulness reward score assigned by the model. Higher values (e.g., > 0) indicate helpful, informative responses that effectively address the instruction. Lower values (e.g., < 0) indicate less helpful responses that may be vague, irrelevant, or inadequate

### Citation

This reward model was developed and utilized for multi-objective alignment research, particularly focusing on harmlessness and helpfulness alignment objectives:

```bibtex
@article{yang2024rewards,
  title={Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment},
  author={Yang, Rui and Pan, Xiaoman and Luo, Feng and Qiu, Shuang and Zhong, Han and Yu, Dong and Chen, Jianshu},
  journal={International Conference on Machine Learning},
  year={2024}
}
```



---

## GraNdScorer

### Overview

The **GraNd Scorer** (Gradient Normed) is a model-based evaluation tool designed to measure the importance and informativeness of individual training examples for supervised fine-tuning (SFT) data. Introduced in the paper [Paul et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html), this method identifies valuable training examples by computing gradient norms early in the training process. The core insight is that examples producing larger gradient norms during early training phases tend to be more informative and crucial for model generalization.

### Metric Definition:

* **Definition:** The GraNd score is computed as the L2 norm of all parameter gradients after a single forward-backward pass on a data sample:
  
  ```
  GraNd(x) = ||∇θ L(x; θ)||₂
  ```
  
  where `L(x; θ)` is the loss for example `x` with model parameters `θ`, and `||·||₂` denotes the L2 norm.

* **Explanation:** This metric quantifies how much a single training example would change the model's parameters if used for training.
  
  * A **higher GraNd score** indicates that the example produces a large gradient, suggesting it contains **informative or challenging content** that can significantly improve the model.
  * A **lower GraNd score** suggests the example is **easy or redundant** relative to the current model state, contributing less to model improvement.

### YAML Configuration

```yaml
name: GraNdScorer
model: Qwen/Qwen3-8B
max_length: 2048
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"GraNdScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | Path to local model or HuggingFace model name for computing gradients. Can be any causal language model compatible with HuggingFace `AutoModelForCausalLM`. Falls back to `gpt2` if the specified model fails to load |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization. Controls the maximum number of tokens processed per example. Longer sequences may provide more accurate scores but require more memory |

### Underlying Model

The scorer uses causal language models (CLMs) to compute gradient norms. By default, it uses [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), but can work with any compatible language model.

The model is used in training mode to enable gradient computation, but no actual parameter updates are performed.

### Scoring Process

The GraNd scoring process follows these steps for each SFT data sample:

1. **Text Preparation**: Concatenate the `instruction`, `input` (if present), and `output` fields to form the complete text sequence.

2. **Tokenization**: 
   - Tokenize the instruction portion separately to determine its length
   - Tokenize the full text (instruction + output) for model input
   - Apply truncation at `max_length` if necessary

3. **Label Masking**: Create labels where:
   - Instruction tokens are set to `-100` (ignored in loss computation)
   - Output tokens retain their original token IDs (used for loss computation)
   - This ensures gradients are computed only with respect to the output/response generation

4. **Gradient Computation**:
   - Set model to training mode and zero existing gradients
   - Perform forward pass to compute the loss on output tokens
   - Perform backward pass to compute parameter gradients

5. **Gradient Norm Calculation**: Compute the L2 norm across all parameter gradients:
   ```
   gradient_norm = sqrt(Σᵢ ||∇θᵢ||₂²)
   ```
   where the sum is over all model parameters with computed gradients

6. **Cleanup**: Zero gradients and return model to evaluation mode

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 245.7382
}
```

- `id`: The unique identifier of the data sample (from the input data's `id` field)
- `score`: The computed GraNd score (gradient L2 norm) as a floating-point number. Higher values indicate more informative/important examples. Values are non-negative (L2 norms are always ≥ 0). Magnitude depends on model size and architecture

### Citation

```bibtex
@article{paul2021deep,
  title={Deep learning on a data diet: Finding important examples early in training},
  author={Paul, Mansheej and Ganguli, Surya and Dziugaite, Gintare Karolina},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={20596--20607},
  year={2021}
}
```


---

## HESScorer

### Overview

The **High-Entropy Sum (HES) Scorer** is a training-free metric designed to evaluate the quality and complexity of Chain-of-Thought (CoT) reasoning samples. Proposed in the paper ["Unified Data Selection for LLM Reasoning"](https://openreview.net/pdf?id=heVn5cNfje), HES addresses a critical limitation in traditional metrics: they perform coarse-grained, global evaluation that treats all tokens equally, diluting the signal from truly critical reasoning steps.

Unlike metrics such as average entropy or perplexity that average over all tokens, HES focuses exclusively on **high-entropy forking tokens**—the key decision points in the reasoning process where the model faces multiple plausible paths. By summing only the entropy of the top 0.5% highest-entropy tokens, HES captures the genuine complexity and learning value of reasoning samples while filtering out predictable, trivial content.

### Metric Definition:

* **Definition:** 
  
  HES is calculated by summing the entropy values of the top *p* percentile (default *p* = 0.5%) of tokens with the highest entropy within the completion (reasoning) part of a sample:
  
  ```
  HES = Σ H(token_i) for token_i ∈ Top_p%(entropies)
  ```
  
  where `H(token_i)` is the Shannon entropy (in bits) of the token probability distribution at position *i*.

* **Explanation:** 
  
  * A **higher HES score** indicates greater diversity and complexity of reasoning patterns at critical forking points, suggesting **higher learning value** and more challenging reasoning paths.
  * A **lower HES score** suggests fewer critical decision points or more deterministic reasoning, indicating **simpler or lower-quality** reasoning samples.
  
  The key insight is that in long CoT reasoning, truly difficult-to-predict tokens are in the minority. By focusing on these high-entropy tokens rather than averaging across all tokens, HES effectively identifies samples with substantial reasoning complexity.

### YAML Configuration

```yaml
name: HESScorer
model: Qwen/Qwen2.5-7B-Instruct
percentile_cutoff: 0.005
batch_size: 8
max_length: 4096
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"HESScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen2.5-7B-Instruct"` | HuggingFace model path for the causal language model used to compute token entropies |
| `percentile_cutoff` | float | `0.005` | Fraction of highest-entropy tokens to include in HES calculation (0.5% by default) |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |
| `max_length` | integer | `4096` | Maximum sequence length for tokenization (prompt + completion) |

### Underlying Model

The HES Scorer requires a **causal language model** to compute token-level entropy from logits. Unlike task-specific scorers, HES is **model-agnostic** and can work with any decoder-only language model from the Hugging Face transformers library that supports `AutoModelForCausalLM`.

The scorer automatically handles model loading, tokenization, and entropy computation. No additional training or fine-tuning of the model is required—HES is entirely **training-free**.

### Scoring Process

The HES scoring pipeline operates through the following steps:

#### 1. **Input Preparation**
   * Each sample consists of `instruction`, `input`, and `output` fields
   * The prompt is constructed as: `instruction + "\n" + input` (or just `instruction` if input is empty)
   * The completion is the `output` field containing the CoT reasoning

#### 2. **Tokenization and Length Management**
   * Concatenate prompt + completion to form the full text
   * Tokenize separately to determine prompt length (where completion starts)
   * Check total length against `max_length`
   * If exceeding limit, truncate from the end (completion is truncated, prompt preserved when possible)
   * Flag truncated samples for later analysis

#### 3. **Model Forward Pass**
   * Process samples in batches for efficiency
   * Use left-padding for batch processing (right-padding for single samples)
   * Run model inference with `torch.no_grad()` to obtain logits for each token position
   * Output shape: `(batch_size, sequence_length, vocabulary_size)`

#### 4. **Entropy Calculation**
   * For each token position in the **completion span only** (excluding prompt):
     * Extract logits for predicting that token
     * Convert to probabilities: `p = softmax(logits)`
     * Compute Shannon entropy: `H = -Σ(p_i × log₂(p_i))`
     * Add small epsilon (1e-9) to prevent log(0)
   * Collect all token entropies for the completion

#### 5. **HES Aggregation**
   * Compute the percentile threshold: `threshold = percentile(entropies, (1 - p) × 100)`
   * Select tokens with entropy ≥ threshold
   * If no tokens selected (edge case), use the single maximum entropy token
   * Sum the selected high-entropy values to get the final HES score

#### 6. **Result Collection**
   * Record HES score, completion token count, entropy threshold, and truncation flag
   * Clear CUDA cache between batches to manage memory

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 245.73,
  "completion_token_length": 512,
  "entropy_threshold": 8.42,
  "truncated": false
}
```

- `id`: Unique identifier from the input sample
- `score`: HES score (sum of entropies for top percentile highest-entropy tokens). Higher values indicate more complex reasoning
- `completion_token_length`: Number of tokens in the completion (reasoning output)
- `entropy_threshold`: Entropy cutoff value at the (1-p) percentile for selecting high-entropy tokens
- `truncated`: Flag indicating whether the sample exceeded max_length and was truncated

### Citation

```bibtex
@misc{anonymous2025unified,
  title={Unified Data Selection for {LLM} Re{ASON}ing},
  author={Anonymous},
  year={2025},
  note={Manuscript under review. Submitted to ICLR 2025},
  howpublished={\url{https://openreview.net/forum?id=heVn5cNfje}}
}
```


---

## IFDScorer

### Overview

Instruction Following Difficulty (IFD) is a metric introduced to quantify the complexity of instruction following in large language models (LLMs). It compares the model's perplexity when generating outputs with and without instructional context. By calculating the ratio of conditional perplexity to direct perplexity, IFD measures how much an instruction affects the difficulty of generating the corresponding output. Higher IFD scores (> 1) indicate that the instruction increases generation difficulty, suggesting the instruction-output pair is harder to follow or poorly aligned.

### Metric Definition:

* **Definition:** 

  Given an instruction Q and answer/output A, the IFD score is computed as:
  
  **IFD(Q,A) = perplexity(A|Q) / perplexity(A)**
  
  Where:
  - `perplexity(A)` = perplexity of generating answer A without any instruction context
  - `perplexity(A|Q)` = perplexity of generating answer A given instruction Q

* **Explanation:** This metric compares the **conditional perplexity** (model's difficulty in generating the output when instruction is provided) with the **direct perplexity** (model's difficulty in generating the output without instruction). The ratio indicates how much the instruction affects answer generation:

  * A **higher IFD score (> 1)** suggests the model has **more difficulty** generating the answer when given the instruction, indicating the instruction-answer pair is **harder to follow** or poorly aligned.
  * A **lower IFD score (< 1)** indicates the instruction provides **guidance that reduces perplexity**, suggesting the instruction-answer pair is **easier to follow** and well-aligned.
  * An **IFD score ≈ 1** suggests the instruction provides **minimal effect** on the answer generation difficulty.

### YAML Configuration

```yaml
name: IFDScorer
model: Qwen/Qwen2.5-3B-Instruct
max_length: 2048
batch_size: 1
template: "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"
template_no_input: "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"IFDScorer"` | Identifier for the scorer |
| `model` | string | `"openai-community/gpt2"` | HuggingFace model path or local directory for the language model (fallback: `openai-community/gpt2`) |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `1` | Number of samples to process in parallel per forward pass |
| `template` | string | See above | Template for formatting instruction with input field |
| `template_no_input` | string | See above | Template for formatting instruction without input field |


### Underlying Model

The scorer can use any causal language model from HuggingFace (e.g., [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)) to compute:

  * **Direct Answer Perplexity:** s(A) - perplexity of generating the answer without instruction
  * **Conditioned Answer Perplexity:** s(A|Q) - perplexity of generating the answer given instruction

### Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` field)
   - Input (from `input` field if present)
   - Output/Answer (from `output` field)

2. **Prompt Formatting**: Construct the prompt using the configured template:
   - If `input` is provided: use `template` with both instruction and input
   - If `input` is empty: use `template_no_input` with only instruction

3. **Direct Answer Perplexity Calculation**: Compute perplexity for the output `A` **without any instructional context** by:
   - Tokenizing the output alone
   - Computing cross-entropy loss for all output tokens
   - Calculating perplexity as `exp(loss)`
   - This measures the model's **inherent ability** to generate the output independently

4. **Conditioned Answer Perplexity Calculation**: Compute perplexity for the output `A` **given the full instruction context** by:
   - Concatenating prompt and output: `prompt + output`
   - Masking prompt tokens (setting labels to -100)
   - Computing cross-entropy loss only for output tokens
   - Calculating perplexity as `exp(loss)`
   - This measures how well the model can generate the output **when provided with instructions**

5. **IFD Score Computation**: The final IFD metric is calculated as the perplexity ratio:
   - `IFD(Q,A) = perplexity(A|Q) / perplexity(A)`
   - Higher ratio indicates the instruction **hinders** answer generation (more difficult)
   - Lower ratio indicates the instruction **helps** answer generation (easier)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.25
}
```

- `id`: Unique identifier for the input sample (from the input data's `id` field)
- `score`: IFD score calculated as `perplexity(A|Q) / perplexity(A)`

### Citation

```bibtex
@inproceedings{li2024quantity,
  title={From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning},
  author={Li, Ming and Zhang, Yong and Li, Zhitao and Chen, Jiuhai and Chen, Lichang and Cheng, Ning and Wang, Jianzong and Zhou, Tianyi and Xiao, Jing},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={7595--7628},
  year={2024}
}
```


---

## InfOrmScorer

### Overview

The **INF Outcome Reward Model (INF-ORM) Scorer** is a state-of-the-art reward model-based evaluation tool designed to assess the alignment quality and preference of supervised fine-tuning (SFT) data. Built upon the Llama-3.1-70B-Instruct architecture and trained with the INF-ORM-Preference-Magnitude-80K dataset, this scorer leverages advanced techniques including scaled Bradley-Terry loss, modified score head architecture, and model merging to achieve top-tier performance.

As of December 2024, INF-ORM-Llama3.1-70B ranks **first** on the [RewardBench leaderboard](https://huggingface.co/spaces/allenai/reward-bench) with a score of 95.1, demonstrating exceptional capability in evaluating chat responses, safety, and reasoning tasks.

Unlike traditional heuristic or synthetic scoring methods, the INF-ORM Scorer provides a learning-based evaluation that captures nuanced preferences in instruction-response pairs, making it particularly suitable for data curation, quality assessment, and alignment training in large language models.

### Metric Definition:

* **Definition:** 

  Given an instruction-response pair, the INF-ORM Scorer assigns a scalar reward score representing the overall quality and alignment of the response in the context of the given instruction.

* **Explanation:** 

  The reward score reflects the model's learned preferences from large-scale human-annotated preference data:
  
  * A **higher INF-ORM score** indicates that the response is well-aligned, helpful, accurate, and preferred according to human evaluation standards.
  * A **lower INF-ORM score** suggests deficiencies in quality, coherence, safety, or instruction-following behavior.

### YAML Configuration

```yaml
name: InfOrmScorer
model: infly/INF-ORM-Llama3.1-70B
batch_size: 32
max_length: 4096
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"InfOrmScorer"` | Identifier for the scorer |
| `model` | string | `"infly/INF-ORM-Llama3.1-70B"` | HuggingFace model path or local path to the reward model checkpoint |
| `batch_size` | integer | `32` | Number of samples to process in parallel during evaluation |
| `max_length` | integer | `4096` | Maximum token length for input sequences (sequences exceeding this will be truncated) |

### Underlying Model

The scorer uses [**infly/INF-ORM-Llama3.1-70B**](https://huggingface.co/infly/INF-ORM-Llama3.1-70B), a 70-billion parameter reward model that ranks **first** on the RewardBench leaderboard (December 2024) with a score of 95.1. 

**Key Features:**
- **Base Architecture:** Llama-3.1-70B-Instruct with a modified two-layer MLP score head (Linear → ReLU → Linear)
- **Training Data:** INF-ORM-Preference-Magnitude-80K dataset with magnitude annotations (1-3 scale)
- **Training Method:** Scaled Bradley-Terry loss with magnitude weighting and model merging techniques

**Performance Benchmarks (RewardBench, December 2024):**

| Rank | Model | Score | Chat | Chat Hard | Safety | Reasoning |
|------|-------|-------|------|-----------|--------|-----------|
| **1** | **INF-ORM-Llama3.1-70B** | **95.1** | **96.6** | **91.0** | **93.6** | **99.1** |
| 2 | Skywork-Reward-Gemma-2-27B-v0.2 | 94.3 | 96.1 | 89.9 | 93.0 | 98.1 |
| 3 | Llama-3.1-Nemotron-70B-Reward | 94.1 | 97.5 | 85.7 | 95.1 | 98.1 |

### Scoring Process

1. **Input Formatting**: For each data sample containing `instruction`, optional `input`, and `output` fields, the scorer constructs a conversation in chat template format:
   ```python
   [
       {"role": "user", "content": "<instruction> + <input>"},
       {"role": "assistant", "content": "<output>"}
   ]
   ```

2. **Tokenization**: Conversations are tokenized using the model's chat template with `apply_chat_template()`. Sequences exceeding `max_length` are truncated with a warning.

3. **Batch Processing**: Multiple samples are processed simultaneously based on `batch_size` configuration for efficient GPU utilization.

4. **Model Inference**: The model processes tokenized inputs through:
   - Llama-3.1-70B transformer layers to extract contextual representations
   - Modified two-layer MLP score head (Linear → ReLU → Linear) to produce logits
   - Extraction of the final sequence position's logit as the reward score

5. **Score Extraction**: Raw logits are converted to float values and returned as reward scores for each sample.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.96875
}
```

- `id`: Unique identifier for the data sample (preserved from input dataset's `id` field)
- `score`: The reward score assigned by the INF-ORM model (higher values indicate better quality and alignment)

### Citation

```bibtex
@misc{INF-ORM-Llama3.1-70B, 
      url={https://huggingface.co/infly/INF-ORM-Llama3.1-70B},
      title={INF-ORM-Llama3.1-70B},
      year={2024},
      author={Minghao Yang, Chao Qu, Xiaoyu Tan}
}
```



---

## InstagScorer

### Overview

The **InsTag Scorer** (Instruction Tagging Scorer) is a model-based evaluation tool designed to measure the *complexity* of instructions in supervised fine-tuning (SFT) datasets. Proposed in the paper [Lu et al., 2023](https://arxiv.org/abs/2308.07074), InsTag provides a fine-grained semantic and intention-based approach to analyzing user queries by identifying and tagging diverse instruction characteristics.

InsTag addresses the challenge of quantifying instruction diversity and complexity—two critical factors for successful SFT datasets. By leveraging a specialized tagging model, this scorer automatically identifies semantic tags and user intentions within instructions, providing an objective measure of instruction complexity based on the number and variety of identified tags.

### Metric Definition:

* **Definition:** 

  The InsTag Complexity score is calculated as the **number of semantic and intention tags** identified in a given instruction.

* **Explanation:** 
  
  The scorer prompts a fine-tuned language model to analyze the instruction and generate a JSON output containing identified tags with explanations. The complexity is quantified as:
  
  * **List output:** If the model returns a list of tag objects, the score equals the length of the list.
  * **Single dict output:** If the model returns a single tag dictionary, the score is 1.
  * **Parse failure:** If the output cannot be parsed as valid JSON, the score is 0.

* **Interpretation:**
  
  * A **higher InsTag Complexity score** indicates that the instruction contains multiple semantic dimensions or user intentions, suggesting **greater complexity and diversity**.
  * A **lower score** (0 or 1) suggests the instruction is **simple and uni-dimensional**.

### YAML Configuration

```yaml
name: InstagScorer
model: OFA-Sys/InsTagger
max_new_tokens: 512
batch_size: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"InstagScorer"` | Identifier for the scorer |
| `model` | string | `"OFA-Sys/InsTagger"` | Path to the local model or Hugging Face model identifier for instruction tagging |
| `max_new_tokens` | integer | `512` | Maximum number of new tokens to generate for tag outputs (valid range: 1-2047) |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |


### Underlying Model

The scorer uses [**OFA-Sys/InsTagger**](https://huggingface.co/OFA-Sys/InsTagger), a causal language model specifically fine-tuned for instruction tagging tasks. The model is trained to identify and categorize fine-grained semantic tags and user intentions across a comprehensive taxonomy of 6.6K tags covering diverse query types.

### Scoring Process

1. **Prompt Construction**: For each instruction, the scorer constructs a structured prompt following a chat template:
   ```
   A chat between a curious user and an artificial intelligence assistant. 
   The assistant gives helpful, detailed, and polite answers to the human's questions.
   
   USER: Please identify tags of user intentions in the following user query and 
   provide an explanation for each tag. Please respond in the JSON format 
   {"tag": str, "explanation": str}.
   User query: [INSTRUCTION]
   
   ASSISTANT:
   ```
   If the data item contains both `instruction` and `input` fields, they are concatenated as: `instruction + "\n" + input`.

2. **Batch Tokenization**: Instructions are tokenized with left-padding (required for causal language models). Input length is limited to `2048 - max_new_tokens` to reserve space for generation.

3. **Tag Generation**: The model generates tags in JSON format through greedy decoding with `max_new_tokens` parameter controlling output length.

4. **Score Calculation**: The generated JSON output is parsed to compute the complexity score:
   - **Valid JSON list:** `score = len(json_output)` (number of tags)
   - **Valid JSON dict:** `score = 1` (single tag)
   - **Invalid JSON:** `score = 0` (parsing failure)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3
}
```

- `id`: Unique identifier for the data sample (extracted from input data's `id` field)
- `score`: InsTag Complexity score representing the number of semantic/intention tags identified in the instruction (range: 0 to N, where higher values indicate more complex instructions)

### Citation

```bibtex
@article{lu2023instag,
  title={\# instag: Instruction tagging for analyzing supervised fine-tuning of large language models},
  author={Lu, Keming and Yuan, Hongyi and Yuan, Zheng and Lin, Runji and Lin, Junyang and Tan, Chuanqi and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.07074},
  year={2023}
}
```



---

## MIWVScorer

### Overview

The **MIWV (Model Instruction Weakness Value) Scorer** is a model-based evaluation tool designed to quantify the importance of instruction data in enhancing a language model's capabilities. Proposed in the paper [Jiang et al., 2025](https://arxiv.org/abs/2511.07074), this method identifies the most beneficial data for instruction tuning by measuring the discrepancies in the model's responses when using In-Context Learning (ICL). 

The key insight behind MIWV is that high-quality instruction data should maximize the performance gains for a given LLM during instruction tuning. Rather than focusing solely on data quality scores, MIWV evaluates how much a specific instruction sample can help improve the model's weakness areas by comparing its performance with and without relevant contextual examples.

### Metric Definition:

* **Definition:** 
  
  MIWV(x) = Loss_one-shot(x) - Loss_zero-shot(x)
  
  where:
  - `Loss_zero-shot(x)` is the cross-entropy loss of generating the output given only the instruction
  - `Loss_one-shot(x)` is the cross-entropy loss of generating the output with the most similar sample as an ICL example
  - The most similar sample is determined by embedding similarity using a specified distance metric

* **Explanation:** This metric quantifies how much the model struggles with an instruction even when provided with a relevant example:
  
  * A **higher MIWV score** (positive value) indicates that the model performs **worse** with the ICL example than without it, suggesting this is a **weakness area** where the model needs improvement. Such data is **more valuable** for instruction tuning.
  * A **lower MIWV score** (negative or close to zero) suggests the ICL example helps the model, indicating the instruction is already within the model's capabilities and may be **less critical** for training.

### YAML Configuration

```yaml
name: MIWVScorer
model: /path/to/your/model
embedding_path: /path/to/embeddings.npy
batch_size: 8
max_length: 2048
distance_metric: cosine
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"MIWVScorer"` | Identifier for the scorer |
| `model` | string | (required) | Path to the causal language model used for loss computation. Can be any HuggingFace-compatible model path |
| `embedding_path` | string | (required) | Path to a `.npy` file containing precomputed embeddings for all samples in the dataset. The embedding array shape should be `(num_samples, embedding_dim)`, and the order must match the dataset order |
| `batch_size` | integer | `8` | Number of samples to process simultaneously. Larger values speed up computation but require more GPU memory |
| `max_length` | integer | `2048` | Maximum token length for input sequences. Sequences exceeding this length will be truncated with a warning |
| `distance_metric` | string | `"cosine"` | Metric used to find the most similar sample for ICL. Available options: `cosine` (cosine distance), `euclidean` (Euclidean distance), `squared_euclidean` (squared Euclidean distance), `manhattan` (Manhattan distance) |

### Underlying Model

The MIWVScorer can work with **any causal language model** that supports the HuggingFace `AutoModelForCausalLM` interface. The choice of model depends on your specific use case and instruction tuning target.

Moreover, MIWVScorer requires **pre-computed embeddings** that generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for MIWVScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

```bash
python data_scorer/model_based/utils/embed.py \
    --embedder_model /path/to/embedding/model \
    --input_path /path/to/your/dataset.jsonl \
    --output_path /path/to/output/embeddings.npy \
    --fields instruction input \
    --max_tokens 32768 \
    --tokenize_batch_size 16384 \
    --embed_batch_size 16384
```

#### Script Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--embedder_model` | string | `Qwen/Qwen3-Embedding-8B` | Path or name of the vLLM model for computing embeddings (task=embed) |
| `--input_path` | string | *required* | Path to the input JSONL file containing your dataset |
| `--output_path` | string | *required* | Path to save the output `.npy` embedding file |
| `--fields` | list | `["instruction", "input", "output"]` | Field names to extract from JSONL and concatenate with newlines. Specify multiple fields to combine |
| `--max_tokens` | int | `32768` | Maximum number of tokens allowed per text; texts exceeding this will be truncated |
| `--tokenize_batch_size` | int | `16384` | Batch size for tokenization (encode_batch). Adjust based on memory |
| `--embed_batch_size` | int | `16384` | Batch size for embedding computation. Adjust based on GPU/memory |
| `--truncate_report_path` | string | `""` | Optional: Write line numbers of truncated samples to this text file |

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the MIWVScorer configuration.

### Scoring Process

1. **Embedding-Based Similarity Computation:**
   - Load precomputed embeddings for all samples in the dataset
   - For each sample, compute distances to all other samples using the specified distance metric
   - Identify the most similar sample (minimum distance, excluding self)

2. **Text Construction:**
   - **Zero-shot format:** `User: {instruction}\nAssistant: {output}`
   - **One-shot format:** 
     ```
     User: {similar_instruction}
     Assistant: {similar_output}
     User: {instruction}
     Assistant: {output}
     ```

3. **Loss Computation:**
   - Tokenize both zero-shot and one-shot texts with the target model
   - Compute cross-entropy loss only on the output tokens (prompt tokens are masked with -100)
   - Use batch processing for efficiency

4. **MIWV Calculation:**
   - MIWV = one-shot loss - zero-shot loss
   - Positive scores indicate data that exposes model weaknesses

5. **Result Aggregation:**
   - Store MIWV score along with metadata about the most similar sample

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.2345,
  "most_similar_idx": 42,
  "most_similar_id": "sample_042"
}
```

- `id`: Unique identifier of the evaluated sample. Extracted from the `id` field in the input data, or uses the sample index if not present
- `score`: The MIWV score (float). Higher values indicate more important data for instruction tuning. Positive values suggest the model struggles even with ICL examples (weakness area)
- `most_similar_idx`: Integer index of the most similar sample in the dataset that was used as the ICL example
- `most_similar_id`: The unique identifier of the most similar sample (for traceability and debugging)

### Citation

```bibtex
@article{jiang2025importance,
  title={Importance-Aware Data Selection for Efficient LLM Instruction Tuning},
  author={Jiang, Tingyu and Li, Shen and Song, Yiyao and Zhang, Lan and Zhu, Hualei and Zhao, Yuan and Xu, Xiaohang and Taura, Kenjiro and Wang, Hao Henry},
  journal={arXiv preprint arXiv:2511.07074},
  year={2025}
}
```



---

## NormLossScorer

### Overview

The **NormLoss Scorer** is a model-based evaluation tool that assesses SFT data quality through the lens of **compression efficiency**. Inspired by the finding that "compression represents intelligence linearly" ([Huang et al., 2024](https://arxiv.org/abs/2404.09937)), this scorer leverages language models as data compressors to measure the complexity and quality of instruction-response pairs. The underlying principle is that better data exhibits predictable patterns that can be efficiently compressed by language models, while lower-quality or overly complex data results in higher compression costs.

By computing the **normalized cross-entropy loss** (in bits per token), NormLoss Scorer provides an unsupervised, model-based metric that correlates with data quality and model learning efficiency. This approach is particularly useful for identifying high-quality training samples that align well with the model's learned representations.

### Metric Definition:

* **Definition:** 

```
NormLoss = (1 / N) × Σ -log₂ P(token_i | context)
```

where `N` is the number of tokens in the sequence, and `P(token_i | context)` is the model's predicted probability for each token given its context.

* **Explanation:** This metric measures the **average number of bits** required to encode each token in the text using the language model as a compressor. It reflects how well the model can predict and compress the given text:
  
  * A **lower NormLoss score** indicates the text is **more compressible** and aligns better with the model's learned patterns, suggesting **higher quality** or **better fit** with the model's knowledge distribution.
  * A **higher NormLoss score** suggests the text is **less compressible**, containing more surprising or unpredictable content relative to the model's expectations, which may indicate **lower quality** or **higher complexity**.

* **Key Advantages:**
  
  * **Information-theoretic foundation:** Rooted in the principle that compression and prediction are equivalent - better compression implies better understanding
  * **Unsupervised metric:** No need for labeled data or human annotations
  * **Model-aligned:** Measures data quality from the perspective of the specific model's learned distribution

### YAML Configuration

```yaml
name: NormLossScorer
model: meta-llama/Llama-3.1-8B
max_length: 2048
batch_size: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"NormLossScorer"` | Identifier for the scorer |
| `model` | string | `"meta-llama/Llama-3.1-8B"` | HuggingFace model path or local path for the causal language model used to compute cross-entropy loss |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization. Sequences longer than this will be truncated |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |

### Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute cross-entropy loss. By default, it uses **meta-llama/Llama-3.1-8B**, but can be configured to use any autoregressive language model.

### Scoring Process

1. **Input Processing**: For each data sample, the scorer concatenates:
   - Instruction (from `instruction` field)
   - Optional input text (from `input` field if present)
   - Response (from `output` field)
   
   Format: `text = instruction + '\n' + [input + '\n'] + output`

2. **Tokenization**: The concatenated text is tokenized with padding and truncation at `max_length`

3. **Forward Pass**: Compute token-level log probabilities through the causal language model

4. **Cross-Entropy Computation**: Calculate loss for each valid token as:
   `loss_i = -log P(token_i | token_1, ..., token_{i-1})`

5. **Normalization**: Average the loss over valid tokens and convert to bits per token:
   `NormLoss = (Σ loss_i / N) / ln(2)`

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.456
}
```

- `id`: Unique identifier for the data sample (from the `id` field in input data, or empty string if not present)
- `score`: The computed NormLoss value representing normalized cross-entropy in bits per token
  - **Lower values** indicate better compression and potentially higher quality
  - **Higher values** indicate poorer compression and potentially lower quality or higher complexity

### Citation

```bibtex
@article{shum2025predictive,
  title={Predictive data selection: The data that predicts is the data that teaches},
  author={Shum, Kashun and Huang, Yuzhen and Zou, Hongjian and Ding, Qi and Liao, Yixuan and Chen, Xiaoxin and Liu, Qian and He, Junxian},
  journal={arXiv preprint arXiv:2503.00808},
  year={2025}
}
```


---

## NuclearNormScorer

### Overview

The **Nuclear Norm Scorer** is a gradient-based evaluation tool designed to assess the quality of instruction-following and reasoning data through spectral analysis of layer-wise gradients. This scorer computes the nuclear norm (sum of singular values) of gradient matrices derived from attention layer parameters (Q, K, V, O) during model fine-tuning, providing insights into gradient stability and data quality.

Based on the research paper ["How Instruction and Reasoning Data shape Post-Training: Data Quality through the Lens of Layer-wise Gradients"](https://arxiv.org/abs/2504.10766), this method reveals that higher-quality data typically exhibits lower nuclear norms, indicating more stable and focused gradient updates. The nuclear norm serves as a complementary metric to effective rank, capturing the magnitude-weighted diversity of gradient singular values.

### Metric Definition:

* **Definition:** 
  
  The Nuclear Norm is computed through singular value decomposition (SVD) of gradient matrices:
  
  ```
  Nuclear_Norm = Σ(σ_i)
  
  where σ_i are singular values from SVD
  ```
  
  Where:
  - `σ_i` represents the i-th singular value from SVD of the gradient matrix
  - The nuclear norm is the sum of all singular values
  - Also known as the trace norm or Schatten 1-norm

* **Explanation:** Nuclear Norm quantifies the magnitude and complexity of gradient updates:
  
  * A **lower Nuclear Norm** indicates that gradients have **smaller magnitudes** and are **more concentrated**, suggesting **higher data quality** and more stable learning dynamics. This implies the training sample induces focused, efficient gradient updates.
  * A **higher Nuclear Norm** suggests that gradients have **larger magnitudes** or are **more dispersed** across dimensions, potentially indicating **noisier training signals** or less efficient learning patterns.
  
  The metric is computed separately for Query (Q), Key (K), Value (V), and Output (O) projection matrices in attention layers. Compared to effective rank which measures dimensionality, nuclear norm captures the overall scale of gradient updates, providing complementary insights into training dynamics.

### YAML Configuration

```yaml
name: NuclearNormScorer
model: Qwen/Qwen3-8B
max_length: 2048
start_layer_index: 16
num_layers: 4
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"NuclearNormScorer"` | Scorer identifier, must be set to `NuclearNormScorer` |
| `model` | string | `"Qwen/Qwen3-8B"` | Path to the language model, either a local directory or a HuggingFace model ID. The model is used to compute gradients during forward and backward passes. If loading fails, defaults to `gpt2` as fallback. |
| `max_length` | integer | `2048` | Maximum token length for input sequences. Text exceeding this length will be truncated with a warning message. Recommended values: 1024-4096 depending on GPU memory and model context length. |
| `start_layer_index` | integer/None | `None` | The starting index of transformer layers to analyze (0-indexed). If `None`, only the last layer is analyzed. If specified, layers from `start_layer_index` to `start_layer_index + num_layers - 1` are analyzed. Must be within `[0, total_layers - 1]`. |
| `num_layers` | integer | `1` | Number of consecutive layers to analyze starting from `start_layer_index`. Must be a positive integer. If `start_layer_index` is `None`, this parameter is ignored. The final nuclear norm scores are averaged across all specified layers. |

**Note:** The combination of `start_layer_index` and `num_layers` allows flexible analysis of specific layer ranges. For example, to analyze the last 4 layers of a 32-layer model, set `start_layer_index: 28` and `num_layers: 4`.

### Underlying Model

The Nuclear Norm Scorer requires a **causal language model** for gradient computation. By default, it uses `Qwen/Qwen3-8B`

The scorer is architecture-agnostic and supports various transformer models including:
- GPT-style models (GPT-2, GPT-Neo, etc.)
- LLaMA family models
- Qwen family models
- GPT-NeoX architecture models

The model is used in training mode to compute gradients but **is not updated**—gradients are computed solely for analysis purposes. The scorer automatically detects the model architecture and locates attention layer parameters (Q, K, V, O projections) accordingly.

### Scoring Process

1. **Data Preparation:**
   - Concatenate the `instruction`, `input` (if present), and `output` fields into a single text sequence
   - Text format: `instruction + " " + output` (with input appended to instruction if provided)
   - Check original text length before truncation and display detailed warning if truncation occurs, including item ID and token counts

2. **Tokenization:**
   - Tokenize the concatenated text using the model's tokenizer
   - Apply padding and truncation to `max_length`
   - Set labels equal to input_ids for language modeling loss computation
   - **Label Masking:** Set instruction portion labels to `-100` to compute loss only on the output portion, focusing gradient analysis on response generation

3. **Forward Pass:**
   - Set the model to training mode
   - Zero out any existing gradients
   - Perform forward pass to compute the language modeling loss (only on output tokens due to label masking)

4. **Backward Pass:**
   - Compute gradients via backpropagation: `loss.backward()`
   - Gradients are accumulated in the `.grad` attribute of each parameter

5. **Layer Selection:**
   - Retrieve all transformer layers from the model
   - Determine target layers based on `start_layer_index` and `num_layers`:
     - If `start_layer_index` is `None`: Use only the **last layer**
     - Otherwise: Use layers `[start_layer_index : start_layer_index + num_layers]`

6. **Gradient Extraction:**
   - For each target layer, extract attention module parameters:
     - **Standard format:** Separate Q, K, V, O projection weights
     - **GPT-2 format:** Combined QKV in `c_attn` (split by dimension), separate O in `c_proj`
   - Retrieve gradient matrices from `.weight.grad` of each projection

7. **Nuclear Norm Computation (for each Q, K, V, O gradient matrix):**
   - **Step 7.1:** Reshape gradient matrix to 2D if necessary
   - **Step 7.2:** Perform Singular Value Decomposition (SVD): `U, S, Vh = torch.linalg.svd(grad_matrix)`
   - **Step 7.3:** Compute Nuclear Norm as the sum of all singular values: `Nuclear_Norm = sum(S)`
   - **Note:** Unlike effective rank which normalizes and computes entropy, nuclear norm directly sums the singular values, capturing both magnitude and diversity

8. **Aggregation:**
   - Collect nuclear norms for Q, K, V, O across all target layers
   - Compute average nuclear norm for each projection type
   - Return four averaged scores: Q_NuclearNorm, K_NuclearNorm, V_NuclearNorm, O_NuclearNorm

9. **Memory Cleanup:**
   - Clear gradients, delete temporary tensors
   - Set model back to evaluation mode
   - Empty CUDA cache if using GPU to prevent memory leaks (automatically handled)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_identifier",
  "Q_NuclearNorm": 245.67,
  "K_NuclearNorm": 198.34,
  "V_NuclearNorm": 289.12,
  "O_NuclearNorm": 223.45
}
```

- `id`: The unique identifier of the sample, extracted from the input data's `id` field
- `Q_NuclearNorm`: Nuclear norm of gradients from Query projection matrices, averaged across all specified layers. Lower values indicate more focused query attention updates.
- `K_NuclearNorm`: Nuclear norm of gradients from Key projection matrices, averaged across all specified layers. Reflects the magnitude of key representation updates.
- `V_NuclearNorm`: Nuclear norm of gradients from Value projection matrices, averaged across all specified layers. Indicates the scale of value transformation updates.
- `O_NuclearNorm`: Nuclear norm of gradients from Output projection matrices, averaged across all specified layers. Represents the overall magnitude of attention output transformations.

**Interpretation Guidelines:**

- **Higher-quality data** generally correlates with **lower nuclear norms** across all four projection types, indicating more stable and efficient gradient updates
- **Lower nuclear norms** suggest that gradients are concentrated and focused, reflecting better training stability
- **Higher nuclear norms** may indicate noisier or less efficient learning patterns
- **Zero values** indicate computation failures (missing gradients or SVD errors) and should be investigated
- **Relative comparisons** between samples are more meaningful than absolute values, as nuclear norms depend on model architecture, layer depth, and parameter dimensions
- **Complementary to Effective Rank:** While effective rank measures gradient dimensionality (higher is better for quality), nuclear norm measures gradient magnitude (lower is better for quality)

**Example Output:**

```json
[
  {
    "id": 1,
    "Q_NuclearNorm": 156.34,
    "K_NuclearNorm": 142.89,
    "V_NuclearNorm": 178.23,
    "O_NuclearNorm": 165.91
  },
  {
    "id": 2,
    "Q_NuclearNorm": 387.56,
    "K_NuclearNorm": 412.34,
    "V_NuclearNorm": 456.78,
    "O_NuclearNorm": 398.12
  }
]
```

### Citation

```bibtex
@article{li2025instruction,
  title={How instruction and reasoning data shape post-training: Data quality through the lens of layer-wise gradients},
  author={Li, Ming and Li, Yanhong and Li, Ziyue and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2504.10766},
  year={2025}
}
```



---

## PPLScorer

### Overview

The **PPL (Perplexity) Scorer** is a model-based evaluation tool that measures how well a language model predicts a given text sequence. Perplexity is a fundamental metric in natural language processing that quantifies the uncertainty of a language model when generating text. A lower perplexity score indicates that the model finds the text more predictable and natural, while a higher score suggests the text is more surprising or difficult for the model to predict.

This scorer is particularly useful for assessing the quality and naturalness of SFT (Supervised Fine-Tuning) data, as it can identify samples that are either too simple (very low perplexity) or potentially noisy/anomalous (very high perplexity).

### Metric Definition:

* **Definition:** PPL = exp(L), where L is the average cross-entropy loss per token.
  
  Formally: PPL(x) = exp(-1/N × Σ log P(x_i | x_<i))
  
  where N is the number of tokens, and P(x_i | x_<i) is the probability of token x_i given all previous tokens.

* **Explanation:** Perplexity measures how "surprised" a language model is by a given text sequence.
  
  * A **lower PPL score** indicates the text is more predictable and natural according to the model, suggesting the sample follows common patterns and is of **higher quality**.
  * A **higher PPL score** suggests the text is less predictable, which could indicate either complex/diverse content or potentially **noisy or low-quality data**.

### YAML Configuration

```yaml
name: PPLScorer
model: meta-llama/Llama-3.1-8B
max_length: 2048
batch_size: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"PPLScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path or local path to a causal language model compatible with `AutoModelForCausalLM` |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization. Sequences longer than this will be truncated |
| `batch_size` | integer | `8` | Number of samples to process simultaneously in each batch |

### Underlying Model

The PPL Scorer can work with **any causal language model** compatible with Hugging Face's `AutoModelForCausalLM` API. 

The choice of model depends on your evaluation needs. Larger models generally provide more accurate perplexity estimates but require more computational resources.

### Scoring Process

The PPL Scorer follows this pipeline to evaluate each sample:

1. **Text Concatenation**: For each data item, the scorer concatenates the `instruction`, `input` (if present), and `output` fields with newlines, forming a complete text sequence.

2. **Tokenization with Padding**: The text is tokenized using the model's tokenizer with:
   * **Padding enabled**: Ensures all sequences in a batch have the same length
   * **Truncation enabled**: Limits sequences to `max_length`
   * **Pad token handling**: If the model lacks a pad token (e.g., Llama, Qwen), the EOS token is used as the pad token

3. **Label Preparation**: The scorer creates labels by:
   * Cloning the `input_ids` as labels
   * Replacing all padding token positions with `-100` to ensure they are ignored during loss calculation

4. **Loss Calculation**: For each sample individually:
   * The model computes the cross-entropy loss between predicted and actual tokens
   * Only non-padding tokens contribute to the loss (padding tokens with label `-100` are ignored)
   * The loss represents the average negative log-likelihood per valid token

5. **Perplexity Computation**: The final perplexity is calculated as: **PPL = exp(loss)**

6. **Batch Processing**: Samples are processed in batches according to the configured `batch_size` for efficiency.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 12.45
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` exists, an empty string is used
- `score`: The perplexity score for this sample. Lower values indicate the text is more predictable and natural to the model, while higher values suggest greater difficulty or unexpectedness

### Citation

```bibtex
@article{jelinek1977perplexity,
  title={Perplexity—a measure of the difficulty of speech recognition tasks},
  author={Jelinek, Fred and Mercer, Robert L and Bahl, Lalit R and Baker, James K},
  journal={The Journal of the Acoustical Society of America},
  volume={62},
  number={S1},
  pages={S63--S63},
  year={1977},
  publisher={Acoustical Society of America}
}
```



---

## ProfessionalismScorer

### Overview

The **Professionalism Scorer** is a model-based evaluation tool designed to assess the **degree of expertise and technical knowledge** required to understand SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates the professional depth and technical complexity of text content, measuring how much specialized knowledge is needed to comprehend the material.

The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable professionalism assessments on a continuous 0-5 scale, enabling fine-grained assessment of the technical complexity and expertise requirements of instruction-following datasets.

### Metric Definition:

* **Definition:** 
  
  A continuous score ranging from 0 to 5 that quantifies the degree of expertise and technical knowledge required to understand the text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** The Professionalism metric evaluates the technical depth and specialized knowledge requirements of text content based on:
  1. **Technical Complexity** - The level of domain-specific terminology and concepts
  2. **Expertise Requirements** - The background knowledge needed to comprehend the content
  3. **Accessibility** - How specialized or general-audience the content is

* **Score Interpretation:**
  * **0.0-0.9**: No technical knowledge required; content accessible to everyone
  * **1.0-1.9**: Minimal technical knowledge needed; simple content (e.g., children's books, basic tutorials)
  * **2.0-2.9**: Basic professional knowledge required; general-audience content (e.g., popular science articles)
  * **3.0-3.9**: Moderate expertise needed; intermediate complexity (e.g., advanced articles, technical documentation)
  * **4.0-4.9**: Significant professional knowledge required; complex content (e.g., academic papers, technical reports)
  * **5.0**: Advanced expertise essential; highly professional content (e.g., advanced research papers, patents)

### YAML Configuration

```yaml
name: ProfessionalismScorer
model: opendatalab/meta-rater-professionalism-rating
batch_size: 16
max_length: 8192
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ProfessionalismScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-professionalism-rating"` | HuggingFace model path or local path to the professionalism rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_length` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated with warnings) |

### Underlying Model

The scorer uses [opendatalab/meta-rater-professionalism-rating](https://huggingface.co/opendatalab/meta-rater-professionalism-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 91.57% F1 score, 93.78% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

### Scoring Process

1. **Text Concatenation**: For each data sample, the scorer concatenates the fields in the following order:
   - `content = instruction + "\n" + input + "\n" + output`
   - If the `input` field is empty, it uses: `content = instruction + "\n" + output`

2. **Tokenization**: The concatenated text is tokenized using the ModernBERT tokenizer with left-side padding for batch processing, truncation at `max_length` (default 8,192 tokens), and explicit truncation warnings for texts exceeding max length

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5)

4. **Score Calculation**: Instead of using the argmax prediction, the scorer computes a continuous score using weighted probability averaging: `score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]` where `P(class_i)` is the softmax probability of class `i`

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch to optimize memory usage

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.42
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the professionalism score. Higher values indicate greater technical depth and expertise requirements

### Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```


---

## QuRateScorer

### Overview

The **QuRating Scorer** is a model-based evaluation tool designed to assess the quality of training data for language models across multiple dimensions. Proposed in the paper [Wettig et al., 2024](https://arxiv.org/abs/2402.09739), QuRating provides a principled approach to data selection by evaluating text quality along several interpretable axes. Unlike single-dimensional quality metrics, QuRating enables fine-grained analysis of what makes training data valuable, allowing practitioners to curate datasets based on specific quality attributes relevant to their use case.

The scorer employs a specialized sequence classification model trained to predict multiple quality dimensions simultaneously, providing both chunk-level and document-level quality assessments.

### Metric Definition:

* **Definition:** 

  QuRating evaluates text across multiple quality dimensions, each scored independently. The default dimensions include:
  
  - **writing_style**: Assesses the clarity, coherence, and stylistic quality of the text
  - **required_expertise**: Measures the level of domain knowledge or expertise reflected in the content
  - **facts_and_trivia**: Evaluates the presence and accuracy of factual information
  - **educational_value**: Quantifies how informative and instructive the content is

* **Explanation:** 

  Each dimension is scored independently, and the final score represents a weighted average across text chunks (weighted by token count).
  
  * **Higher scores** in each dimension indicate stronger presence of that quality attribute
  * **Lower scores** suggest deficiency in that particular aspect
  * The multi-dimensional nature allows for targeted data selection based on specific quality requirements

### YAML Configuration

```yaml
name: QuRateScorer
model: princeton-nlp/QuRater-1.3B
labels:
  - writing_style
  - required_expertise
  - facts_and_trivia
  - educational_value
chunk_size: 512
batch_size: 8
device_batch_size: 16
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"QuRateScorer"` | Identifier for the scorer |
| `model` | string | `"princeton-nlp/QuRater-1.3B"` | Path to the QuRater model (local or HuggingFace model ID) |
| `labels` | list | `["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]` | List of quality dimensions to evaluate |
| `chunk_size` | integer | `512` | Maximum tokens per chunk for processing |
| `batch_size` | integer | `8` | Number of complete samples to process simultaneously |
| `device_batch_size` | integer | `16` | Number of text chunks to process per GPU forward pass |

### Underlying Model

The scorer uses [princeton-nlp/QuRater-1.3B](https://huggingface.co/princeton-nlp/QuRater-1.3B), a 1.3B parameter sequence classification model specifically trained for multi-dimensional data quality assessment. The model is based on transformer architecture and outputs scores for four quality dimensions simultaneously.

The model accepts tokenized text input and produces logits for each quality dimension, which are then used as quality scores. It processes text in chunks to handle documents of arbitrary length efficiently.

### Scoring Process

1. **Text Construction**: For each data sample, the scorer constructs the full text by concatenating:
   - Instruction
   - Input (if present)
   - Output/Response

2. **Tokenization and Chunking**: The concatenated text is tokenized without special tokens and split into chunks of size `chunk_size` (default 512 tokens)

3. **Chunk Scoring**: Each chunk is processed through the QuRater model:
   - Chunks are batched by similar length for efficiency
   - The model outputs logits for each quality dimension
   - Scores are computed on CPU for memory efficiency

4. **Aggregation**: Final scores are computed using a weighted average across all chunks:
   - Each chunk's score is weighted by its token count
   - Formula: `score = Σ(chunk_score_i × token_count_i) / Σ(token_count_i)`

5. **Result Compilation**: The scorer produces aggregate scores (weighted average per dimension) and per-chunk scores for detailed analysis

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "length": 1234,
  "num_chunks": 3,
  "writing_style_score": 4.56,
  "writing_style_chunks": [4.2, 4.8, 4.7],
  "required_expertise_score": 3.21,
  "required_expertise_chunks": [3.1, 3.3, 3.2],
  "facts_and_trivia_score": 5.12,
  "facts_and_trivia_chunks": [5.0, 5.2, 5.2],
  "educational_value_score": 4.89,
  "educational_value_chunks": [4.7, 5.0, 5.0]
}
```

- `id`: Unique identifier for the sample (from input data)
- `length`: Total number of tokens in the processed text
- `num_chunks`: Number of chunks the text was split into
- `{label}_score`: Weighted average score for the quality dimension (float)
- `{label}_chunks`: List of scores for each chunk in that dimension (list of floats)

*Note: The `{label}` placeholder is replaced with each configured label name (e.g., `writing_style`, `required_expertise`, etc.)*

### Citation

```bibtex
@inproceedings{wettig2024qurating,
  title={{QuRating}: Selecting High-Quality Data for Training Language Models},
  author={Wettig, Alexander and Gupta, Aatmik and Malik, Saumya and Chen, Danqi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```


---

## RMDeBERTaScorer

### Overview

The **RMDeBERTa Scorer** is a reward model-based evaluation tool designed to assess the quality of instruction-response pairs in supervised fine-tuning (SFT) data. It leverages the OpenAssistant reward model trained on human feedback to predict how preferable a generated response is for a given instruction. The model was trained to distinguish between better and worse responses as judged by humans, making it valuable for quality assessment, QA model evaluation, and toxic response detection.

The scorer utilizes a DeBERTa-v3-large architecture fine-tuned on multiple high-quality preference datasets, achieving strong performance across diverse evaluation benchmarks. It provides a scalar reward score indicating response quality, where higher scores suggest better alignment with human preferences.

### Metric Definition:

* **Definition:** 

  Given an instruction-response pair (Q, A), the reward model outputs a scalar score representing the expected human preference for the response A given instruction Q. The score is computed as `score = model(Q, A).logits[0]`.

* **Explanation:** The reward score quantifies how well a response aligns with human judgment of quality, helpfulness, and appropriateness.
  
  * A **higher reward score** indicates that the response is more likely to be preferred by humans, suggesting better quality, helpfulness, and instruction-following.
  * A **lower reward score** suggests that the response may be less helpful, less accurate, or potentially problematic.

### YAML Configuration

```yaml
name: RMDeBERTaScorer
model: OpenAssistant/reward-model-deberta-v3-large-v2
max_length: 512
batch_size: 32
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"RMDeBERTaScorer"` | Scorer identifier used for logging and output organization |
| `model` | string | `"OpenAssistant/reward-model-deberta-v3-large-v2"` | HuggingFace model path or local model directory |
| `max_length` | integer | `512` | Maximum token length for input sequences (instruction + response). Sequences exceeding this length will be truncated |
| `batch_size` | integer | `32` | Number of samples to process simultaneously. Larger values increase throughput but require more GPU memory |


### Underlying Model

The scorer uses [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2), a reward model based on the DeBERTa-v3-large architecture. This model was trained on diverse human feedback datasets including:

* **webgpt_comparisons**: Web-based question answering comparisons
* **summarize_from_feedback**: Summarization preference data
* **synthetic-instruct-gptj-pairwise**: Synthetic instruction-following pairs
* **anthropic_hh-rlhf**: Anthropic's helpfulness and harmlessness dataset

The model achieves strong validation accuracy across benchmarks, with 61.57% on WebGPT, 71.47% on Summary tasks, 99.88% on SyntheticGPT, and 69.25% on Anthropic RLHF datasets.

### Scoring Process

The RMDeBERTa Scorer evaluates instruction-response pairs through the following pipeline:

1. **Input Preparation:**
   * For each data item, the scorer extracts the `instruction` field and optional `input` field
   * If `input` exists, it concatenates with instruction: `question = instruction + "\n" + input`
   * Otherwise, uses instruction alone: `question = instruction`
   * The `output` field serves as the response to be evaluated

2. **Batch Tokenization:**
   * Question-answer pairs are tokenized together using the model's tokenizer
   * Sequences are padded to uniform length and truncated to `max_length` if necessary
   * A warning is issued for truncated samples with their IDs logged

3. **Model Inference:**
   * Tokenized inputs are passed through the reward model in batches
   * The model outputs logits representing preference scores
   * Scores are extracted as `logits[:, 0]` and converted to CPU tensors

4. **Score Extraction:**
   * Raw scores are returned as floating-point values
   * No normalization or scaling is applied to preserve interpretability

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.347
}
```

- `id`: Unique identifier for the data sample, taken from the input data's `id` field. If not present, defaults to `"index_{i}"` where `i` is the sample position
- `score`: Floating-point reward score indicating response quality. Higher values indicate better quality and stronger alignment with human preferences

### Citation

```bibtex
@misc{openassistant_debertav3_rewardmodel_v2,
  title        = {OpenAssistant Reward Model - DeBERTa-v3-large-v2},
  author       = {{OpenAssistant Team}},
  howpublished = {\url{https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2}},
  note         = {Accessed: 2025-03-xx},
  year         = {2023}
}
```


---

## ReadabilityScorer

### Overview

The **Readability Scorer** is a model-based evaluation tool designed to assess the **ease of understanding and text clarity** of SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates how easily text can be read and comprehended by assessing factors such as clarity, coherence, vocabulary complexity, sentence structure, and grammatical correctness. The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable readability assessments on a continuous 0-5 scale.

### Metric Definition:

* **Definition:** 

  A continuous score ranging from 0 to 5 that quantifies the ease of understanding and clarity of text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** 

  The Readability metric evaluates text comprehensibility based on multiple linguistic dimensions:
  1. **Clarity and Coherence** - Logical flow and organization of ideas
  2. **Vocabulary Complexity** - Appropriate use of words and terminology
  3. **Sentence Structure** - Sentence length, complexity, and construction
  4. **Grammar and Spelling** - Correctness of language use

* **Score Interpretation:**
  
  * **0.0-0.9**: Absolutely not readable; severe clarity issues, incoherent content
  * **1.0-1.9**: Somewhat readable but contains significant clarity or coherence issues, complex vocabulary, or numerous errors
  * **2.0-2.9**: Generally clear and coherent with occasional grammar, spelling errors, or convoluted structures
  * **3.0-3.9**: Clear and coherent for the most part, using appropriate vocabulary with minor grammar/spelling issues
  * **4.0-4.9**: Very clear and coherent with very few or no errors, proper punctuation, and easy-to-follow structures
  * **5.0**: Outstanding clarity and coherence, effective communication with minimal errors that don't interfere with understanding

### YAML Configuration

```yaml
name: ReadabilityScorer
model: opendatalab/meta-rater-readability-rating
batch_size: 16
max_length: 8192
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ReadabilityScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-readability-rating"` | HuggingFace model path or local path to the readability rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_length` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated with warnings) |

### Underlying Model

The scorer uses [opendatalab/meta-rater-readability-rating](https://huggingface.co/opendatalab/meta-rater-readability-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 87.47% F1 score, 94.13% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

### Scoring Process

1. **Text Concatenation**: For each data sample, the scorer concatenates the fields in the following order:
   - `content = instruction + "\n" + input + "\n" + output`
   - If the `input` field is empty, it uses: `content = instruction + "\n" + output`

2. **Tokenization**: The concatenated text is tokenized using the ModernBERT tokenizer with left-side padding, truncation at `max_length` (default 8,192 tokens), and automatic padding for batch inference

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5)

4. **Score Calculation**: The scorer computes a **continuous score** using weighted probability averaging:
   - `score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]`
   - where `P(class_i)` is the softmax probability of class `i`

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.67
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the readability score (higher values indicate better readability and clearer communication)

### Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```


---

## ReasoningScorer

### Overview

The **Reasoning Scorer** is a model-based evaluation tool designed to assess the **complexity of logical thinking and analytical reasoning** in SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates the depth and sophistication of reasoning processes demonstrated in text content, measuring how complex the logical thinking, inference, and analytical capabilities required are. The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable reasoning complexity assessments on a continuous 0-5 scale.

### Metric Definition:

* **Definition:** 

  A continuous score ranging from 0 to 5 that quantifies the complexity of logical thinking and analytical reasoning required in the text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** The Reasoning metric evaluates the sophistication of cognitive processes based on:
  1. **Logical Complexity** - The depth of logical reasoning and inference chains
  2. **Analytical Depth** - The sophistication of analysis and critical thinking
  3. **Deductive/Inductive Reasoning** - The quality of reasoning patterns and conclusions
  4. **Problem-Solving Sophistication** - The complexity of problem decomposition and solution strategies

* **Score Interpretation:**
  * **0.0-0.9**: No discernible reasoning; simple statements or facts with no logical connections
  * **1.0-1.9**: Minimal reasoning with significant logical flaws or very basic cause-effect relationships
  * **2.0-2.9**: Basic reasoning with some inconsistencies; simple logical connections present
  * **3.0-3.9**: Moderate reasoning with occasional lapses; clear logical structure with some complexity
  * **4.0-4.9**: Strong reasoning with minor issues; sophisticated analysis and well-developed arguments
  * **5.0**: Exceptional reasoning with flawless logic; complex multi-step reasoning and deep analytical insights

### YAML Configuration

```yaml
name: ReasoningScorer
model: opendatalab/meta-rater-reasoning-rating
batch_size: 16
max_length: 8192
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ReasoningScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-reasoning-rating"` | HuggingFace model path or local path to the reasoning rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_length` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated with warnings) |

### Underlying Model

The scorer uses [opendatalab/meta-rater-reasoning-rating](https://huggingface.co/opendatalab/meta-rater-reasoning-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 91.57% F1 score, 93.78% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

### Scoring Process

The Reasoning Scorer follows a systematic evaluation pipeline:

1. **Text Concatenation**: For each data sample, the scorer concatenates the fields in the following order:
   ```
   content = instruction + "\n" + input + "\n" + output
   ```
   If the `input` field is empty, it uses:
   ```
   content = instruction + "\n" + output
   ```

2. **Tokenization**: The concatenated text is tokenized using the ModernBERT tokenizer with:
   * Left-side padding for batch processing
   * Truncation at `max_length` (default 8,192 tokens)
   * Automatic padding for batch inference
   * Explicit truncation warnings for texts exceeding max length

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5).

4. **Score Calculation**: Instead of using the argmax prediction, the scorer computes a **continuous score** using weighted probability averaging:
   ```
   score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]
   ```
   where `P(class_i)` is the softmax probability of class `i`.

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch to optimize memory usage.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.85
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the reasoning complexity score (higher values indicate more sophisticated logical thinking and analytical depth)

### Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```


---

## SelectitModelScorer

### Overview

The **SelectIT Model Scorer** is an ensemble-based evaluation tool that leverages **model-level uncertainty** to assess the quality of instruction-tuning (SFT) data. Inspired by the SelectIT framework from [Liu et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b130a5691815f550977e331f8bec08ae-Paper-Conference.pdf), this scorer focuses on combining predictions from **multiple language models** to produce robust quality assessments.

By employing multiple models with different capacities or architectures, this approach captures diverse perspectives on data quality. The key insight is that **disagreement between models** can reveal ambiguous or low-quality samples, while **consensus across models** indicates high-quality, well-formed instruction-response pairs. This model-level ensemble strategy is particularly effective at filtering out edge cases that might fool a single model.

### Metric Definition:

* **Definition:**
  
  This scorer employs a **multi-model ensemble approach** to evaluate instruction-tuning data quality through three hierarchical levels:
  
  1. **Token-level Scoring**: Each model computes a probability distribution over rating tokens (1-5) to generate an expected score for each prompt.
  2. **Sentence-level Aggregation**: For each model, k different rating prompts are applied to the same sample, and scores are aggregated with a standard deviation penalty to ensure prompt-level consistency.
  3. **Model-level Ensemble** (Core Innovation): Multiple language models independently evaluate the same data, and their scores are combined using weighted averaging to leverage model diversity and reduce individual model biases.

* **Explanation:** 
  
  The final SelectIT Model score reflects the **consensus across multiple models**, weighted by their respective importance. This ensemble strategy provides several benefits:
  
  * A **higher score** (closer to 5) indicates **strong agreement among models** that the sample is high-quality, well-formed, and suitable for instruction tuning.
  * A **lower score** (closer to 1) suggests **model disagreement or consistently low ratings**, indicating the sample may be ambiguous, poorly written, or problematic.
  * The multi-model design ensures that scores are **robust to individual model biases** and capture diverse quality perspectives.

* **Key Advantages:**
  
  * **Model-level ensemble**: Primary strength lies in combining multiple models (e.g., LLaMA-2-7B + LLaMA-2-13B) to capture complementary quality signals and reduce evaluation variance
  * **Weighted aggregation**: Flexible weighting scheme allows prioritizing certain models based on their reliability or domain expertise
  * **Bias mitigation**: Model diversity helps filter out false positives/negatives that might occur with single-model evaluation
  * **Probabilistic foundation**: Token-level probability distributions provide fine-grained quality estimates rather than binary judgments

### YAML Configuration

```yaml
name: SelectitModelScorer
models:
  - meta-llama/Llama-2-7b-hf
  - meta-llama/Llama-2-13b-hf
model_weights: [0.5, 0.5]
rp_file: scorers/SelectIT_rating_prompt.txt
k: 5
alpha: 0.2
max_length: 512
batch_size: 16
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SelectitModelScorer"` | Identifier for the scorer |
| `models` | list | - | List of model paths for ensemble scoring. Multiple models enable model-level uncertainty assessment |
| `model_weights` | list | Equal weights | Weights for each model in the ensemble. Must match the length of `models` |
| `rp_file` | string | `"scorers/SelectIT_rating_prompt.txt"` | Path to the rating prompt template file containing k different prompts for sentence-level reflection |
| `k` | integer | `5` | Number of different rating prompt templates to use per sample for sentence-level uncertainty assessment |
| `alpha` | float | `0.2` | Standard deviation penalty coefficient. Higher values penalize inconsistent ratings more strongly |
| `max_length` | integer | `512` | Maximum token length for input sequences. Should be between 1 and 2048 |
| `batch_size` | integer | `16` | Number of samples to process in each batch |

### Underlying Model

The SelectIT Scorer is model-agnostic and can work with **any autoregressive language model** that supports causal language modeling.

Users can specify any compatible models from Hugging Face Hub or local paths. The scorer automatically handles tokenization and uses the model's native tokenizer. For ensemble evaluation, it is recommended to use 2-3 models of different sizes or architectures to capture diverse perspectives on data quality.

### Scoring Process

The SelectIT scoring process follows a three-level uncertainty framework:

#### 1. Token-level Self-Reflection

For each instruction-response pair and each rating prompt:
- The model generates logits for the next token after the prompt
- Probabilities for rating tokens ("1", "2", "3", "4", "5") are extracted
- These probabilities are normalized to form a rating distribution
- Expected score is calculated: `score = Σ(rating × probability)`

#### 2. Sentence-level Self-Reflection

For each sample:
- **k** different rating prompts are applied (typically k=5)
- Each prompt produces a token-level score
- Mean (μ) and standard deviation (σ) of k scores are computed
- Sentence-level score is adjusted: `sentence_score = μ / (1 + alpha × σ)`
- This penalizes samples with inconsistent ratings across different prompts

#### 3. Model-level Self-Reflection

When multiple models are used:
- Each model independently produces sentence-level scores
- Scores are combined using weighted averaging: `final_score = Σ(model_weight_i × score_i)`
- This leverages uncertainty between different models for robust evaluation

#### Complete Pipeline

```
Input: instruction-response pair
  ↓
Generate k rating prompts
  ↓
For each model:
  ↓
  For each prompt:
    → Compute token probabilities
    → Calculate expected score
  ↓
  Aggregate with std penalty (sentence-level)
  ↓
Weighted average across models (model-level)
  ↓
Output: Final SelectIT score (1-5)
```

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_001",
  "score": 4.23
}
```

- `id`: The unique identifier of the data sample (from the original dataset)
- `score`: The final SelectIT score, ranging from 1.0 to 5.0
  - **1.0-2.0**: Low quality data, likely inconsistent or problematic
  - **2.0-3.0**: Below average quality
  - **3.0-4.0**: Good quality data
  - **4.0-5.0**: High quality, consistent instruction-tuning data

Scores are continuous values, not discrete integers, reflecting the probabilistic nature of the scoring process.

### Citation

```bibtex
@article{liu2024selectit,
  title={SelectIT: Selective instruction tuning for LLMs via uncertainty-aware self-reflection},
  author={Liu, Liangxin and Liu, Xuebo and Wong, Derek F and Li, Dongfang and Wang, Ziyi and Hu, Baotian and Zhang, Min},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={97800--97825},
  year={2024}
}
```



---

## SelectitSentenceScorer

### Overview

The **SelectIT Sentence Scorer** is a model-based evaluation tool that implements the sentence-level uncertainty assessment from the SelectIT framework proposed in [Liu et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b130a5691815f550977e331f8bec08ae-Paper-Conference.pdf). Unlike the full SelectIT scorer which uses multiple models (model-level uncertainty), this scorer focuses on leveraging **token-level** and **sentence-level** uncertainty within a single foundation model to assess instruction-tuning data quality.

This approach evaluates the same instruction-response pair using multiple different rating prompts and measures the consistency of ratings across prompts. High-quality data samples receive consistent ratings across different prompt formulations, while low-quality or ambiguous samples exhibit high variance. The scorer is lightweight, requiring only a single model while still providing robust quality estimates through prompt-based uncertainty assessment.

### Metric Definition:

* **Definition:** 

  Given an instruction-response pair, the scorer computes:
  
  1. **Token-level Self-Reflection**: For each rating prompt j, compute the expected rating as `score_j = Σ(rating_i × P(rating_i))` where P(rating_i) is the probability the model assigns to rating i ∈ {1,2,3,4,5}
  2. **Sentence-level Self-Reflection**: Apply k different rating prompts to the same sample, compute mean μ and standard deviation σ of the k token-level scores
  3. **Normalized Score**: `final_score = μ / (1 + alpha × σ)` where alpha controls the penalty for inconsistency

* **Explanation:** This metric measures data quality through **rating consistency across multiple prompt formulations**:
  
  * A **higher score** (closer to 5) indicates **high-quality data** with consistent ratings across multiple prompts, suggesting the model confidently evaluates the sample positively.
  * A **lower score** (closer to 1) suggests **low-quality data** with inconsistent ratings, indicating ambiguity or problems in the instruction-response pair.
  * A **score close to 3** represents neutral or average quality, often assigned to samples with high variance.

* **Key Advantages:**
  
  * **Prompt-based uncertainty**: Measures quality through consistency rather than a single rating
  * **Lightweight**: Requires only one model (unlike model-level ensemble methods)
  * **Interpretable**: Scores range from 1-5 with clear quality implications
  * **Robust**: Inconsistency penalty prevents high scores for ambiguous samples

### YAML Configuration

```yaml
name: SelectitSentenceScorer
model: meta-llama/Llama-2-7b-hf
rp_file: scorers/SelectIT_rating_prompt.txt
k: 5
alpha: 0.2
max_length: 512
batch_size: 16
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SelectitSentenceScorer"` | Identifier for the scorer |
| `model` | string | `"princeton-nlp/QuRater-1.3B"` | HuggingFace model path or local path to the causal language model used for rating |
| `rp_file` | string | `"scorers/SelectIT_rating_prompt.txt"` | Path to the rating prompt template file containing k different prompts for sentence-level reflection |
| `k` | integer | `5` | Number of different rating prompt templates to use per sample for uncertainty assessment |
| `alpha` | float | `0.2` | Standard deviation penalty coefficient; higher values apply stronger penalties for inconsistency |
| `max_length` | integer | `512` | Maximum sequence length for tokenization (must be between 1 and 2048) |
| `batch_size` | integer | `16` | Number of samples to process in parallel per forward pass |

### Underlying Model

The SelectIT Sentence Scorer is **model-agnostic** and can work with any autoregressive causal language model. The implementation provides flexibility in model selection:

- **Default fallback**: `Qwen/Qwen3-8B` - A specialized model for rating instruction-tuning data

The scorer automatically handles different tokenizers and dynamically identifies the token IDs corresponding to ratings "1" through "5" for each model. Users can specify any compatible model from Hugging Face Hub or provide a local model path.

### Scoring Process

1. **Input Processing**: For each data sample, extract instruction and response fields

2. **Prompt Generation**: Create k different rating prompts by combining each rating template with the instruction-response pair:
   ```
   [Rating Prompt Template]
   Instruction: [instruction text]
   Response: [response text]
   The answer is:
   ```

3. **Token-level Probability Extraction**: For each of the k prompts:
   - Feed the prompt to the model
   - Extract logits for the next token position after "The answer is:"
   - Compute softmax probabilities over rating tokens {"1", "2", "3", "4", "5"}
   - Calculate expected score: `score_j = 1×P("1") + 2×P("2") + 3×P("3") + 4×P("4") + 5×P("5")`

4. **Sentence-level Aggregation**: Collect k token-level scores and compute:
   - Mean: `μ = (1/k) × Σ score_j`
   - Standard deviation: `σ = sqrt((1/k) × Σ(score_j - μ)²)`

5. **Score Computation**: Apply consistency penalty to obtain final score:
   ```
   final_score = μ / (1 + alpha × σ)
   ```
   Low σ (consistent ratings) → minimal penalty; High σ (inconsistent ratings) → substantial penalty

6. **Batch Processing**: All k×n prompts (for n samples) are processed together in batches for efficiency

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.15
}
```

- `id`: Unique identifier of the data sample from the original dataset
- `score`: Final SelectIT Sentence score, a continuous value ranging from 1.0 to 5.0
  - **1.0-2.0**: Low quality - inconsistent ratings or poor content
  - **2.0-3.0**: Below average quality - moderate inconsistency
  - **3.0-4.0**: Good quality - consistent ratings with minor variations
  - **4.0-5.0**: High quality - highly consistent ratings across all prompts

### Citation

```bibtex
@article{liu2024selectit,
  title={SelectIT: Selective instruction tuning for LLMs via uncertainty-aware self-reflection},
  author={Liu, Liangxin and Liu, Xuebo and Wong, Derek F and Li, Dongfang and Wang, Ziyi and Hu, Baotian and Zhang, Min},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={97800--97825},
  year={2024}
}
```



---

## SelectitTokenScorer

### Overview

The **SelectIT Token Scorer** is a model-based evaluation tool that implements the token-level and sentence-level uncertainty assessment from the SelectIT framework proposed in [Liu et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b130a5691815f550977e331f8bec08ae-Paper-Conference.pdf). This scorer provides a lightweight yet effective approach to assess instruction-tuning data quality using a single foundation model.

**By default (k=1)**, this scorer **focuses on token-level uncertainty** by computing the probability distribution over rating tokens (1-5) for a single rating prompt. This provides an efficient quality estimate based on the model's confidence in its ratings.

**When k>1**, the scorer additionally introduces **prompt-level uncertainty** assessment by evaluating the same sample with multiple different rating prompts and measuring consistency across prompts (sentence-level self-reflection). This provides more robust quality estimates by penalizing samples where the model provides inconsistent ratings across different prompt formulations, but increases computational cost.


### Metric Definition:

* **Definition:**
  
  **When k=1 (default - token-level uncertainty only):**
  1. **Token-level Self-Reflection**: The foundation model rates the instruction-response pair from 1 to 5 by computing the probability distribution over rating tokens: `score = Σ(rating_i × P(rating_i))`
  2. **Final Score**: `final_score = score` (no consistency penalty applied)
  
  **When k>1 (token-level + prompt-level uncertainty):**
  1. **Token-level Self-Reflection**: For each of the k rating prompts, compute the token-level score: `score_j = Σ(rating_i × P(rating_i))`
  2. **Sentence-level Self-Reflection**: Measure uncertainty across different prompts via standard deviation to penalize inconsistent ratings
  3. **Final Score Calculation**: `final_score = μ / (1 + alpha × σ)`
  
  Where:
  - `μ` = mean of k token-level scores
  - `σ` = standard deviation of k token-level scores
  - `alpha` = penalty coefficient for inconsistency (default: 0.2)

* **Explanation:** 
  
  **With k=1**: The score purely reflects token-level uncertainty - how confident the model is in its rating based on the probability distribution over rating tokens (1-5). Higher scores indicate the model assigns high probability to high ratings.
  
  **With k>1**: The score additionally considers prompt-level consistency. 
  * A **higher score** (closer to 5) indicates **high-quality data** where the model consistently assigns high ratings across all prompts.
  * A **lower score** (closer to 1) suggests **low-quality data** with either low ratings or high inconsistency across prompts.
  * The standard deviation penalty (controlled by `alpha`) ensures that samples with high rating variance receive lower scores, promoting data consistency.

### YAML Configuration

```yaml
name: SelectitTokenScorer
model: meta-llama/Llama-3.1-8B
rp_file: scorers/SelectIT/rating_prompt.txt
k: 1
alpha: 0.2
max_length: 2048
batch_size: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SelectitTokenScorer"` | Identifier for the scorer |
| `model` | string | `"meta-llama/Llama-3.1-8B"` | Path to the foundation model (local or Hugging Face Hub). Can be any autoregressive causal language model |
| `rp_file` | string | `"scorers/SelectIT/rating_prompt.txt"` | Path to the rating prompt template file containing k different prompts for evaluation. Each line should contain one rating prompt template |
| `k` | integer | `1` | Number of different rating prompt templates to use per sample. **Default k=1 focuses on token-level uncertainty only**. Set k>1 to introduce prompt-level uncertainty assessment. Higher values provide more robust consistency estimates but increase computation |
| `alpha` | float | `0.2` | Standard deviation penalty coefficient. Controls how strongly inconsistency across prompts penalizes the final score when k>1. Higher values apply stronger penalties. **This parameter has no effect when k=1** |
| `max_length` | integer | `2048` | Maximum token length for input sequences. Longer sequences will be truncated |
| `batch_size` | integer | `8` | Number of samples to process in each batch. Larger batches improve throughput but require more GPU memory |

### Underlying Model

The SelectIT Token Scorer is **model-agnostic** and can work with any autoregressive causal language model. The implementation provides flexibility in model selection:

- **Default fallback**: `meta-llama/Llama-3.1-8B` - A capable foundation model optimized for instruction understanding

The scorer automatically handles different tokenizers and dynamically identifies the token IDs corresponding to ratings "1" through "5" for each model. Users can specify any compatible model from Hugging Face Hub or provide a local model path. The scorer will automatically fall back to the default model if the specified model fails to load.

### Scoring Process

The SelectIT Token scoring process operates differently depending on the k parameter:

**When k=1 (default)**: Focuses solely on **token-level uncertainty** by computing probability distribution over rating tokens for a single prompt.

**When k>1**: Integrates both **token-level** and **sentence-level (prompt-level) uncertainty** assessment.

#### 1. Token-level Self-Reflection

For each instruction-response pair and each of the k rating prompts (when k=1, only one prompt is used):

1. **Prompt Construction**: Combine rating prompt template with instruction and response
   ```
   [Rating Prompt Template]
   Instruction: [instruction text]
   Response: [response text]
   The answer is:
   ```

2. **Probability Extraction**: 
   - Tokenize and feed the prompt to the model
   - Extract logits for the next token position after the prompt
   - Compute softmax probabilities for rating tokens ("1", "2", "3", "4", "5")
   - Normalize to ensure probabilities sum to 1.0

3. **Expected Score Calculation**:
   ```
   token_score = Σ(rating × P(rating))
   = 1×P("1") + 2×P("2") + 3×P("3") + 4×P("4") + 5×P("5")
   ```

#### 2. Sentence-level Consistency Assessment (Only when k>1)

**When k=1**: This step is skipped. The final score equals the token-level score from the single prompt.

**When k>1**: After obtaining k token-level scores for each sample:

1. **Compute Statistics**:
   - Mean (μ): `μ = (1/k) × Σ token_score_j`
   - Standard deviation (σ): `σ = sqrt((1/k) × Σ(token_score_j - μ)²)`

2. **Apply Consistency Penalty**:
   ```
   final_score = μ / (1 + alpha × σ)
   ```
   - Consistent ratings (low σ) → minimal penalty → score ≈ μ
   - Inconsistent ratings (high σ) → substantial penalty → score < μ

#### Complete Pipeline

**When k=1 (token-level uncertainty only):**
```
Input: instruction-response pair
  ↓
Validate required fields (instruction, output)
  ↓
Generate single rating prompt
  ↓
Tokenize prompt
  ↓
Forward pass through model
  ↓
Extract token probabilities for ratings 1-5
  ↓
Normalize probability distribution
  ↓
Calculate expected score: final_score = Σ(rating × P(rating))
  ↓
Output: SelectIT Token score (1-5)
```

**When k>1 (token-level + prompt-level uncertainty):**
```
Input: instruction-response pair
  ↓
Validate required fields (instruction, output)
  ↓
Generate k different rating prompts
  ↓
Batch tokenize all prompts
  ↓
Single forward pass through model
  ↓
For each prompt:
  → Extract token probabilities for ratings 1-5
  → Normalize probability distribution
  → Calculate expected score
  ↓
Collect k token-level scores
  ↓
Compute mean (μ) and std (σ)
  ↓
Apply penalty: final_score = μ / (1 + alpha × σ)
  ↓
Output: SelectIT Token score (1-5)
```

#### Batch Processing

The scorer implements efficient batch processing:
- All k×n prompts (for n samples in a batch) are tokenized together using `padding="longest"`
- Single forward pass through the model for all prompts
- Truncation warnings are issued if prompts exceed `max_length`
- Results are grouped back by sample for final score computation
- Invalid samples (missing required fields) receive a default score of 3.0

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.28
}
```

- `id`: The unique identifier of the data sample (extracted from the original dataset's `"id"` field)
- `score`: The final SelectIT Token score, a continuous value ranging from 1.0 to 5.0
  - **1.0-2.0**: Low quality - low ratings or highly inconsistent across prompts
  - **2.0-3.0**: Below average quality - moderate ratings with some inconsistency
  - **3.0**: Default score assigned to invalid or unparseable data
  - **3.0-4.0**: Good quality - decent ratings with acceptable consistency
  - **4.0-5.0**: High quality - high ratings with strong consistency across all prompts

**Score Interpretation:**
- **When k=1**: Score magnitude purely reflects token-level uncertainty - the expected rating based on probability distribution over rating tokens
- **When k>1**: 
  - **Score magnitude** is determined by the average rating from the model across k prompts
  - **Consistency penalty**: High variance across prompts reduces the score, even if the average is high
  - **Robustness**: Using multiple prompts provides more stable estimates than single-prompt evaluation
- **Default handling**: Samples with missing required fields (`instruction` or `output`) receive a neutral score of 3.0

### Citation

```bibtex
@article{liu2024selectit,
  title={SelectIT: Selective instruction tuning for LLMs via uncertainty-aware self-reflection},
  author={Liu, Liangxin and Liu, Xuebo and Wong, Derek F and Li, Dongfang and Wang, Ziyi and Hu, Baotian and Zhang, Min},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={97800--97825},
  year={2024}
}
```



---

## SkyworkRewardScorer

### Overview

The **Skywork Reward Scorer** is a model-based evaluation tool that leverages the Skywork Reward Model, a large-scale reward model trained on 26 million high-quality preference pairs, designed to assess the alignment quality of supervised fine-tuning (SFT) data. Unlike heuristic or synthetic scoring strategies, the Skywork Reward Model is grounded in extensive human-LLM joint evaluations and sets a new standard for reward modeling. It is suitable for ranking, filtering, or curating SFT data for alignment training.

### Metric Definition:

* **Definition:** 

  Given an instruction-response pair, the reward scorer assigns a scalar reward score, representing how preferable or aligned the response is in the context of the instruction.

* **Explanation:**
  
  * A **higher Skywork Reward Score** indicates that the response is preferred by the reward model, demonstrating better quality, alignment, and task-following behavior.
  * A **lower score** suggests deficiencies in quality, alignment, or task-following behavior.
  * The reward model provides a **unified preference signal** trained on extensive human feedback data, making it more reliable than heuristic metrics.

### YAML Configuration 

```yaml
name: SkyworkRewardScorer
model: Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M
max_length: 4096
batch_size: 16
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SkyworkRewardScorer"` | Identifier for the scorer |
| `model` | string | `"Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M"` | HuggingFace model path for the reward model |
| `max_length` | integer | `4096` | Maximum sequence length for tokenization |
| `batch_size` | integer | `16` | Number of samples to process in parallel per forward pass |


### Underlying Model

The scorer uses the largest model from the Skywork-Reward-V2 series, [Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M). The model is trained on a vast corpus of human-LLM preference data, achieving state-of-the-art performance across benchmarks such as: RewardBench v1 & v2, RMB, RM-Bench, and JudgeBench.

### Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` and optional `input` fields)
   - Response (from `output` field)

2. **Prompt Construction**: The instruction and response are formatted according to the Skywork Reward Model's chat template

3. **Forward Pass**: The formatted conversation is passed through the reward model

4. **Score Extraction**: The model outputs a scalar reward score representing the preference strength for the response given the instruction

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_identifier",
  "score": 2.5678
}
```

- `id`: Unique identifier for the sample
- `score`: Reward score assigned by the model (higher values indicate better alignment and quality)

### Citation

```bibtex
@article{liu2025skywork,
  title={Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy},
  author = {Liu, Chris Yuhao and Zeng, Liang and Xiao, Yuzhen and He, Jujie and Liu, Jiacai and Wang, Chaojie and Yan, Rui and Shen, Wei and Zhang, Fuxiang and Xu, Jiacheng and Liu, Yang and Zhou, Yahui},
  journal={arXiv preprint arXiv:2507.01352},
  year={2025}
}
```


---

## Task2VecScorer

### Overview

The **Task2Vec Diversity Coefficient** is a data quality metric designed to quantify the variability and diversity of natural language datasets. Proposed in [Miranda et al., 2023](https://arxiv.org/abs/2306.13840), this method moves beyond simple dataset scale considerations to assess the structural and semantic diversity of pre-training data. The diversity coefficient measures the expected distance between Task2Vec embeddings of data samples, providing a formal and interpretable measure of how varied the content is within a dataset.

Unlike scale-focused approaches, the diversity coefficient captures the richness and variety of data—characteristics that are crucial for training models with strong general capabilities and in-context learning abilities. Higher diversity scores indicate greater variability in the dataset, which has been shown to correlate with improved downstream model performance.

### Metric Definition:

* **Definition:** 

  The diversity coefficient is computed as the expected pairwise cosine distance between Task2Vec embeddings of randomly sampled batches from the dataset.

  ```
  Diversity = E[cosine_distance(embedding_i, embedding_j)]
  ```

  where each embedding is the diagonal of the Fisher Information Matrix (FIM) computed from a fixed probe network (GPT-2) fine-tuned on the target text.

* **Explanation:** 

  The diversity coefficient quantifies the level of structural and semantic diversity in natural language data:
  
  * A **higher diversity score** indicates greater variability in the dataset, suggesting more diverse concepts, richer vocabulary, and varied semantic content. This typically leads to better downstream performance on diverse evaluation tasks.
  * A **lower diversity score** suggests that the dataset contains more homogeneous or repetitive content with limited semantic variability.

### YAML Configuration

```yaml
name: Task2VecScorer
model: openai-community/gpt2
last_layer_only: false
max_length: 512
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"Task2VecScorer"` | Identifier for the scorer |
| `model` | string | `"openai-community/gpt2"` | Path to the GPT-2 model used as the probe network for computing Task2Vec embeddings. Can be a local path or a HuggingFace model identifier |
| `last_layer_only` | boolean | `false` | Whether to compute FIM using only the last layer parameters (`true`: faster computation, lower dimensionality) or all model parameters (`false`: more comprehensive representation) |
| `max_length` | integer | `512` | Maximum sequence length for tokenization. Texts longer than this value will be truncated |

### Underlying Model

The scorer uses **GPT-2** as the fixed probe network for computing Task2Vec embeddings. By default, it uses [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), but users can specify other GPT-2 variants or local model paths through the `model` configuration parameter.

The probe network remains fixed across all samples to ensure embeddings are comparable. The model's final layer is fine-tuned with a next-token prediction objective to compute the Fisher Information Matrix for each text sample.

### Scoring Process

The Task2Vec scoring process consists of three main stages:

#### 1. Fisher Information Matrix (FIM) Computation

For each text sample in the dataset:
  * The text is tokenized using the GPT-2 tokenizer (with truncation to `max_length`)
  * The model performs a forward pass with gradient computation enabled
  * For each token position, the gradient of the log probability with respect to model parameters is computed
  * The squared gradients are accumulated and averaged over the sequence length
  * The diagonal of the FIM serves as the Task2Vec embedding for that sample

#### 2. Embedding Distance Calculation

After computing embeddings for all samples:
  * A pairwise cosine distance matrix is computed between all Task2Vec embeddings
  * For each sample, the average distance to all other samples is calculated (excluding self-distance)

#### 3. Diversity Score Aggregation

The final diversity coefficient is computed as the mean of all pairwise average distances, representing the overall variability of the dataset.

**Note:** Unlike other scorers that assign individual scores to each sample, Task2VecScorer computes a single aggregate score for the entire dataset, making it suitable for dataset-level quality assessment rather than sample-level filtering.

### Output Format

The scorer returns a dictionary containing the following metrics:

```json
{
  "score": 0.8234,
  "num_samples": 1000,
  "num_anomalous": 5,
  "num_truncated": 120,
  "truncation_rate": 0.12,
  "last_layer_only": false,
  "embedding_dim": 124439808
}
```

- `score`: The diversity coefficient computed as the expected pairwise cosine distance between Task2Vec embeddings. Higher values indicate greater dataset diversity
- `num_samples`: Total number of valid samples successfully processed from the dataset
- `num_anomalous`: Count of records that could not be processed due to JSON parsing errors or missing required fields
- `num_truncated`: Number of samples that exceeded `max_length` and were truncated during tokenization
- `truncation_rate`: Proportion of truncated samples (num_truncated / num_samples)
- `last_layer_only`: Boolean indicating whether the FIM was computed using only the last layer parameters
- `embedding_dim`: The dimensionality of the Task2Vec embeddings, which depends on the number of parameters considered

### Citation

```bibtex
@article{miranda2023beyond,
  title={Beyond scale: The diversity coefficient as a data quality metric for variability in natural language data},
  author={Miranda, Brando and Lee, Alycia and Sundar, Sudharsan and Casasola, Allison and Koyejo, Sanmi},
  journal={arXiv preprint arXiv:2306.13840},
  year={2023}
}
```


---

## TextbookScorer

### Overview

The **Textbook Quality Scorer** (also known as Educational Value Classifier) is a FastText-based evaluation tool designed to assess the educational value of text data. Inspired by the "Textbooks Are All You Need" philosophy, this scorer classifies whether text from the web or instruction-following datasets has high educational value. It provides a fast, CPU-based solution that can process over 2000 examples per second, making it suitable for large-scale data curation in LLM pretraining and instruction tuning.

The scorer is particularly useful for filtering and ranking training data based on educational content quality, helping to improve the quality of data used for model training following the principle of "garbage in, garbage out."

### Metric Definition:

* **Definition:** 

  The Educational Value score is calculated as a weighted average of three classification labels:
  
  * **Low** (score = 0): Bottom 25% educational value
  * **Mid** (score = 1): Middle 25-75% educational value  
  * **High** (score = 2): Top 25% educational value

* **Score Range:** [0, 2]

* **Explanation:** 

  The final score represents the expected educational value of the text based on the probability distribution across the three categories.
  
  * A **higher Educational Value score** (closer to 2) indicates text with rich educational content, such as scientific explanations, academic materials, or well-structured instructional content.
  * A **lower Educational Value score** (closer to 0) suggests text with minimal educational content, such as memes, casual conversations, or low-quality web content.

### YAML Configuration

```yaml
name: TextbookScorer
model: kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2
batch_size: 32
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TextbookScorer"` | Identifier for the scorer |
| `model` | string | `"kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2"` | Path to a local FastText model or HuggingFace model ID |
| `batch_size` | integer | `32` | Number of samples to process in each batch |

### Underlying Model

The scorer uses [kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2](https://huggingface.co/kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2) by default. This is a FastText model trained on web/raw text to predict educational value. The model is quantized for efficiency and can run on CPU, making it highly accessible and fast.

Alternatively, you can specify a custom FastText model by providing a local path in the configuration. The model should be stored as `model.bin` within the specified directory.

### Scoring Process

1. **Text Preparation:**
   * For each data item, the scorer extracts the `instruction` and `output` fields.
   * If an `input` field exists, it is concatenated as: `instruction + '\n' + input + '\n' + output`
   * Otherwise: `instruction + '\n' + output`
   * All newlines in the concatenated text are replaced with spaces for FastText compatibility.

2. **Batch Prediction:**
   * Texts are processed in batches for efficiency (default batch size: 32).
   * The FastText model predicts probabilities for all three labels (Low, Mid, High) for each text.

3. **Score Calculation:**
   * For each sample, the final score is computed as a weighted sum:
     ```
     Score = P(Low) × 0 + P(Mid) × 1 + P(High) × 2
     ```
   * This produces a continuous score in the range [0, 2].

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.456
}
```

- `id`: The unique identifier of the data sample (extracted from the input data's `id` field, or empty string if not present)
- `score`: A float value in the range [0, 2] representing the educational value of the sample. Higher values indicate greater educational quality

### Citation

```bibtex
@misc{ktsui2024cpueduvalue,
      title={Low Latency CPU Based Educational Value Classifier With Generic Educational Value}, 
      author={Ken Tsui and Huu Nguyen},
      year={2024},
}
```



---

## ThinkingProbScorer

### Overview

The **Thinking Probability Scorer** is a model-based evaluation tool designed to quantify the *necessity of deep reasoning* for a given **math problem**. This scorer uses a language model RL-trained with the [AdaptThink](https://github.com/THU-KEG/AdaptThink) framework to estimate the probability that the model would *engage in explicit thinking* ("Thinking" mode) versus *directly providing a solution* ("NoThinking" mode), based on the perceived difficulty of the problem.

By measuring the model's propensity to invoke a reasoning process, this scorer provides an interpretable proxy for **problem difficulty** in mathematical reasoning tasks.

### Metric Definition:

* **Definition:** 

  `Thinking_Prob = 1 - P(</think>)`

* **Explanation:** This metric estimates the *difficulty* of a problem by measuring how unlikely the model is to immediately output the `</think>` token (i.e., to choose NoThinking mode).
  
  * A **higher value** (closer to 1) indicates that the model is **less likely to skip thinking**, suggesting the problem is *hard* and requires deeper reasoning.
  * A **lower value** (closer to 0) indicates the model would confidently produce a final answer **without any thinking**, suggesting the problem is *simple* and straightforward.
  * The metric ranges from **0 to 1**, where higher scores indicate greater problem complexity.

* **Key Advantages:**
  
  * **Adaptive difficulty assessment:** Automatically detects when problems require explicit reasoning steps
  * **Model-agnostic interpretation:** Based on learned behavior from RL training rather than hand-crafted heuristics
  * **Single-token efficiency:** Requires only one forward pass to compute the thinking probability

### YAML Configuration

```yaml
name: ThinkingProbScorer
model: THU-KEG/AdaptThink-7B-delta0.05
batch_size: 128
num_gpu_per_job: 1
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ThinkingProbScorer"` | Identifier for the scorer |
| `model` | string | `"THU-KEG/AdaptThink-7B-delta0.05"` | HuggingFace model path or local directory for the AdaptThink model |
| `batch_size` | integer | `128` | Number of samples to process in parallel per forward pass |
| `num_gpu_per_job` | integer | `1` | Number of GPUs allocated per scoring job |


### Underlying Model

The scorer uses [THU-KEG/AdaptThink-7B-delta0.05](https://huggingface.co/THU-KEG/AdaptThink-7B-delta0.05), a language model trained with reinforcement learning to **adaptively choose** between two modes:

* **Thinking Mode:** `[thinking process]</think>[final solution]` - The model generates explicit reasoning steps before providing the final answer
* **NoThinking Mode:** `</think>[final solution]` - The model directly outputs the final answer by immediately emitting the `</think>` token

The RL training enables the model to learn when deep reasoning is necessary versus when a direct answer is sufficient, making the first-token probability of `</think>` an effective proxy for problem difficulty.

### Scoring Process

1. **Input Processing**: Math problems are passed through the tokenizer with the default chat template applied to format the input correctly.

2. **Single Token Generation**: The model is instructed to generate only **one token**, and the forward pass computes the logits for all possible first tokens.

3. **Probability Extraction**: The probability of generating `</think>` as the first token is extracted from the logprobs output.

4. **Score Computation**: The metric `Thinking_Prob` is computed as `1 - P(</think>)`, where:
   - Higher probabilities of `</think>` indicate simpler problems (low Thinking_Prob)
   - Lower probabilities of `</think>` indicate harder problems requiring reasoning (high Thinking_Prob)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "thinking_prob": 0.8523,
  "no_thinking_prob": 0.1477,
  "score": 0.8523
}
```

- `thinking_prob`: The computed Thinking_Prob score (1 - P(</think>))
- `no_thinking_prob`: The raw probability P(</think>) of immediately outputting the end-of-thinking token
- `score`: The final difficulty score (same as thinking_prob)

### Citation

```bibtex
@article{zhang2025adaptthink,
  title={Adaptthink: Reasoning models can learn when to think},
  author={Zhang, Jiajie and Lin, Nianyi and Hou, Lei and Feng, Ling and Li, Juanzi},
  journal={arXiv preprint arXiv:2505.13417},
  year={2025}
}
```


---

## UPDScorer

### Overview

The **UPD Scorer** (Unpredictable Diversity Scorer) is a model-based evaluation tool designed to quantify the diversity and difficulty of supervised fine-tuning (SFT) data by measuring the **unpredictability** of model responses. Proposed in the paper [Zhang et al., 2025](https://arxiv.org/abs/2503.11441) as part of the D3 (Diversity, Difficulty, and Dependability) framework, this metric evaluates how unexpected each token in the output is, given the instruction context and preceding tokens.

The UPD metric combines two key aspects of token generation to identify samples that challenge the model with unpredictable yet coherent responses: (1) **Predictability** measured by cross-entropy loss, and (2) **Distribution Concentration** measured by Shannon entropy.

### Metric Definition:

* **Definition:** 

  Given an instruction and output pair, the UPD scorer computes:
  
  1. **Token-Level UPD Score:** For each token position *t* in the output:
     ```
     UPD_t = σ(L_t) × max(0, 1 - H_t / log(V))
     ```
     Where:
     - `L_t = -log(P(y_t | x, y_<t))` is the **cross-entropy loss** for predicting token *y_t*
     - `H_t = -Σ P(y) × log(P(y))` is the **Shannon entropy** of the probability distribution
     - `V` is the **vocabulary size**
     - `σ(·)` is the **sigmoid function**
  
  2. **Sample-Level UPD Score:** The final score is the **average UPD** across all output tokens:
     ```
     UPD = (1/N) × Σ UPD_t
     ```
     Where *N* is the number of tokens in the output.

* **Explanation:** This metric measures the **unpredictability and diversity** of model outputs:
  
  * A **higher UPD score** (0.5 - 1.0) indicates that the output is **unpredictable and diverse**, with tokens that are hard to predict but generated from relatively focused distributions. These samples exhibit high diversity and difficulty.
  * A **medium UPD score** (0.2 - 0.5) suggests moderate unpredictability, representing typical instruction-following behavior.
  * A **lower UPD score** (0.0 - 0.2) indicates the output is **predictable** or has high-entropy (near-uniform) distributions, suggesting either memorized patterns or completely uncertain predictions.

* **Key Advantages:**
  
  * **Combines predictability and concentration:** Jointly considers cross-entropy loss and Shannon entropy for robust diversity assessment
  * **Filters random outputs:** The entropy normalization term `max(0, 1 - H_t / log(V))` filters out tokens with extremely high entropy (near-uniform distributions)
  * **Identifies valuable training samples:** Focuses on cases where the model has a relatively concentrated distribution but still struggles to predict the correct token

### YAML Configuration

```yaml
name: UPDScorer
model: meta-llama/Llama-2-7b-hf
max_length: 2048
batch_size: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UPDScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path for the causal language model used to compute probabilities. Can be any autoregressive LM |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |


### Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute token-level probabilities and probability distributions. By default, it uses **Qwen/Qwen3-8B**, but can be configured to use any autoregressive language model. The model computes the probability distribution over the vocabulary at each token position, enabling the calculation of cross-entropy loss and Shannon entropy for UPD scoring.

### Scoring Process

1. **Data Preparation**: For each sample, construct the full text as `instruction + input (if exists) + output`, tokenize the instruction part separately to determine where the output begins, and tokenize the full text with padding and truncation to `max_length`

2. **Batch Forward Pass**: Process multiple samples in parallel and run the causal language model to obtain logits for each token position

3. **Token-Level UPD Computation**: For each token *t* in the output portion, compute:
   - Cross-entropy loss: `L_t = -log(P(y_t | x, y_<t))`
   - Shannon entropy: `H_t = -Σ P(y) × log(P(y))`
   - Token UPD score: `UPD_t = σ(L_t) × max(0, 1 - H_t / log(V))`

4. **Sample-Level Aggregation**: Collect UPD scores for all tokens in the output and compute the average UPD as the final sample score

5. **Edge Case Handling**: Samples exceeding `max_length` are truncated with a warning; samples with no valid output tokens receive a score of `0.0`; padding tokens are automatically skipped during computation

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.4523
}
```

- `id`: Unique identifier from the input data (empty string if not provided)
- `score`: Average UPD score across all output tokens. Range: [0, 1]. Higher values indicate more unpredictable and diverse outputs.

### Citation

```bibtex
@article{zhang2025d3,
  title={D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning},
  author={Zhang, Jia and Zhang, Chen-Xi and Liu, Yao and Jin, Yi-Xuan and Yang, Xiao-Wen and Zheng, Bo and Liu, Yi and Guo, Lan-Zhe},
  journal={arXiv preprint arXiv:2503.11441},
  year={2025}
}
```


---

## UniEvalD2tScorer

### Overview

The **UniEval Data-to-Text (D2t) Scorer** is a model-based evaluation tool designed to assess the quality of data-to-text generation outputs. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), UniEval provides a unified multi-dimensional evaluation framework for text generation tasks. This scorer specifically evaluates the naturalness and informativeness of generated text in data-to-text scenarios, where structured data is transformed into natural language descriptions.

Unlike traditional metrics that rely on simple n-gram matching (e.g., BLEU, ROUGE), UniEval leverages a fine-tuned language model to capture semantic quality and provides more nuanced assessments aligned with human judgments.

### Metric Definition:

* **Definition:** 

  UniEval D2t evaluates generated text along multiple quality dimensions:
  
  1. **Naturalness:** Measures how fluent, grammatical, and human-like the generated text appears
  2. **Informativeness:** Measures how much relevant information from the reference data is captured in the generated output
  3. **Overall:** (optional) An aggregate quality score that combines multiple dimensions into a single unified metric

* **Explanation:** 

  * A **higher naturalness score** indicates that the output reads smoothly and naturally, while a **lower score** suggests awkward phrasing, grammatical errors, or unnatural language constructs.
  * A **higher informativeness score** indicates that the generated text effectively conveys the key information, while a **lower score** suggests missing or insufficient content coverage.
  * All scores are **normalized between 0 and 1**, where higher values indicate better quality.

* **Key Advantages:**
  
  * **Multi-dimensional evaluation:** Provides separate scores for different quality aspects rather than a single monolithic metric
  * **Semantic understanding:** Leverages fine-tuned language models to capture semantic quality beyond simple n-gram matching
  * **Human alignment:** Trained on human evaluation data to better correlate with human judgments
  * **Reference-based assessment:** Compares generated text against gold standard references for more grounded evaluation

### YAML Configuration

```yaml
name: UniEvalD2tScorer
model: MingZhong/unieval-sum
dimensions: ['naturalness', 'informativeness']
max_length: 1024
batch_size: 8
overall: true
device: cuda:0
cache_dir: null
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalD2tScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-sum"` | Path or identifier for the UniEval model. Can be either a Hugging Face model identifier or a local path to a downloaded model directory |
| `dimensions` | list of strings | `['naturalness', 'informativeness']` | List of evaluation dimensions to compute. Valid options: `'naturalness'` (evaluates text fluency and grammaticality), `'informativeness'` (evaluates information coverage) |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization. Text exceeding this limit will be truncated with a warning |
| `batch_size` | integer | `8` | Number of samples to process in each batch during evaluation. Larger values speed up evaluation but require more GPU memory |
| `overall` | boolean | `true` | Whether to compute an overall aggregate score combining all dimensions |
| `device` | string | auto-detect | Device for model inference (e.g., `cuda:0`, `cpu`). If not specified, automatically selects `cuda:0` if available, otherwise `cpu` |
| `cache_dir` | string or null | `null` | Directory to cache downloaded models. If `null`, uses the default Hugging Face cache location |


### Underlying Model

The scorer uses **[MingZhong/unieval-sum](https://huggingface.co/MingZhong/unieval-sum)** by default, which is a T5-based model fine-tuned on multiple text generation evaluation tasks. The model is trained to predict human evaluation scores across various quality dimensions.

### Scoring Process

The UniEval D2t scoring process operates as follows:

#### Single Item Evaluation

1. **Data Extraction**: Extract the `output` (generated text) and `reference` (ground truth text) from each data item. The `reference` field is required and must not be empty.

2. **Truncation Check**: Tokenize the concatenated text (`output + reference`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued with the item ID.

3. **Evaluation**: Pass the output-reference pair through the UniEval D2t evaluator, which:
   - Encodes both texts using the T5-based model
   - Computes dimension-specific scores based on the model's learned representations
   - Optionally computes an overall quality score

4. **Score Return**: Return a dictionary containing scores for each specified dimension.

#### Batch Evaluation

For efficiency when processing entire datasets:

1. **Data Loading**: Read all data items from the input JSONL file and extract output-reference pairs.

2. **Truncation Validation**: Check all items for potential truncation issues and report the total count.

3. **Batch Processing**: Process data in batches of size `batch_size`, where each batch is evaluated in parallel.

4. **Result Aggregation**: Collect scores for all items and format them with corresponding item IDs.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_D2t_Naturalness": 0.8723,
  "UniEval_D2t_Informativeness": 0.7891,
  "UniEval_D2t_Overall": 0.8307
}
```

- `id`: Unique identifier for the data item (copied from the input data)
- `UniEval_D2t_Naturalness`: Naturalness score (0-1 range), measuring text fluency and grammaticality. Only present if `'naturalness'` is in the `dimensions` configuration
- `UniEval_D2t_Informativeness`: Informativeness score (0-1 range), measuring information coverage. Only present if `'informativeness'` is in the `dimensions` configuration
- `UniEval_D2t_Overall`: Overall aggregate quality score (0-1 range), combining all dimensions. Only present if `overall` is set to `true` in the configuration

**Note**: The output will only include dimensions that were specified in the configuration. Dimension names are automatically capitalized in the output field names.

### Citation

```bibtex
@article{zhong2022towards,
  title={Towards a unified multi-dimensional evaluator for text generation},
  author={Zhong, Ming and Liu, Yang and Yin, Da and Mao, Yuning and Jiao, Yizhu and Liu, Pengfei and Zhu, Chenguang and Ji, Heng and Han, Jiawei},
  journal={arXiv preprint arXiv:2210.07197},
  year={2022}
}
```



---

## UniEvalDialogScorer

### Overview

The **UniEval Dialog Scorer** is a model-based evaluation tool designed to assess the quality of dialogue generation systems. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), UniEval provides a unified multi-dimensional evaluation framework for text generation tasks. This scorer specifically evaluates dialogue responses across multiple quality dimensions including naturalness, coherence, engagingness, groundedness, and understandability.

Unlike traditional dialogue evaluation metrics that focus on single aspects or require extensive human annotations, UniEval Dialog leverages a fine-tuned language model to provide comprehensive, automated assessments that align closely with human judgments across multiple quality facets.

### Metric Definition:

* **Definition:** 

  UniEval Dialog evaluates generated dialogue responses along the following dimensions:

  * **Naturalness**: Measures how fluent, grammatical, and human-like the dialogue response appears. A higher naturalness score indicates that the response reads smoothly and naturally, while a lower score suggests awkward phrasing, grammatical errors, or robotic language.

  * **Coherence**: Measures how logically consistent and contextually relevant the response is to the dialogue history. A higher coherence score indicates strong logical flow and contextual appropriateness, while a lower score suggests disconnected or irrelevant responses.

  * **Engagingness**: Measures how interesting, engaging, and conversationally appropriate the response is. A higher engagingness score indicates that the response is likely to maintain user interest and encourage continued conversation, while a lower score suggests bland or disengaging content.

  * **Groundedness**: Measures how well the response is grounded in the provided context or knowledge source. A higher groundedness score indicates that the response accurately reflects and utilizes the given context, while a lower score suggests the response deviates from or contradicts the contextual information.

  * **Understandability**: Measures how clear and easy to understand the response is for users. A higher understandability score indicates that the response is straightforward and comprehensible, while a lower score suggests confusing or ambiguous content.

  * **Overall** (optional): When enabled, provides an aggregate quality score that combines multiple dimensions into a single unified metric.

* **Explanation:** All scores are normalized between 0 and 1, where higher values indicate better quality across all dimensions.

### YAML Configuration

```yaml
name: UniEvalDialogScorer
model: MingZhong/unieval-dialog
dimensions: ['naturalness', 'coherence', 'engagingness', 'groundedness', 'understandability']
max_length: 1024
batch_size: 8
overall: true
device: cuda:0
cache_dir: null
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalDialogScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-dialog"` | Path or identifier for the UniEval Dialog model. Can be a Hugging Face model identifier or a local path to a downloaded model directory |
| `dimensions` | list of strings | `['naturalness', 'coherence', 'engagingness', 'groundedness', 'understandability']` | List of evaluation dimensions to compute. Valid options: `'naturalness'`, `'coherence'`, `'engagingness'`, `'groundedness'`, `'understandability'` |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization. Text exceeding this limit will be truncated with a warning |
| `batch_size` | integer | `8` | Number of samples to process in each batch during evaluation. Larger values speed up evaluation but require more GPU memory |
| `overall` | boolean | `true` | Whether to compute an overall aggregate score combining all dimensions |
| `device` | string | auto-detect | Device for model inference (e.g., `cuda:0`, `cpu`). If not specified, automatically selects `cuda:0` if available, otherwise `cpu` |
| `cache_dir` | string or null | `null` | Directory to cache downloaded models. If `null`, uses the default Hugging Face cache location |


### Underlying Model

The scorer uses **[MingZhong/unieval-dialog](https://huggingface.co/MingZhong/unieval-dialog)** by default, which is a T5-based model specifically fine-tuned for multi-dimensional dialogue evaluation. The model is trained to predict human evaluation scores across various dialogue quality dimensions.

### Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` field)
   - Input (optional, from `input` field)
   - Output (from `output` field) - the generated dialogue response to be evaluated
   - Context (from `context` field) - required contextual information or knowledge base (must not be empty)

2. **Data Preparation**: 
   - Combine `instruction` and `input` (if present) to form the dialogue history (`source`)
   - Format the source with proper line breaks (ending with `\n\n` as required by UniEval)
   - Validate the `context` (must not be empty, automatically adds trailing `\n` if missing)

3. **Tokenization & Truncation Check**: Tokenize the concatenated text (`source + output + context`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued

4. **Model Evaluation**: Pass the source-output-context triplet through the UniEval Dialog evaluator, which:
   - Encodes all components using the T5-based model
   - Computes dimension-specific scores based on the model's learned representations
   - Optionally computes an overall quality score

5. **Batch Processing**: For efficiency, data is processed in batches of size `batch_size`, with each batch evaluated in parallel

6. **Score Computation**: Return a dictionary containing scores for each specified dimension (naturalness, coherence, engagingness, groundedness, understandability, and optionally overall)

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_Dialog_Naturalness": 0.8923,
  "UniEval_Dialog_Coherence": 0.8567,
  "UniEval_Dialog_Engagingness": 0.8234,
  "UniEval_Dialog_Groundedness": 0.8791,
  "UniEval_Dialog_Understandability": 0.8645,
  "UniEval_Dialog_Overall": 0.8632
}
```

- `id`: Unique identifier for the data item
- `UniEval_Dialog_Naturalness`: Naturalness score (0-1 range), measuring text fluency and grammaticality
- `UniEval_Dialog_Coherence`: Coherence score (0-1 range), measuring logical consistency and contextual relevance
- `UniEval_Dialog_Engagingness`: Engagingness score (0-1 range), measuring conversational quality and user engagement
- `UniEval_Dialog_Groundedness`: Groundedness score (0-1 range), measuring fidelity to the provided context
- `UniEval_Dialog_Understandability`: Understandability score (0-1 range), measuring clarity and comprehensibility
- `UniEval_Dialog_Overall`: Overall aggregate quality score (0-1 range), combining all dimensions (only present if `overall=true`)

**Note**: The output will only include dimensions that were specified in the configuration. Dimension names are automatically capitalized in the output field names.

### Citation

```bibtex
@article{zhong2022towards,
  title={Towards a unified multi-dimensional evaluator for text generation},
  author={Zhong, Ming and Liu, Yang and Yin, Da and Mao, Yuning and Jiao, Yizhu and Liu, Pengfei and Zhu, Chenguang and Ji, Heng and Han, Jiawei},
  journal={arXiv preprint arXiv:2210.07197},
  year={2022}
}
```



---

## UniEvalFactScorer

### Overview

The **UniEvalFactScorer** is a model-based evaluation tool designed to assess the factual consistency of generated text with respect to source documents. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), this scorer is part of the UniEval unified evaluation framework for text generation tasks. It specifically focuses on detecting factual errors, hallucinations, and inconsistencies in generated outputs by comparing them against source information.

This scorer is particularly valuable for tasks such as summarization, question answering, and information extraction, where maintaining factual accuracy is critical. Unlike simple lexical overlap metrics, UniEval Fact leverages a fine-tuned language model to capture semantic-level consistency and identify subtle factual discrepancies.

### Metric Definition:

* **Definition:** 

  UniEval Fact evaluates generated text along a single primary dimension of **Consistency (Factual Consistency)**, which measures how factually consistent the generated output is with the source document. This metric assesses whether the generated text accurately reflects the information in the source without introducing false claims, contradictions, or hallucinated content.

* **Explanation:** The score is normalized between 0 and 1:
  
  * A **higher consistency score** (closer to 1) indicates that the generated output is highly faithful to the source document, with no factual errors or hallucinations.
  * A **lower consistency score** (closer to 0) suggests that the output contains factual inconsistencies, contradictions, or information not supported by the source.

* **Key Advantages:**
  
  * **Semantic-level evaluation:** Unlike simple lexical overlap metrics, UniEval Fact leverages a fine-tuned language model to capture semantic-level consistency
  * **Hallucination detection:** Specifically designed to identify subtle factual discrepancies and hallucinated content
  * **Source-grounded:** Provides objective assessment by comparing generated text against source documents

### YAML Configuration

```yaml
name: UniEvalFactScorer
model: MingZhong/unieval-fact
max_length: 1024
batch_size: 8
device: cuda:0
cache_dir: null
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalFactScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-fact"` | Path or identifier for the UniEval Fact model. Can be a Hugging Face model identifier or local path |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization. Text exceeding this limit will be truncated with a warning |
| `batch_size` | integer | `8` | Number of samples to process in each batch during evaluation |
| `device` | string | `"cuda:0"` | Device for model inference (e.g., `cuda:0`, `cpu`). Auto-selects if not specified |
| `cache_dir` | string or null | `null` | Directory to cache downloaded models. Uses default Hugging Face cache location if `null` |

### Underlying Model

The scorer uses **[MingZhong/unieval-fact](https://huggingface.co/MingZhong/unieval-fact)** by default, which is a T5-based model specifically fine-tuned for factual consistency evaluation. The model is trained to detect factual errors, hallucinations, and inconsistencies by comparing generated text against source documents.

### Scoring Process

The UniEval Fact scoring process operates as follows:

#### Data Requirements

The scorer expects each data item to contain:
- **`instruction`**: The original instruction or prompt
- **`input`** (optional): Additional source information or context
- **`output`**: The generated text to be evaluated for factual consistency

#### Single Item Evaluation

1. **Data Extraction**: Extract and construct the required fields from each data item:
   - Combine `instruction` and `input` (if present) to form the source document
   - The source serves as the ground truth against which factual consistency is measured
   - Extract the `output` (generated text to be evaluated)

2. **Truncation Check**: Tokenize the concatenated text (`source + output`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued with the item ID to alert users of potential information loss.

3. **Consistency Evaluation**: Pass the source-output pair through the UniEval Fact evaluator, which:
   - Encodes both the source and output using the T5-based model
   - Computes a consistency score based on semantic alignment and factual accuracy
   - Detects contradictions, hallucinations, and unsupported claims

4. **Score Return**: Return a dictionary containing the consistency score.

#### Batch Evaluation

For efficiency when processing entire datasets:

1. **Data Loading**: Read all data items from the input JSONL file and extract source-output pairs.

2. **Truncation Validation**: Check all items for potential truncation issues and report the total count of items that exceed the maximum length.

3. **Batch Processing**: Process data in batches of size `batch_size`, where each batch is evaluated in parallel to maximize throughput.

4. **Result Aggregation**: Collect consistency scores for all items and format them with corresponding item IDs.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_Fact_Consistency": 0.8945
}
```

- `id`: Unique identifier for the data item (copied from the input data)
- `UniEval_Fact_Consistency`: Factual consistency score (0-1 range), measuring how well the generated output aligns with the source document without introducing factual errors or hallucinations. Higher scores indicate better factual consistency

### Citation

```bibtex
@article{zhong2022towards,
  title={Towards a unified multi-dimensional evaluator for text generation},
  author={Zhong, Ming and Liu, Yang and Yin, Da and Mao, Yuning and Jiao, Yizhu and Liu, Pengfei and Zhu, Chenguang and Ji, Heng and Han, Jiawei},
  journal={arXiv preprint arXiv:2210.07197},
  year={2022}
}
```



---

## UniEvalSumScorer

### Overview

The **UniEval Summarization Scorer** is a model-based evaluation tool designed to assess the quality of text summarization outputs. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), this scorer is part of the UniEval unified evaluation framework for text generation tasks. It provides comprehensive multi-dimensional evaluation of summaries, measuring coherence, consistency, fluency, and relevance.

Unlike traditional summarization metrics such as ROUGE that rely on lexical overlap, UniEval Summarization leverages a fine-tuned language model to capture semantic quality and provide more nuanced assessments that better correlate with human judgments across multiple quality dimensions.

### Metric Definition:

* **Definition:** 

  UniEval Summarization evaluates generated summaries along the following dimensions:
  
  1. **Coherence:** Measures the logical flow and structural organization of the summary
  2. **Consistency:** Measures the factual consistency between the summary and the source document
  3. **Fluency:** Measures the grammatical correctness and readability of the summary
  4. **Relevance:** Measures how well the summary captures the key information from the source document
  5. **Overall** (optional): Aggregate quality score combining all dimensions

* **Explanation:** Each dimension provides specific quality insights:
  
  * **Coherence**: A **higher score** indicates well-organized content with smooth transitions between sentences and ideas, while a **lower score** suggests disjointed or poorly structured content.
  * **Consistency**: A **higher score** indicates accurate reflection of source information without contradictions or hallucinations, while a **lower score** suggests factual errors or misrepresentations.
  * **Fluency**: A **higher score** indicates grammatically correct, natural-sounding, and easy-to-read text, while a **lower score** suggests grammatical errors, awkward phrasing, or unclear expression.
  * **Relevance**: A **higher score** indicates focus on important content and omission of irrelevant details, while a **lower score** suggests missed key points or unnecessary information.

* **Key Advantages:**
  
  * **Multi-dimensional assessment:** Provides comprehensive evaluation across multiple quality aspects
  * **Semantic understanding:** Leverages fine-tuned language models instead of lexical overlap
  * **Human correlation:** Better aligns with human judgments compared to traditional metrics
  * **Normalized scores:** All scores range from 0 to 1 for consistent interpretation

### YAML Configuration

```yaml
name: UniEvalSumScorer
model: MingZhong/unieval-sum
dimensions: ['coherence', 'consistency', 'fluency', 'relevance']
max_length: 1024
batch_size: 8
overall: true
device: cuda:0
cache_dir: null
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalSumScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-sum"` | HuggingFace model path or local directory for the UniEval Summarization model |
| `dimensions` | list | `['coherence', 'consistency', 'fluency', 'relevance']` | List of evaluation dimensions: `'coherence'`, `'consistency'`, `'fluency'`, `'relevance'` |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization (text exceeding this will be truncated) |
| `batch_size` | integer | `8` | Number of samples to process in each batch (larger values require more GPU memory) |
| `overall` | boolean | `true` | Whether to compute an overall aggregate score combining all dimensions |
| `device` | string | `"cuda:0"` (auto) | Device for model inference (`cuda:0`, `cpu`, etc.) |
| `cache_dir` | string/null | `null` | Directory to cache downloaded models (null uses default HuggingFace cache) |

### Underlying Model

The scorer uses **[MingZhong/unieval-sum](https://huggingface.co/MingZhong/unieval-sum)** by default, which is a T5-based model specifically fine-tuned for multi-dimensional summarization evaluation. The model is trained to predict human evaluation scores across various summary quality dimensions.

### Scoring Process

The UniEval Summarization scoring process operates as follows:

#### Data Requirements

The scorer expects each data item to contain:
- **`instruction`**: The summarization instruction or prompt
- **`input`** (optional): Additional context or source document information
- **`output`**: The generated summary to be evaluated
- **`reference`**: Required reference summary (must not be empty)

#### Single Item Evaluation

1. **Data Extraction**: Extract and construct the required fields from each data item:
   - Combine `instruction` and `input` (if present) to form the source document
   - Extract the `output` (generated summary to be evaluated)
   - Extract and validate the `reference` (must not be empty)

2. **Truncation Check**: Tokenize the concatenated text (`source + output + reference`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued with the item ID to alert users of potential information loss.

3. **Multi-dimensional Evaluation**: Pass the source-output-reference triplet through the UniEval Summarization evaluator, which:
   - Encodes all components using the T5-based model
   - Computes dimension-specific scores (coherence, consistency, fluency, relevance) based on the model's learned representations
   - Optionally computes an overall quality score combining all dimensions

4. **Score Return**: Return a dictionary containing scores for each specified dimension.

#### Batch Evaluation

For efficiency when processing entire datasets:

1. **Data Loading**: Read all data items from the input JSONL file and extract source-output-reference triplets.

2. **Truncation Validation**: Check all items for potential truncation issues and report the total count of items that exceed the maximum length.

3. **Batch Processing**: Process data in batches of size `batch_size`, where each batch is evaluated in parallel to maximize throughput and GPU utilization.

4. **Result Aggregation**: Collect scores for all items and format them with corresponding item IDs.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_Sum_Coherence": 0.8756,
  "UniEval_Sum_Consistency": 0.8934,
  "UniEval_Sum_Fluency": 0.9012,
  "UniEval_Sum_Relevance": 0.8623,
  "UniEval_Sum_Overall": 0.8831
}
```

- `id`: Unique identifier for the data item (copied from input data)
- `UniEval_Sum_Coherence`: Coherence score (0-1) measuring logical flow and structural organization (only if `'coherence'` in dimensions)
- `UniEval_Sum_Consistency`: Consistency score (0-1) measuring factual consistency with source document (only if `'consistency'` in dimensions)
- `UniEval_Sum_Fluency`: Fluency score (0-1) measuring grammatical correctness and readability (only if `'fluency'` in dimensions)
- `UniEval_Sum_Relevance`: Relevance score (0-1) measuring coverage of key information (only if `'relevance'` in dimensions)
- `UniEval_Sum_Overall`: Overall aggregate quality score (0-1) combining all dimensions (only if `overall=true`)

### Citation

```bibtex
@article{zhong2022towards,
  title     = {Towards a unified multi-dimensional evaluator for text generation},
  author    = {Zhong, Ming and Liu, Yang and Yin, Da and Mao, Yuning and Jiao, Yizhu and Liu, Pengfei and Zhu, Chenguang and Ji, Heng and Han, Jiawei},
  journal   = {arXiv preprint arXiv:2210.07197},
  year      = {2022}
}
```


---
