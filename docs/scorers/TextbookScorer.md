# TextbookScorer

## Overview

The **Textbook Quality Scorer** (also known as Educational Value Classifier) is a FastText-based evaluation tool designed to assess the educational value of text data. Inspired by the "Textbooks Are All You Need" philosophy, this scorer classifies whether text from the web or instruction-following datasets has high educational value. It provides a fast, CPU-based solution that can process over 2000 examples per second, making it suitable for large-scale data curation in LLM pretraining and instruction tuning.

The scorer is particularly useful for filtering and ranking training data based on educational content quality, helping to improve the quality of data used for model training following the principle of "garbage in, garbage out."

## Metric Definition:

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

## YAML Configuration

```yaml
name: TextbookScorer
model: kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2
batch_size: 32
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TextbookScorer"` | Identifier for the scorer |
| `model` | string | `"kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2"` | Path to a local FastText model or HuggingFace model ID |
| `batch_size` | integer | `32` | Number of samples to process in each batch |

## Underlying Model

The scorer uses [kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2](https://huggingface.co/kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2) by default. This is a FastText model trained on web/raw text to predict educational value. The model is quantized for efficiency and can run on CPU, making it highly accessible and fast.

Alternatively, you can specify a custom FastText model by providing a local path in the configuration. The model should be stored as `model.bin` within the specified directory.

## Scoring Process

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

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.456
}
```

- `id`: The unique identifier of the data sample (extracted from the input data's `id` field, or empty string if not present)
- `score`: A float value in the range [0, 2] representing the educational value of the sample. Higher values indicate greater educational quality

## Citation

```bibtex
@misc{ktsui2024cpueduvalue,
      title={Low Latency CPU Based Educational Value Classifier With Generic Educational Value}, 
      author={Ken Tsui and Huu Nguyen},
      year={2024},
}
```

