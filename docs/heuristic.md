# Heuristic Evaluation

Heuristic evaluation methods rely on simple, rule-based criteria to assess data.

## Output Token Length (QA)

### Overview

The `Output Token Length` metric quantifies the token numbers of the response. This metric helps to evaluate the complexity of the model's output and how much effort the model used to answer specific prompt. The length is calculated using a specific tokenizer, in this case, the `o200k_base` encoder.

### YAML Configuration

```yaml
# Length Configuration

# Scorer name
name: OutputTokenLengthScorer
# Encoder name
encoder: o200k_base
```
