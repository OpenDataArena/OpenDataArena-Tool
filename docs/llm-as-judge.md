# LLM-as-Judge Evaluation

The LLM-as-Judge framework leverages the capabilities of a LLM to serve as an automated evaluator, assessing the quality of data with precision and consistency.

## Metric Definition

### Difficulty (Q)

- `Q_ode_Difficulty`: Evaluates the complexity of code-related questions, considering algorithmic challenges, code structure, and implementation difficulty.
- `Q_Math_Difficulty`: Evaluates the complexity of math-related questions, considering mathematical concepts, solution steps, and reasoning difficulty.

### Relevance (QA)

- `QA_Relevance`: Specifically evaluates whether the answer remains focused on the question, avoiding digressions or irrelevant information.

### Clarity (Q & QA)

- `Q_Clarity`: Evaluates whether the question is clearly expressed, free from ambiguity, and easy to comprehend.
- `QA_Clarity`: Evaluates whether the answer is articulated with clarity, linguistic fluency, and logical structure.

### Coherence (Q & QA)

- `Q_Coherence`: Evaluates the internal logic of the question, ensuring it is coherent and devoid of contradictions.
- `QA_Coherence`: Evaluates the relevance of the answer to the question, ensuring logical consistency in arguments and evidence.

### Completeness (Q & QA)

- `Q_Completeness`: Evaluates whether the question provides sufficient information for a model to generate a complete answer.
- `QA_Completeness`: Evaluates whether the answer fully addresses the user's question, covering all essential aspects.

### Complexity (Q & QA)

- `Q_Complexity`: Evaluates the inherent difficulty of the question, considering factors like multiple concepts, multi-step reasoning, or specialized domain knowledge.
- `QA_Complexity`: Evaluates the depth of analysis and reasoning in the answer, as well as the complexity of the problem addressed.

### Correctness (Q & QA)

- `Q_Correctness`: Evaluates the accuracy of the facts or premises presented in the question.
- `QA_Correctness`: Evaluates the accuracy of the information, facts, and logical reasoning in the answer.

### Meaningfulness (Q & QA)

- `Q_Meaningfulness`: Evaluates whether the question is meaningful, offering practical value or thought-provoking content.
- `QA_Meaningfulness`: Evaluates whether the answer provides valuable, in-depth insights that encourage further reflection.

### All

- `Q/QA_All`: Combine all metrics above, score all metrics in one turn, without outputting the reasoning process.

## Framework Architecture

- `main.py`: Initiates the evaluation process, loads the configuration, and sets up the evaluator.
- **`config.yaml`**: A configuration file that defines API keys, model parameters, input/output paths, and evaluation metrics.
- `config.py`: Uses Pydantic to load and validate the configuration file, ensuring its correctness.
- `evaluator.py`: The core evaluator that processes data asynchronously and interacts with the LLM API to obtain scores.
- **`prompts/`**: Stores the prompt templates required for evaluation, allowing flexible prompt management.
- **`validators.py`**: Validates the responses from the LLM to ensure their format and content are correct.
- `tools/process_scores.py`: Provides a post-processing tool to merge score results back into the original data file.
- `output/`: A directory that stores evaluation results, error logs, and processed IDs.

## YAML Configuration

All configurations for the framework are centralized within a single `config.yaml` file, ensuring streamlined management and easy adjustments. Below is a detailed example configuration:

```yaml
# API and Model Configuration
openai:
  api_key: "your_api_key" # or use "env:OPENAI_API_KEY" to read from environment variables
  base_url: "your_base_url" # or your custom base url

model: "gpt-4.1-nano"
concurrency: 1024 # Number of concurrent requests to the API, 128 ~ 1024 is recommended
timeout: 30 # Timeout for each API request in seconds
retry: 3 # Number of retries for failed API calls
chunk_size: 2048 # Number of items to read from the input file at once

temperature: 0.1 # The sampling temperature, between 0 and 2
top_p: 1.0 # The nucleus sampling probability

# Directory Configuration
input_path: "../data_process/example_input_add_key.jsonl"
output_path: "output"
prompts_dir: "prompts"
id_track_file: "output/scored_ids.txt" # Add this line to enable ID tracking

# Metrics Configuration
# Define which metrics to run for each mode (Q or QA)
# The names must correspond to the prompt files in `prompts_dir`
# e.g., 'Correctness' for QA mode requires a 'QA_Correctness.txt' file.
metrics:
  Q: # Metrics for Question-only evaluation
    - "All"
    # - "Clarity"
    # - "Coherence"
    # - "Completeness"
    # - "Complexity"
    # - "Correctness"
    # - "Meaningness"
  QA: # Metrics for Question-Answer evaluation
    - "All"
    # - "Clarity"
    # - "Coherence"
    # - "Completeness"
    # - "Complexity"
    # - "Correctness"
    # - "Meaningness"
    # - "Relevance"
```

## Scoring Process

1. **Configuration Loading**: The process begins by loading the configuration from `config.yaml`, which includes API credentials, model settings, and evaluation metrics.
2. **Prompt Loading**: The evaluator loads the necessary prompt templates from the `prompts/` directory based on the metrics specified in the configuration.
3. **Data Reading**: Input data is read from JSONL files specified in the `input_dir`, reading in `chunk_size` items at a time to avoid blocking during data reading or excessive memory usage, with each line representing a separate evaluation item. Use the corresponding fields from the data to format `{instruction}` and `{output}` in the prompt.
4. **Asynchronous Evaluation**: The evaluator processes each item asynchronously, sending requests to the LLM API using the loaded prompts and model settings.
5. **Retry Mechanism**: If an API request fails due to rate limits or other transient errors, the evaluator automatically retries the request up to a specified number of times (`retry`), with exponential backoff between attempts to increase the chances of success. To avoid repeated errors, the `temperature` will also increase exponentially each time, but the maximum value is capped at 1.
6. **Response Validation**: The responses from the LLM are validated to ensure they meet the expected format. This includes checking that the output is a valid JSON object, contains all required keys, and that all score values are integers within the range of 1 to 10.
7. **Result Writing**: Successfully evaluated items are written to `_scored.jsonl` files, while any errors encountered are logged in `_errors.jsonl` files. Semi-streaming writing is adopted: after a chunk of data with size `chunk_size` is processed, it is written to the file together.
8. **ID Tracking**: If enabled, the IDs of evaluated items are tracked to prevent re-evaluation of the same items in future runs.
9. **Post-Processing**: The scores can be merged back into the original data files using the `process_scores.py` tool for further analysis or reporting.

## Usage

To effectively utilize the LLM-as-Judge framework, please follow the detailed steps outlined below:

### 1. Configure `config.yaml`

Begin by setting up your `config.yaml` file. This involves specifying your API keys, selecting your preferred model, and configuring other necessary parameters as detailed in the YAML configuration section above. This step is crucial for ensuring that the framework operates with the correct settings.

- `model` should be models supporting `Structured outputs` like `gpt-4.1-nano` and `gpt-4o-mini` to get json format outputs.
- For **Q_Code/Math_Diversity** metrics, we recommend set the `temperture=1.0` following the orginal OpenThoughts paper settings. For other metrics, `temperture=0.1` is recommend.

### 2. Prepare Data

- Configure `input_file` in `config.yaml`: This must be the path to a single `.jsonl` file to be scored.
- Each line in your `jsonl` file should be a JSON object containing the fields `id`, `instruction`, and `output`.

### 3. Prepare Prompts

- Navigate to the `prompts/` directory and create a prompt file for each evaluation metric as defined in your `config.yaml`.
- The naming convention for these files should be `Mode_MetricName.txt` (e.g., `QA_Correctness.txt` or `Q_Clarity.txt`).
- Within these prompt files, use `{instruction}` and `{output}` as placeholders for the QA mode.

### 4. Validator Configurartion (optional)

- In our project, in order to improve efficiency and cost-effectiveness, we score multiple metrics in a single round of dialogue at once (see the prompt for the **All** metric), and we do not require the output of reasoning or score process.
- For more fine-grained scoring, it is recommended to output the reason before determining the score (as shown in the prompt implementations for other metrics). If this scoring method is to be used, the `llm_as_judge/lvalidator.py` file needs to be modified to remove the check that enforces the values returned by the LLM in the JSON to be integers between 0 and 10, in order to allow the inclusion of reasoning in the output.

### 4. Run Evaluation

To execute the evaluation process, run the main script from your terminal using the following command:

```bash
python -m llm_as_judge.main
```

Alternatively, if you want to use a different configuration file:

```bash
python -m llm_as_judge.main --config-path /path/to/your/config.yaml
```

### 5. Post-Process

- To align the output with other tools and maintain flexibility, you can perform an optional post-processing step. The main evaluation script outputs a `scores.jsonl` file containing only the `id` and `scores`. You can use `process_scores.py` to merge these scores back into your original data file, which may contain other fields.

```bash
python tools/process_scores.py --scores_file [PATH_TO_SCORES_FILE] --data_file [PATH_TO_DATA_FILE] --output_file [PATH_TO_OUTPUT_FILE]
```

  The result will be a output file that corresponding placeholder `null` replaced.

## Citation

```bibtex
@article{guha2025openthoughts,
    author  = {Guha, Etash and Marten, Ryan and Keh, Sedrick and Raoof, Negin and Smyrnis, Georgios and Bansal, Hritik and Nezhurina, Marianna and Mercat, Jean and Vu, Trung and Sprague, Zayne and others},
    journal = {arXiv preprint arXiv:2506.04178},
    title   = {OpenThoughts: Data Recipes for Reasoning Models},
    year    = {2025},
}

@inproceedings{liu-etal-2024-alignbench,
    address   = {Bangkok, Thailand},
    author    = {Liu, Xiao and Lei, Xuanyu  and Wang, Shengyuan  and Huang, Yue  and Feng, Andrew  and Wen, Bosi  and Cheng, Jiale  and Ke, Pei  and Xu, Yifan  and Tam, Weng Lam  and Zhang, Xiaohan  and Sun, Lichao  and Gu, Xiaotao  and Wang, Hongning  and Zhang, Jing  and Huang, Minlie  and Dong, Yuxiao  and Tang, Jie},
    booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    doi       = {10.18653/v1/2024.acl-long.624},
    month     = aug,
    pages     = {11621--11640},
    publisher = {Association for Computational Linguistics},
    title     = {{A}lign{B}ench: Benchmarking {C}hinese Alignment of Large Language Models},
    url       = {https://aclanthology.org/2024.acl-long.624/},
    year      = {2024},
}
```
