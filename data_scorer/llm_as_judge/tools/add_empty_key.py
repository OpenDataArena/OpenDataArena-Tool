import json
from tqdm import tqdm
import os
import sys

def save_jsonl(data, path):
    if '/' in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    with open(path, "w", encoding="utf-8") as f:
        for sample in tqdm(data):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_jsonl(path, max_lines=None):
    data = []
    skip_count = 0

    # Improve integer-to-string conversion limits (Python 3.11+)
    sys.set_int_max_str_digits(0)

    try:
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            for i, line in enumerate(tqdm(f), start=1):
                if max_lines is not None and i > max_lines:
                    break
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error at line {i}: {e}")
                    skip_count += 1
                    continue
                except Exception as e:
                    print(f"[Warning] Unexpected error at line {i}: {e}")
                    skip_count += 1
                    continue
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while opening/reading the file: {e}")
        return []

    print(f"Total lines processed: {len(data) + skip_count}, Successfully loaded: {len(data)}, Skipped: {skip_count}")
    return data


def add_empty_key(input_path, output_path):
    ds = load_jsonl(input_path)

    ds_add_key = []

    Q_scores_template = {
        'Clarity': None,  # llm-as-judge
        'Coherence': None,  # llm-as-judge
        'Completeness': None,  # llm-as-judge
        'Complexity': None,  # llm-as-judge
        'Correctness': None,  # llm-as-judge
        'Meaningfulness': None,  # llm-as-judge
        'Difficulty': None,  # llm-as-judge
        'Deita_Complexity': None,  # model-based
        'Thinking_Prob': None,  # model-based
    }
    QA_scores_template = {
        'Clarity': None,  # llm-as-judge
        'Coherence': None,  # llm-as-judge
        'Completeness': None,  # llm-as-judge
        'Complexity': None,  # llm-as-judge
        'Correctness': None,  # llm-as-judge
        'Meaningfulness': None,  # llm-as-judge
        'Relevance': None,  # llm-as-judge
        'IFD': None,  # model-based
        'Deita_Quality': None,  # model-based
        'Reward_Model': None,  # model-based
        'A_Length': None,  # heuristic
        'Fail_Rate': None,  # model-based
    }

    for idx, item in enumerate(ds):
        assert "instruction" in item and "output" in item, f"item {idx} is not valid"
        item["id"] = idx

        # 检查并补充 Q_scores
        if "Q_scores" not in item or not isinstance(item["Q_scores"], dict):
            item["Q_scores"] = {}
        for k, v in Q_scores_template.items():
            if k not in item["Q_scores"]:
                item["Q_scores"][k] = v

        # 检查并补充 QA_scores
        if "QA_scores" not in item or not isinstance(item["QA_scores"], dict):
            item["QA_scores"] = {}
        for k, v in QA_scores_template.items():
            if k not in item["QA_scores"]:
                item["QA_scores"][k] = v

        ds_add_key.append(item)

    save_jsonl(ds_add_key, output_path)