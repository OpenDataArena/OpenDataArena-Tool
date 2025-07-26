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