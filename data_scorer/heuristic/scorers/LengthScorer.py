from .base_scorer import BaseScorer
import json
from typing import Dict, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from .utils import get_total_lines

class OutputTokenLengthScorer(BaseScorer):
    def _validate_config(self):
        if "encoder" not in self.config:
            print(
                "Warning: No encoder specified in config. Using default 'o200k_base' encoder.")
            self.config['encoder'] = 'o200k_base'

    def _setup(self):
        self.encoder = tiktoken.get_encoding(
            self.config.get("encoder", "o200k_base"))
        print("Setting up OutputTokenLengthScorer successfully")

    def score_item(self, data_item):
        if "output" not in data_item or data_item["output"] is None or (isinstance(data_item["output"], str) and len(data_item["output"]) == 0):
            return 0
        return len(self.encoder.encode(data_item["output"]))

    def _process_line(self, line):
        item = json.loads(line.strip())
        return {
            "id":item.get("id", ""),
            "A_Length": self.score_item(item)
        }

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        with ThreadPoolExecutor(max_workers=128) as executor:
            with open(dataset, 'r') as f:
                line_generator = (line for line in f)
                results = list(tqdm(executor.map(
                    self._process_line, line_generator), total=num_lines, desc=self.config['name']))
        return results