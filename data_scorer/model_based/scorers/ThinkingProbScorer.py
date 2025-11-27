from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from .utils import get_total_lines
import json
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams


class ThinkingProbScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model: THU-KEG/AdaptThink-7B-delta0.05.")
            self.config['model'] = 'THU-KEG/AdaptThink-7B-delta0.05'
        else:
            if not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: THU-KEG/AdaptThink-7B-delta0.05"
                )
                self.config['model'] = 'THU-KEG/AdaptThink-7B-delta0.05'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. "
                )

        if "batch_size" not in self.config:
            print("Warning: No batch_size specified. Using default value of 1.")
            self.config['batch_size'] = 1
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        try:
            # 禁用 v1 引擎，使用 v0 引擎以避免多进程环境下的问题
            import os
            os.environ["VLLM_USE_V1"] = "0"
            
            # self.model = LLM(model=self.config['model'])
            self.model = LLM(
                model=self.config['model'],
                enforce_eager=True,                      # 直接走 eager
                disable_custom_all_reduce=True,          # 禁用自定义 all_reduce
                trust_remote_code=True,                  # 信任远程代码
                gpu_memory_utilization=0.9,              # GPU 内存利用率
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'], 
                cache_dir='./cache',
                trust_remote_code=True
            )
        except Exception as e:
            print(
                f"Warning: Failed to load model from remote. Error: {e}")
            print("Loading THU-KEG/AdaptThink-7B-delta0.05 model.")
            
            import os
            os.environ["VLLM_USE_V1"] = "0"
            
            self.model = LLM(
                model='THU-KEG/AdaptThink-7B-delta0.05',
                enforce_eager=True,
                disable_custom_all_reduce=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                'THU-KEG/AdaptThink-7B-delta0.05', 
                cache_dir='./cache',
                trust_remote_code=True
            )

        print("Setting up ThinkProbScorer successfully")

    def score_item(self, data_item):
        pass

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)

        ds, results = [], []

        with open(dataset, 'r') as f:
            for line in tqdm(f, total=num_lines, desc=self.config['name']):
                ds.append(json.loads(line.strip()))

        prompts = [item['instruction'] for item in ds]

        messages = [
            [{"role": "user", "content": item}] for item in prompts
        ]

        text = [
            self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msg in tqdm(messages)
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            stop=['</think>'],
            include_stop_str_in_output=True,
            logprobs=20,
        )

        for i in tqdm(range(0, len(text), self.config['batch_size'])):
            end = min(i + self.config['batch_size'], len(text))
            batch_text = text[i:end]

            outputs = self.model.generate(batch_text, sampling_params)

            for j, output in enumerate(outputs):
                if 151649 not in output.outputs[0].logprobs[0]:
                    prob_eot = 0
                else:
                    logprob_eot = output.outputs[0].logprobs[0][151649]
                    prob_eot = math.exp(logprob_eot.logprob)

                difficulty = round(1 - prob_eot, 3)

                results.append({
                    "id": ds[i + j]['id'],
                    "score": difficulty
                })

        return results
