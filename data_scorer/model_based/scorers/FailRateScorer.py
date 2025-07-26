import os
import json
import subprocess
from typing import Dict, List
from .base_scorer import BaseScorer
from .utils import get_total_lines
import pandas as pd
import glob


class FailRateScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print("Warning: No model specified in config. Using default model: Qwen/Qwen3-8B")
            self.config['model'] = 'Qwen/Qwen3-8B'

        if "output_path" not in self.config:
            print("Warning: No output_path specified. Using default value of results/fail_rate_output")
            self.config['output_path'] = 'results/fail_rate_output'
        
        if "metrics_sample_size" not in self.config:
            print("Warning: No metrics_sample_size specified. Using default value of 4.")
            self.config['metrics_sample_size'] = 4
        else:
            valid_sizes = [1, 4, 8, 16, 32, 64]
            if self.config['metrics_sample_size'] not in valid_sizes:
                print(f"Warning: metrics_sample_size must be one of {valid_sizes}. Using default value of 4.")
                self.config['metrics_sample_size'] = 4
        
        if "generation_size" not in self.config:
            print("Warning: No generation_size specified. Using default value of 4096.")
            self.config['generation_size'] = 4096
        
        print(f"Using model: {self.config['model']}")
        print(f"Using metrics_sample_size: {self.config['metrics_sample_size']}")
        print(f"Using generation_size: {self.config['generation_size']}")

    def _setup(self):
        print("Setting up FailRateScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        raise NotImplementedError("FailRateScorer does not support single item scoring. Use evaluate() method for batch processing.")

    def _create_task_file(self, split_file: str, split_id: int, output_dir: str) -> str:
        metrics_sample_size = self.config['metrics_sample_size']
        generation_size = self.config['generation_size']
        
        task_content = f'''# community_tasks/my_math_task_split_{split_id}.py
import os
import sys
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics

def prompt_fn(line, task_name: str = None):
    MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{{Question}}
""".strip()
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.replace("{{Question}}", line["instruction"]),
        choices=[str(line["answer"])], 
        gold_index=0,
    )

TASK = LightevalTaskConfig(
    name="my_math_dataset_eval_{split_id}",
    prompt_function=prompt_fn,
    suite=["community"],
    hf_repo="dataset_split_{split_id}",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[
        Metrics.math_pass_at_1_{metrics_sample_size}n  # sample {metrics_sample_size} pass@1
    ],
    generation_size={generation_size},
)

TASKS_TABLE = [TASK]
'''
        
        task_file = os.path.join(output_dir, f"task_split_{split_id}.py")
        with open(task_file, 'w', encoding='utf-8') as f:
            f.write(task_content)
        
        return task_file

    def _run_lighteval(self, dataset_path: str, num_splits: int, output_dir: str, env=None) -> str:
        work_dir = output_dir
        os.makedirs(work_dir, exist_ok=True)
        
        if env is None:
            env = os.environ.copy()
        
        try:
            import re
            match = re.search(r'data_part_(\d+)\.jsonl$', dataset_path)
            if match:
                split_id = int(match.group(1))
            else:
                split_id = 0
            
            print(f"Processing data split {split_id} from file: {dataset_path}")
            
            split_file = dataset_path
            
            lighteval_split_id = split_id + 1
            dataset_dir = os.path.join(work_dir, f"dataset_split_{lighteval_split_id}")
            os.makedirs(dataset_dir, exist_ok=True)
            
            import shutil
            shutil.copy2(split_file, os.path.join(dataset_dir, "train.jsonl"))
            
            print(f"Created dataset directory: {dataset_dir}")
            print(f"Dataset file: {os.path.join(dataset_dir, 'train.jsonl')}")
            
            tasks_dir = os.path.join(work_dir, "tasks")
            os.makedirs(tasks_dir, exist_ok=True)
            task_file = self._create_task_file(split_file, lighteval_split_id, tasks_dir)
            
            final_task_file = os.path.join(tasks_dir, f"my_math_task_split_{lighteval_split_id}.py")
            if task_file != final_task_file:
                shutil.move(task_file, final_task_file)
            
            print("Starting lighteval evaluation...")
            model_args = f"model_name={self.config['model']},data_parallel_size=1,dtype=bfloat16,max_num_batched_tokens=4096,max_model_length=4096,gpu_memory_utilization=0.90,generation_parameters={{max_new_tokens:{self.config['generation_size']},temperature:0.6,top_p:0.95}}"
            
            results_dir = os.path.join(work_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"Running evaluation for split {split_id}...")
            
            import uuid
            unique_cache_dir = f"/tmp/vllm_cache_{uuid.uuid4().hex[:8]}"
            os.makedirs(unique_cache_dir, exist_ok=True)
            
            env['VLLM_CACHE_DIR'] = unique_cache_dir
            env['TORCH_COMPILE_DISABLE'] = '1'
            env['TORCHDYNAMO_DISABLE'] = '1'

            cmd = [
                'lighteval', 'vllm', model_args, 
                f"community|my_math_dataset_eval_{lighteval_split_id}|0|0",
                '--use-chat-template',
                '--custom-tasks', final_task_file,
                '--save-details',
                '--output-dir', os.path.join(results_dir, f'results_split_{lighteval_split_id}')
            ]
            
            print(f"Using independent cache directory: {unique_cache_dir}")
            
            print(f"Running lighteval command in directory: {work_dir}")
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=work_dir)
            
            if result.returncode != 0:
                print(f"Error running lighteval: {result.stderr}")
                raise RuntimeError("lighteval failed")
            
            print(f"Evaluation completed for split {split_id}")
            
            return results_dir
            
        except Exception as e:
            print(f"Error in lighteval evaluation: {e}")
            raise

    def _extract_fail_rates(self, results_dir: str, num_splits: int, original_dataset: str) -> List[float]:
        result_lines = []
        
        import re
        match = re.search(r'data_part_(\d+)\.jsonl$', original_dataset)
        if match:
            split_id = int(match.group(1))
        else:
            split_id = 0
        
        lighteval_split_id = split_id + 1
        results_split_dir = os.path.join(results_dir, f"results_split_{lighteval_split_id}")
        model_name = self.config['model']
        
        parquet_pattern = os.path.join(results_split_dir, "details", model_name, "*", f"details_community|my_math_dataset_eval_{lighteval_split_id}|0_*.parquet")
        parquet_files = glob.glob(parquet_pattern)
        
        if not parquet_files:
            print(f'No parquet files found for split {split_id}')
            return result_lines
        
        parquet_file = parquet_files[0]
        print(f'Processing parquet file: {parquet_file}')
        
        try:
            df = pd.read_parquet(parquet_file)
            print(f'Read {len(df)} rows of data')
            
            metrics_sample_size = self.config['metrics_sample_size']
            for _, row in df.iterrows():
                metrics = row.get('metrics', {})
                metric_key = f'math_pass@1:{metrics_sample_size}_samples'
                sample_pass1 = metrics.get(metric_key, 0.0)
                fail_rate = 1.0 - sample_pass1 if sample_pass1 is not None else None
                result_lines.append(fail_rate)
                
        except Exception as e:
            print(f'Failed to read parquet file: {e}')
        
        print(f'Total extracted {len(result_lines)} fail_rate results for split {split_id}')
        return result_lines

    def evaluate(self, dataset_path: str) -> List[Dict]:
        print(f"Starting fail_rate evaluation for dataset: {dataset_path}")
        
        total_lines = get_total_lines(dataset_path)
        
        import re
        match = re.search(r'data_part_(\d+)\.jsonl$', dataset_path)
        if match:
            split_id = int(match.group(1))
            print(f"Dataset has {total_lines} samples, processing split {split_id + 1} (from main_para.py data partitioning)")
        else:
            split_id = 0
            print(f"Dataset has {total_lines} samples, processing as single split")
        
        num_splits = 1
        
        work_dir = f"{self.config['output_path']}/temp/split_{split_id}"
        work_dir = os.path.abspath(work_dir)
        
        os.makedirs(work_dir, exist_ok=True)
        
        vllm_cache_dir = os.path.expanduser("~/.cache/vllm")
        if os.path.exists(vllm_cache_dir):
            import shutil
            try:
                shutil.rmtree(vllm_cache_dir)
                print(f"Cleaned VLLM cache directory: {vllm_cache_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean VLLM cache: {e}")
        
        torch_cache_dir = os.path.expanduser("~/.cache/torch")
        if os.path.exists(torch_cache_dir):
            import shutil
            try:
                shutil.rmtree(torch_cache_dir)
                print(f"Cleaned torch cache directory: {torch_cache_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean torch cache: {e}")
        
        hf_cache_dir = os.getenv("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        if os.path.exists(hf_cache_dir):
            import shutil
            try:
                for split_id in range(1, 9):
                    dataset_cache_dir = os.path.join(hf_cache_dir, f"dataset_split_{split_id}")
                    if os.path.exists(dataset_cache_dir):
                        shutil.rmtree(dataset_cache_dir)
                        print(f"Cleaned dataset cache directory: {dataset_cache_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean dataset cache: {e}")
        
        env = os.environ.copy()
        if 'CONDA_DEFAULT_ENV' not in env:
            env['CONDA_DEFAULT_ENV'] = 'oda'
        
        env['HF_DATASETS_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
        
        results_dir = self._run_lighteval(dataset_path, num_splits, work_dir, env)
        
        fail_rates = self._extract_fail_rates(results_dir, num_splits, dataset_path)
        
        results = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    fail_rate = fail_rates[idx] if idx < len(fail_rates) else None
                    
                    results.append({
                        "id": data.get("id", str(idx)),
                        "Fail_Rate": fail_rate
                    })
                except json.JSONDecodeError:
                    results.append({
                        "id": str(idx),
                        "Fail_Rate": None
                    })
        
        print(f"Fail_rate evaluation completed. Processed {len(results)} samples.")
        
        try:
            temp_files_to_clean = [
                os.path.join(work_dir, "tasks"),
                os.path.join(work_dir, "dataset_split_1"),
                os.path.join(work_dir, "run_lighteval_split_1.sh")
            ]
            
            for temp_path in temp_files_to_clean:
                if os.path.exists(temp_path):
                    if os.path.isdir(temp_path):
                        import shutil
                        shutil.rmtree(temp_path)
                    else:
                        os.remove(temp_path)
                    print(f"Cleaned temporary file: {temp_path}")
        except Exception as e:
            print(f"Warning: Failed to clean some temporary files: {e}")
        
        return results 