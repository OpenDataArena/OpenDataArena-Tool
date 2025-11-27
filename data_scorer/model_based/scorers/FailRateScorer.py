import os
import json
import subprocess
from typing import Dict, List
from .base_scorer import BaseScorer
from .utils import get_total_lines
import pandas as pd
import glob
from tqdm import tqdm

class FailRateScorer(BaseScorer):
    def _validate_config(self):
        # Required parameter check
        if "model" not in self.config:
            print("Warning: No model specified in config. Using default model: Qwen/Qwen3-8B")
            self.config['model'] = 'Qwen/Qwen3-8B'

        if "output_path" not in self.config:
            print("Warning: No output_path specified. Using default value of results/fail_rate_output")
            self.config['output_path'] = 'results/fail_rate_output'
        
        # Evaluation parameters
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
        
        # VLLM model parameters
        if "dtype" not in self.config:
            print("Warning: No dtype specified. Using default value of bfloat16.")
            self.config['dtype'] = 'bfloat16'
        
        if "max_num_batched_tokens" not in self.config:
            print("Warning: No max_num_batched_tokens specified. Using default value of 4096.")
            self.config['max_num_batched_tokens'] = 4096
        
        if "max_model_length" not in self.config:
            print("Warning: No max_model_length specified. Using default value of 4096.")
            self.config['max_model_length'] = 4096
        
        if "gpu_memory_utilization" not in self.config:
            print("Warning: No gpu_memory_utilization specified. Using default value of 0.90.")
            self.config['gpu_memory_utilization'] = 0.90
        
        if "temperature" not in self.config:
            print("Warning: No temperature specified. Using default value of 0.6.")
            self.config['temperature'] = 0.6
        
        if "top_p" not in self.config:
            print("Warning: No top_p specified. Using default value of 0.95.")
            self.config['top_p'] = 0.95
        
        print(f"Using model: {self.config['model']}")
        print(f"Using output_path: {self.config['output_path']}")
        print(f"Using metrics_sample_size: {self.config['metrics_sample_size']}")
        print(f"Using generation_size: {self.config['generation_size']}")
        print(f"VLLM parameters:")
        print(f"  - dtype: {self.config['dtype']}")
        print(f"  - max_num_batched_tokens: {self.config['max_num_batched_tokens']}")
        print(f"  - max_model_length: {self.config['max_model_length']}")
        print(f"  - gpu_memory_utilization: {self.config['gpu_memory_utilization']}")
        print(f"  - temperature: {self.config['temperature']}")
        print(f"  - top_p: {self.config['top_p']}")

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
    question = line["instruction"]
    if line.get("input", "").strip():
        question = question + "\\n" + line["input"]
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.replace("{{Question}}", question),
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
            
            split_file = dataset_path
            
            lighteval_split_id = split_id + 1
            dataset_dir = os.path.join(work_dir, f"dataset_split_{lighteval_split_id}")
            os.makedirs(dataset_dir, exist_ok=True)
            
            import shutil
            shutil.copy2(split_file, os.path.join(dataset_dir, "train.jsonl"))
            
            tasks_dir = os.path.join(work_dir, "tasks")
            os.makedirs(tasks_dir, exist_ok=True)
            task_file = self._create_task_file(split_file, lighteval_split_id, tasks_dir)
            
            final_task_file = os.path.join(tasks_dir, f"my_math_task_split_{lighteval_split_id}.py")
            if task_file != final_task_file:
                shutil.move(task_file, final_task_file)
            model_args = f"model_name={self.config['model']},data_parallel_size=1,dtype={self.config['dtype']},max_num_batched_tokens={self.config['max_num_batched_tokens']},max_model_length={self.config['max_model_length']},gpu_memory_utilization={self.config['gpu_memory_utilization']},generation_parameters={{max_new_tokens:{self.config['generation_size']},temperature:{self.config['temperature']},top_p:{self.config['top_p']}}}"
            
            results_dir = os.path.join(work_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            import uuid
            unique_cache_dir = f"/tmp/vllm_cache_{uuid.uuid4().hex[:8]}"
            os.makedirs(unique_cache_dir, exist_ok=True)
            
            env['VLLM_CACHE_DIR'] = unique_cache_dir
            env['TORCH_COMPILE_DISABLE'] = '1'
            env['TORCHDYNAMO_DISABLE'] = '1'
            
            # Reduce VLLM and related library logging output
            env['VLLM_LOGGING_LEVEL'] = 'WARNING'
            env['TRANSFORMERS_VERBOSITY'] = 'error'
            env['TOKENIZERS_PARALLELISM'] = 'false'
            # Hide LightEval internal detailed logs
            env['LIGHTEVAL_LOG_LEVEL'] = 'WARNING'

            cmd = [
                'lighteval', 'vllm', model_args, 
                f"community|my_math_dataset_eval_{lighteval_split_id}|0|0",
                '--use-chat-template',
                '--custom-tasks', final_task_file,
                '--save-details',
                '--output-dir', os.path.join(results_dir, f'results_split_{lighteval_split_id}')
            ]
            
            # Capture output and filter
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Use Popen to filter output in real time
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read and filter output in real time, only display key information
            print()  # Newline, leave space for progress bar
            
            last_progress_line = None
            skip_table = False
            
            for line in process.stdout:
                line = line.rstrip()
                
                # Detect errors
                if 'ERROR' in line.upper() or 'FAILED' in line.upper():
                    print(f"⚠️  {line}")
                    continue
                
                # Detect table start, set skip flag (do not display table)
                if 'Task' in line and 'Version' in line and 'Metric' in line:
                    skip_table = True
                    continue
                
                # Skip table content
                if skip_table:
                    if line.startswith('|') or line.startswith('-'):
                        continue
                    elif line.strip() == '':
                        skip_table = False
                    continue
                
                    # Only keep the latest progress bar (overwrite display)
                if 'Processed prompts:' in line:
                    # Use carriage return to overwrite previous line
                    if last_progress_line:
                        print('\r' + ' ' * len(last_progress_line) + '\r', end='')
                    print(f"⏳ {line}", end='', flush=True)
                    last_progress_line = line
                    # If 100%, newline
                    if '100%' in line:
                        print()
                        last_progress_line = None
                    continue
                
                # Display final Splits progress
                if 'Splits:' in line and '100%' in line:
                    if last_progress_line:
                        print()
                        last_progress_line = None
                    print(f"✓ {line}")
                    continue
            
            returncode = process.wait()
            
            if returncode != 0:
                print(f"\n❌ Error: LightEval failed with return code {returncode}\n")
                raise RuntimeError("lighteval failed")
            
            print(f"   ✓ LightEval completed")
            
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
        
        parquet_pattern = os.path.join(results_split_dir, "details", "*", "*", f"details_community|my_math_dataset_eval_{lighteval_split_id}|0_*.parquet")
        parquet_files = glob.glob(parquet_pattern)
        
        if not parquet_files:
            parquet_pattern = os.path.join(results_split_dir, "details", "*", "*", "*", f"details_community|my_math_dataset_eval_{lighteval_split_id}|0_*.parquet")
            parquet_files = glob.glob(parquet_pattern)
        
        if not parquet_files:
            print(f'   ⚠️  No result files found')
            return result_lines
        
        parquet_file = parquet_files[0]
        
        try:
            df = pd.read_parquet(parquet_file)
            metrics_sample_size = self.config['metrics_sample_size']
            
            # Extract failure rates (with concise progress bar)
            for _, row in tqdm(df.iterrows(), total=len(df), desc="   Extracting", unit=" samples", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                metrics = row.get('metrics', {})
                metric_key = f'math_pass@1:{metrics_sample_size}_samples'
                sample_pass1 = metrics.get(metric_key, 0.0)
                fail_rate = 1.0 - sample_pass1 if sample_pass1 is not None else None
                result_lines.append(fail_rate)
            
            print(f'   ✓ Extracted {len(result_lines)} fail rates')
                
        except Exception as e:
            print(f'   ⚠️  Error: {e}')
        
        return result_lines

    def evaluate(self, dataset_path: str) -> List[Dict]:
        total_lines = get_total_lines(dataset_path)
        
        import re
        match = re.search(r'data_part_(\d+)\.jsonl$', dataset_path)
        if match:
            split_id = int(match.group(1))
        else:
            split_id = 0
        
        print(f"\n{'='*80}")
        print(f"🚀 FailRate Evaluation - Split {split_id}: {total_lines} samples")
        print(f"   Model: {self.config['model']} | Samples: {self.config['metrics_sample_size']}x | Temp: {self.config['temperature']}")
        print(f"{'='*80}")
        
        num_splits = 1
        
        # work_dir = f"{self.config['output_path']}/temp/split_{split_id}"
        work_dir = f"{self.config['output_path']}/job_{split_id}"
        work_dir = os.path.abspath(work_dir)
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Silent cache cleanup (do not output detailed information)
        vllm_cache_dir = os.path.expanduser("~/.cache/vllm")
        if os.path.exists(vllm_cache_dir):
            import shutil
            try:
                shutil.rmtree(vllm_cache_dir)
            except Exception:
                pass
        
        torch_cache_dir = os.path.expanduser("~/.cache/torch")
        if os.path.exists(torch_cache_dir):
            import shutil
            try:
                shutil.rmtree(torch_cache_dir)
            except Exception:
                pass
        
        hf_cache_dir = os.getenv("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        if os.path.exists(hf_cache_dir):
            import shutil
            try:
                for split_id in range(1, 9):
                    dataset_cache_dir = os.path.join(hf_cache_dir, f"dataset_split_{split_id}")
                    if os.path.exists(dataset_cache_dir):
                        shutil.rmtree(dataset_cache_dir)
            except Exception:
                pass
        
        env = os.environ.copy()
        if 'CONDA_DEFAULT_ENV' not in env:
            env['CONDA_DEFAULT_ENV'] = 'oda'
                
        print(f"\n[1/3] 🔄 Running LightEval evaluation...")
        results_dir = self._run_lighteval(dataset_path, num_splits, work_dir, env)
        
        print(f"\n[2/3] 📊 Extracting fail rates from results...")
        fail_rates = self._extract_fail_rates(results_dir, num_splits, dataset_path)
        
        print(f"[3/3] 📦 Assembling final results...")
        
        results = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for idx, line in tqdm(enumerate(lines), total=len(lines), desc="   Assembling", unit=" samples", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            try:
                data = json.loads(line.strip())
                fail_rate = fail_rates[idx] if idx < len(fail_rates) else None
                
                results.append({
                    "id": data.get("id", str(idx)),
                    "score": fail_rate
                })
            except json.JSONDecodeError:
                results.append({
                    "id": str(idx),
                    "score": None
                })
        
        print(f'   ✓ Assembled {len(results)} results')
        
        valid_count = sum(1 for r in results if r['score'] is not None)
        print(f"\n{'='*80}")
        print(f"✅ Completed: {len(results)} samples processed, {valid_count} valid scores")
        print(f"{'='*80}\n")
        
        # Silent cleanup of temporary files
        try:
            import re
            match = re.search(r'data_part_(\d+)\.jsonl$', dataset_path)
            if match:
                split_id = int(match.group(1))
            else:
                split_id = 0
            
            lighteval_split_id = split_id + 1
            
            temp_files_to_clean = [
                os.path.join(work_dir, "tasks"),
                os.path.join(work_dir, f"dataset_split_{lighteval_split_id}"),
                os.path.join(work_dir, f"run_lighteval_split_{lighteval_split_id}.sh")
            ]
            
            for temp_path in temp_files_to_clean:
                if os.path.exists(temp_path):
                    if os.path.isdir(temp_path):
                        import shutil
                        shutil.rmtree(temp_path)
                    else:
                        os.remove(temp_path)
        except Exception:
            pass  # Silent failure
        
        return results 
