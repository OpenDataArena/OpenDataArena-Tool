import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import yaml
from typing import Dict, Any, List
import torch

from utils.utils_jsonl import save_jsonl, merge_jsonl_files, add_id_to_jsonl
from utils.config_loader import ConfigLoader
from scorers.scorer_factory import ScorerFactory


os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def split_dataset(dataset_path: str, num_parts: int, temp_dir: str, prefix: str = "data_part") -> List[str]:
    """Split dataset into multiple parts"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    part_size = max(1, (len(lines) + num_parts - 1) // num_parts)
    part_paths: List[str] = []

    for idx in range(num_parts):
        start, end = idx * part_size, min((idx + 1) * part_size, len(lines))
        if start >= len(lines):
            break
        part_file = os.path.join(temp_dir, f"{prefix}_{idx}.jsonl")
        with open(part_file, "w", encoding="utf-8") as fw:
            fw.writelines(lines[start:end])
        part_paths.append(part_file)

    return part_paths


def evaluate_scorer_with_gpus(
    scorer_config: Dict[str, Any],
    dataset_path: str,
    scores_info_path: str
) -> Dict[str, Any]:
    """
    Evaluate a single scorer (supports multi-GPU)
    
    Note: This function is called in a subprocess where CUDA_VISIBLE_DEVICES has been set externally
    The process can see and use all GPUs assigned to it
    """
    name = scorer_config["params"]["name"]
    sub_name = scorer_config["params"].get("sub_name", name)
    
    print(f"[{name}] GPUs visible to current process: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    
    # Load scorer
    if name == "FailRateScorer":
        from scorers.FailRateScorer import FailRateScorer
        scorer = FailRateScorer(scorer_config.get("params", {}))
    else:
        from scorers.scorer_factory import ScorerFactory
        ScorerFactory.load_scorers(scores_info_path)
        scorer = ScorerFactory.create(name, scorer_config.get("params", {}))

    try:
        result = scorer.evaluate(dataset_path)
        print(f"[{name}] Evaluation completed")
    finally:
        del scorer
        torch.cuda.empty_cache()
        print(f"[{name}] GPU memory cleaned")

    return {sub_name: result}


def run_single_scorer_job(
    scorer_config: Dict[str, Any],
    dataset_path: str,
    num_gpu: int,
    temp_dir: str,
    scores_info_path: str
) -> Dict[str, Any]:
    """
    Run evaluation task for a single scorer
    
    Args:
        scorer_config: Scorer configuration
        dataset_path: Dataset path
        num_gpu: Number of GPUs this job can use (not for splitting data, but for making GPUs visible to the process)
        temp_dir: Temporary directory
        scores_info_path: Path to scores configuration file
    
    Returns:
        Dictionary of evaluation results
    
    Note:
        - If num_gpu > 1, the process will see multiple GPUs, and the model internally decides how to use them
        - Data will not be split, kept intact
    """
    return evaluate_scorer_with_gpus(scorer_config, dataset_path, scores_info_path)


def _run_scorer_dp_worker(
    job_id: int,
    scorer_config: Dict[str, Any],
    data_part: str,
    num_gpu_per_job: int,
    temp_dir: str,
    scores_info_path: str,
    return_dict,
    gpu_ids: List[int] = None
):
    """
    Worker process for data parallel tasks (for a single scorer)
    
    Args:
        job_id: Task ID
        scorer_config: Scorer configuration
        data_part: Data partition path
        num_gpu_per_job: Number of GPUs this job uses (for display, no longer for splitting data)
        temp_dir: Temporary directory
        scores_info_path: Path to scores configuration file
        return_dict: Dictionary for returning results
        gpu_ids: List of GPU IDs assigned to this job
    """
    try:
        scorer_name = scorer_config["params"]["name"]
        print(f"\n[{scorer_name} - Job {job_id}] Starting processing...")
        
        # Set dedicated GPUs for this process - let the process see all GPUs assigned to it
        if gpu_ids and len(gpu_ids) > 0:
            gpu_str = ",".join(map(str, gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            print(f"[{scorer_name} - Job {job_id}] Assigned GPUs: {gpu_str} (total {len(gpu_ids)})")
        
        # Set vllm cache directory (avoid multi-process conflicts)
        import uuid
        cache_dir = f"/tmp/vllm_cache_{uuid.uuid4().hex[:8]}"
        os.environ["VLLM_CACHE_DIR"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        job_temp_dir = os.path.join(temp_dir, f"job_{job_id}")
        os.makedirs(job_temp_dir, exist_ok=True)
        
        # Run scorer evaluation - data is no longer split, entire data_part is processed by this process
        # The process can see all assigned GPUs, and the model internally decides how to use them
        result = run_single_scorer_job(
            scorer_config, data_part, num_gpu_per_job, 
            job_temp_dir, scores_info_path
        )
        
        # Save results
        scorer_output_name = next(iter(result))
        output_file = os.path.join(job_temp_dir, f"{scorer_output_name}.jsonl")
        save_jsonl(result[scorer_output_name], output_file)
        
        # Clean up vllm cache
        try:
            import shutil
            shutil.rmtree(cache_dir)
        except:
            pass
        
        return_dict[job_id] = output_file
        print(f"[{scorer_name} - Job {job_id}] Completed: {output_file}")
        
    except Exception as e:
        scorer_name = scorer_config["params"]["name"]
        print(f"\n[{scorer_name} - Job {job_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[job_id] = f"ERROR_JOB_{job_id}"


def evaluate_scorer_with_data_parallel(
    scorer_config: Dict[str, Any],
    dataset_path: str,
    data_parallel: int,
    num_gpu_per_job: int,
    scorer_temp_dir: str,
    scores_info_path: str
) -> str:
    """
    Evaluate a single scorer using data parallelism
    
    Args:
        scorer_config: Scorer configuration
        dataset_path: Dataset path
        data_parallel: Data parallelism degree (how many parts to split the data into)
        num_gpu_per_job: Number of GPUs each job can use
        scorer_temp_dir: Scorer temporary directory
        scores_info_path: Path to scores configuration file
    
    How it works:
        1. Split data into data_parallel parts
        2. Launch data_parallel processes, each processing one part of data
        3. Each process can see num_gpu_per_job GPUs (via CUDA_VISIBLE_DEVICES)
        4. The model in each process can use all GPUs assigned to it (no further data splitting)
    
    Returns: Path to the merged result file
    """
    scorer_name = scorer_config["params"]["name"]
    
    print(f"\n{'='*60}")
    print(f"[{scorer_name}] Configuration:")
    print(f"  - Data parallelism: {data_parallel} (data will be split into {data_parallel} parts)")
    print(f"  - GPUs per job: {num_gpu_per_job} (each job can see {num_gpu_per_job} GPUs)")
    print(f"  - Total GPU requirement: {data_parallel * num_gpu_per_job}")
    print(f"{'='*60}")
    
    # Check if GPU resources are sufficient
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        available_gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        available_gpu_count = torch.cuda.device_count()
    
    total_gpu_needed = data_parallel * num_gpu_per_job
    if total_gpu_needed > available_gpu_count:
        raise ValueError(
            f"[{scorer_name}] Insufficient GPU resources! "
            f"Need {total_gpu_needed} GPUs (data_parallel={data_parallel} × num_gpu_per_job={num_gpu_per_job}), "
            f"but only {available_gpu_count} GPUs available"
        )
    
    # Split data
    data_parts = split_dataset(
        dataset_path, data_parallel, scorer_temp_dir, 
        prefix=f"{scorer_name}_data_part"
    )
    print(f"[{scorer_name}] Data splitting completed: {len(data_parts)} partitions")
    
    # Get available GPU list
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        available_gpus = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        available_gpus = list(range(torch.cuda.device_count()))
    
    print(f"[{scorer_name}] Available GPUs: {available_gpus}")
    
    # Assign GPUs to each job
    job_gpu_assignments = []
    for job_id in range(data_parallel):
        start_gpu_idx = job_id * num_gpu_per_job
        end_gpu_idx = start_gpu_idx + num_gpu_per_job
        assigned_gpus = available_gpus[start_gpu_idx:end_gpu_idx]
        job_gpu_assignments.append(assigned_gpus)
        print(f"[{scorer_name}] Job {job_id} -> GPU {assigned_gpus}")
    
    # Launch data parallel tasks - use spawn context to avoid global setting conflicts
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    return_dict = manager.dict()
    procs = []
    
    for job_id, data_part in enumerate(data_parts):
        p = ctx.Process(
            target=_run_scorer_dp_worker,
            args=(job_id, scorer_config, data_part, num_gpu_per_job,
                  scorer_temp_dir, scores_info_path, return_dict, job_gpu_assignments[job_id])
        )
        p.start()
        procs.append(p)
        time.sleep(1)  # Stagger startup
    
    # Wait for all tasks to complete
    for p in procs:
        p.join()
    
    # Collect results
    result_files = []
    failed_jobs = []
    
    for job_id in range(data_parallel):
        result_file = return_dict.get(job_id)
        if result_file and not result_file.startswith("ERROR_"):
            result_files.append(result_file)
        else:
            failed_jobs.append(job_id)
            print(f"[{scorer_name} - Job {job_id}] Failed!")
    
    if failed_jobs:
        print(f"⚠️  [{scorer_name}] {len(failed_jobs)} tasks failed: {failed_jobs}")
    
    # Merge results
    if result_files:
        scorer_output_name = scorer_config["params"].get("sub_name", scorer_name)
        merged_file = os.path.join(scorer_temp_dir, f"{scorer_output_name}_merged.jsonl")
        merge_jsonl_files(result_files, merged_file)
        
        with open(merged_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        print(f"✓ [{scorer_name}] Merge completed: {total_lines} records")
        
        # Clean up data partition files (keep scoring results in job directories)
        print(f"[{scorer_name}] Cleaning up data partition files...")
        cleaned_count = 0
        
        # Delete data partition files (contains original data, takes up large space)
        for data_part in data_parts:
            try:
                if os.path.exists(data_part):
                    os.remove(data_part)
                    cleaned_count += 1
            except Exception as e:
                print(f"  Warning: Unable to delete {data_part}: {e}")
        
        print(f"✓ [{scorer_name}] Cleanup completed: deleted {cleaned_count} data partition files")
        print(f"  Note: Scoring results in job_*/ directories are retained for viewing intermediate results")
        
        return merged_file
    else:
        raise Exception(f"[{scorer_name}] All tasks failed!")


def calculate_data_parallel(global_num_gpu: int, num_gpu_per_job: int) -> int:
    """
    Calculate data parallelism based on global GPU count and GPUs needed per task
    
    Args:
        global_num_gpu: Total number of globally available GPUs
        num_gpu_per_job: Number of GPUs needed per task
    
    Returns:
        data_parallel: Calculated data parallelism degree
    """
    if num_gpu_per_job is None or num_gpu_per_job == 0:
        # For scorers that don't need GPUs, default data_parallel to 1
        return 1
    
    # Calculate how many tasks can run in parallel based on GPU count
    data_parallel = global_num_gpu // num_gpu_per_job
    
    # At least 1
    return max(1, data_parallel)


def main():
    """Main function: Automatically calculate data parallelism based on global num_gpu and per-scorer num_gpu_per_job"""
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Dataset Scoring System with Auto Data Parallelism")
    parser.add_argument("--config", default="configs/demo.yaml",
                        help="Path to master YAML config file")
    parser.add_argument("--data_ready", action="store_true",
                        help="If the dataset is already been processed")
    args = parser.parse_args()

    # Load main configuration file
    config = ConfigLoader.load_config(args.config)
    
    # Get global GPU count (required parameter)
    global_num_gpu = config.get("num_gpu", None)
    if global_num_gpu is None:
        raise ValueError("'num_gpu' parameter must be specified in configuration file")
    
    # Get global default GPUs per job (optional, defaults to 1)
    global_num_gpu_per_job = config.get("num_gpu_per_job", 1)
    
    dataset_path = ConfigLoader.get_dataset_path(config)
    output_path = ConfigLoader.get_output_path(config)
    scorers_config = ConfigLoader.get_scorer_configs(config)
    
    print("\n" + "=" * 80)
    print("Auto Data Parallel Scoring System V4")
    print("=" * 80)
    print(f"Main config file   : {args.config}")
    print(f"Dataset path       : {dataset_path}")
    print(f"Output directory   : {output_path}")
    print(f"Global GPU total   : {global_num_gpu}")
    print(f"Global default cfg : num_gpu_per_job={global_num_gpu_per_job}")
    print(f"Scorer count       : {len(scorers_config)}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    master_temp_dir = os.path.join(output_path, "master_temp")
    os.makedirs(master_temp_dir, exist_ok=True)
    
    # Load scorer factory
    from scorers.scorer_factory import ScorerFactory
    ScorerFactory.load_scorers("./scorers/scores_info.json")
    
    # Preprocess data (if needed)
    if not args.data_ready:
        print('\n[Preprocessing] Adding evaluation keys to dataset...')
        processed_data = os.path.join(master_temp_dir, "processed_data.jsonl")
        add_id_to_jsonl(dataset_path, processed_data)
        dataset_path = processed_data
        print(f'✓ Preprocessing completed: {dataset_path}')
    
    # Analyze each scorer's configuration and calculate data parallelism
    print("\n" + "=" * 80)
    print("Scorer Configuration Details (auto-calculate data parallelism):")
    print("=" * 80)
    
    scorer_jobs = []
    
    for idx, sc in enumerate(scorers_config, 1):
        scorer_name = sc["params"]["name"]
        
        # Get num_gpu_per_job for this scorer (prefer scorer's own config, otherwise use global default)
        scorer_gpu_per_job = sc["params"].get("num_gpu_per_job", global_num_gpu_per_job)
        
        # Auto-calculate data_parallel
        scorer_dp = calculate_data_parallel(global_num_gpu, scorer_gpu_per_job)
        scorer_total_gpu = scorer_dp * scorer_gpu_per_job if scorer_gpu_per_job else 0
        
        print(f"  [{idx}] {scorer_name}")
        print(f"       num_gpu_per_job  : {scorer_gpu_per_job if scorer_gpu_per_job else 'N/A (no GPU needed)'}")
        print(f"       data_parallel    : {scorer_dp} (auto-calculated)")
        print(f"       Total GPU demand : {scorer_total_gpu}")
        
        # Display other parameters
        for k, v in sc["params"].items():
            if k not in ["name", "num_gpu_per_job"]:
                print(f"       {k}: {v}")
        
        scorer_jobs.append({
            "config": sc,
            "data_parallel": scorer_dp,
            "num_gpu_per_job": scorer_gpu_per_job if scorer_gpu_per_job else 0,
            "name": scorer_name
        })
    
    print("-" * 80)
    print(f"Note: data_parallel = global GPU count({global_num_gpu}) // GPUs per job")
    print(f"      Each job can see and use all GPUs assigned to it (no further data splitting)")
    print(f"      Scorers that don't need GPUs default to data_parallel=1")
    print("=" * 80)
    
    # Execute evaluation for each scorer
    all_scorer_results = []
    
    for idx, job in enumerate(scorer_jobs, 1):
        scorer_name = job["name"]
        scorer_config = job["config"]
        data_parallel = job["data_parallel"]
        num_gpu_per_job = job["num_gpu_per_job"]
        
        print(f"\n{'#'*80}")
        print(f"# Executing Scorer {idx}/{len(scorer_jobs)}: {scorer_name}")
        print(f"{'#'*80}")
        
        # Create temporary directory for this scorer
        scorer_temp_dir = os.path.join(master_temp_dir, f"scorer_{scorer_name}")
        os.makedirs(scorer_temp_dir, exist_ok=True)
        
        try:
            if data_parallel > 1:
                # Use data parallelism
                result_file = evaluate_scorer_with_data_parallel(
                    scorer_config, dataset_path, data_parallel, num_gpu_per_job,
                    scorer_temp_dir, "./scorers/scores_info.json"
                )
            else:
                # No data parallelism, run directly
                if num_gpu_per_job > 0:
                    print(f"[{scorer_name}] Single task mode (num_gpu={num_gpu_per_job})")
                else:
                    print(f"[{scorer_name}] CPU mode (no GPU needed)")
                
                result = run_single_scorer_job(
                    scorer_config, dataset_path, num_gpu_per_job,
                    scorer_temp_dir, "./scorers/scores_info.json"
                )
                scorer_output_name = next(iter(result))
                result_file = os.path.join(scorer_temp_dir, f"{scorer_output_name}.jsonl")
                save_jsonl(result[scorer_output_name], result_file)
            
            all_scorer_results.append(result_file)
            print(f"✓ [{scorer_name}] Completed: {result_file}")
            
        except Exception as e:
            print(f"❌ [{scorer_name}] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Merge all scorer results - separate pointwise and setwise
    if all_scorer_results:
        print(f"\n{'='*80}")
        print("Merging all Scorer results (distinguish pointwise and setwise)...")
        print(f"{'='*80}")
        
        # Collect scoring files from all scorers
        pointwise_files = {}  # {scorer_name: file_path}
        setwise_files = {}    # {scorer_name: file_path}
        
        # Iterate through all scorer folders under master_temp
        for scorer_dir in os.listdir(master_temp_dir):
            scorer_dir_path = os.path.join(master_temp_dir, scorer_dir)
            if not os.path.isdir(scorer_dir_path) or not scorer_dir.startswith("scorer_"):
                continue
            
            # Find scoring result files (prioritize *_merged.jsonl, then *.jsonl)
            found_file = None
            scorer_name = scorer_dir.replace("scorer_", "")
            
            # First look for _merged.jsonl files
            for fname in os.listdir(scorer_dir_path):
                if fname.endswith("_merged.jsonl"):
                    found_file = os.path.join(scorer_dir_path, fname)
                    break
            
            # If not found, look for .jsonl files (exclude data partitions)
            if not found_file:
                for fname in os.listdir(scorer_dir_path):
                    if fname.endswith(".jsonl") and not fname.startswith(f"{scorer_name}_data_part"):
                        found_file = os.path.join(scorer_dir_path, fname)
                        break
            
            if not found_file:
                print(f"  Warning: Scoring result file not found for {scorer_name}")
                continue
            
            # Read first line to determine if pointwise or setwise
            try:
                with open(found_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        first_obj = json.loads(first_line)
                        if "id" in first_obj:
                            pointwise_files[scorer_name] = found_file
                            print(f"  ✓ {scorer_name}: pointwise scoring ({found_file})")
                        else:
                            setwise_files[scorer_name] = found_file
                            print(f"  ✓ {scorer_name}: setwise scoring ({found_file})")
            except Exception as e:
                print(f"  Warning: Error reading {found_file}: {e}")
        
        print(f"\nFound {len(pointwise_files)} pointwise scorer(s), {len(setwise_files)} setwise scorer(s)")
        
        # Merge pointwise scoring results
        if pointwise_files:
            print(f"\n{'='*80}")
            print("Merging Pointwise scoring results...")
            print(f"{'='*80}")
            
            pointwise_output = os.path.join(output_path, "pointwise_scores.jsonl")
            
            # First read all pointwise scores, organized by id
            id_scores = {}  # {id: {scorer_name: scores}}
            
            for scorer_name, file_path in pointwise_files.items():
                print(f"  Reading scores from {scorer_name}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            item_id = obj.get("id")
                            if item_id is not None:
                                if item_id not in id_scores:
                                    id_scores[item_id] = {}
                                # Remove id field, save all other scoring information
                                score_data = {k: v for k, v in obj.items() if k != "id"}
                                id_scores[item_id][scorer_name] = score_data
                        except Exception as e:
                            print(f"    Warning: Error parsing line: {e}")
            
            # Write merged pointwise results
            with open(pointwise_output, "w", encoding="utf-8") as f:
                for item_id in sorted(id_scores.keys()):
                    result_obj = {
                        "id": item_id,
                        "scores": id_scores[item_id]
                    }
                    f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
            
            print(f"✓ Pointwise results saved: {pointwise_output}")
            print(f"✓ Total records: {len(id_scores)}")
        
        # Merge setwise scoring results
        if setwise_files:
            print(f"\n{'='*80}")
            print("Merging Setwise scoring results...")
            print(f"{'='*80}")
            
            setwise_output = os.path.join(output_path, "setwise_scores.jsonl")
            
            all_setwise_scores = {}
            
            for scorer_name, file_path in setwise_files.items():
                print(f"  Reading scores from {scorer_name}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # Merge all data from this scorer
                            all_setwise_scores[scorer_name] = obj
                        except Exception as e:
                            print(f"    Warning: Error parsing line: {e}")
            
            # Write merged setwise results
            with open(setwise_output, "w", encoding="utf-8") as f:
                f.write(json.dumps(all_setwise_scores, ensure_ascii=False) + "\n")
            
            print(f"✓ Setwise results saved: {setwise_output}")
            print(f"✓ Scorer count: {len(all_setwise_scores)}")
        
        print(f"\n{'='*80}")
        print(f"✓ Successful Scorers: {len(all_scorer_results)}/{len(scorer_jobs)}")
        print(f"{'='*80}")
    else:
        print("\n❌ Error: All Scorers failed!")
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✓ All completed! Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
