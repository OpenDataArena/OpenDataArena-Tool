import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, Any, List
import torch

from utils.utils_jsonl import save_jsonl, merge_multiple_scores, merge_jsonl_files, add_evaluation_keys_to_jsonl
from utils.config_loader import ConfigLoader
from scorers.scorer_factory import ScorerFactory


os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

def evaluate_scorer(scorer_config: Dict[str, Any], dataset_path: str) -> Dict[str, Any]:

    name = scorer_config["params"]["name"]
    sub_name = scorer_config["params"].get("sub_name", name)
    scorer = ScorerFactory.create(name, scorer_config.get("params", {}))

    try:
        result = scorer.evaluate(dataset_path)
        print(f"{name} completed on {dataset_path}.")
    finally:

        del scorer
        torch.cuda.empty_cache()
        print(f"{name} memory cleaned up.")

    return {sub_name: result}


def split_dataset(dataset_path: str, num_parts: int, temp_dir: str) -> List[str]:

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    part_size = max(1, (len(lines) + num_parts - 1) // num_parts)
    part_paths: List[str] = []

    for idx in range(num_parts):
        start, end = idx * part_size, min((idx + 1) * part_size, len(lines))
        if start >= len(lines):
            break
        part_file = os.path.join(temp_dir, f"data_part_{idx}.jsonl")
        with open(part_file, "w", encoding="utf-8") as fw:
            fw.writelines(lines[start:end])
        part_paths.append(part_file)

    return part_paths


def _worker(
    gpu_idx: int,
    scorer_config: Dict[str, Any],
    dataset_path: str,
    temp_dir: str,
    return_dict,
    scores_info_path
):
    """Run one process per GPU to score its own data shard"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        import uuid
        cache_dir = f"/tmp/vllm_cache_{uuid.uuid4().hex[:8]}"
        os.environ["VLLM_CACHE_DIR"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        torch.cuda.set_device(0)

        if scorer_config["params"]["name"] == "FailRateScorer":
            from scorers.FailRateScorer import FailRateScorer
            scorer = FailRateScorer(scorer_config.get("params", {}))
        else:
            from scorers.scorer_factory import ScorerFactory
            ScorerFactory.load_scorers(scores_info_path)
            scorer = ScorerFactory.create(
                scorer_config["params"]["name"], scorer_config.get("params", {}))

        result = scorer.evaluate(dataset_path)
        scorer_name = next(iter(result)) if isinstance(
            result, dict) else scorer_config["params"]["name"]

        out_file = os.path.join(temp_dir, f"{scorer_name}_gpu{gpu_idx}.jsonl")
        save_jsonl(result if isinstance(result, list)
                   else result[scorer_name], out_file)

        try:
            import shutil
            shutil.rmtree(cache_dir)
        except:
            pass

        return_dict[gpu_idx] = out_file

    except Exception as e:
        print(f"Error in worker process for GPU {gpu_idx}: {e}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_idx] = f"ERROR_{gpu_idx}"


def evaluate_with_multi_gpu(
    scorer_config: Dict[str, Any],
    dataset_path: str,
    num_gpu: int,
    temp_dir: str,
    scores_info_path: str
) -> Dict[str, Any]:

    part_paths = split_dataset(dataset_path, num_gpu, temp_dir)
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        actual_gpu_idx_map = [
            int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        assert len(
            actual_gpu_idx_map) == num_gpu, f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']} is not consistent with num_gpu: {num_gpu}"
    else:
        actual_gpu_idx_map = list(range(num_gpu))

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    procs = []

    for gpu_idx in range(num_gpu):
        p = mp.Process(
            target=_worker,
            args=(actual_gpu_idx_map[gpu_idx], scorer_config,
                  part_paths[gpu_idx], temp_dir, return_dict, scores_info_path),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    part_files = []
    for i in range(num_gpu):
        result_file = return_dict[actual_gpu_idx_map[i]]
        if result_file.startswith("ERROR_"):
            print(f"GPU {actual_gpu_idx_map[i]} failed: {result_file}")
            return {scorer_config["params"]["name"]: []}
        part_files.append(result_file)

    scorer_name = scorer_config["params"].get(
        "sub_name", scorer_config["params"]["name"])
    merged_path = os.path.join(temp_dir, f"{scorer_name}.jsonl")
    merge_jsonl_files(part_files, merged_path)

    with open(merged_path, "r", encoding="utf-8") as f:
        merged_scores = [json.loads(line) for line in f]

    return {scorer_name: merged_scores}


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Dataset Scoring System (multi-GPU)")
    parser.add_argument("--config", default="configs/demo.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config = ConfigLoader.load_config(args.config)
    scorers_config = ConfigLoader.get_scorer_configs(config)
    dataset_path = ConfigLoader.get_dataset_path(config)
    output_path = ConfigLoader.get_output_path(config)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    requested_gpu = max(1, ConfigLoader.get_num_gpu(config))

    temp_dir = os.path.join(output_path, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    available_gpu = torch.cuda.device_count()
    if available_gpu < requested_gpu:
        print(
            f"⚠️  Requested {requested_gpu} GPUs but only {available_gpu} available, "
            "will fallback to the maximum available."
        )
        requested_gpu = available_gpu

    print("=" * 60)
    print(f"Dataset Path : {dataset_path}")
    print(f"Dataset Name : {dataset_name}")
    print(f"Output Dir   : {output_path} (temp dir: {temp_dir})")
    print(f"Using GPUs   : {requested_gpu}")
    print("-" * 60)
    print("Scorer Configurations:")
    for i, sc in enumerate(scorers_config, 1):
        print(f"  [{i}] {sc['type']}")
        for k, v in sc["params"].items():
            if k != "name":
                print(f"       {k}: {v}")
    print("=" * 60)

    mp.set_start_method("spawn", force=True)

    from scorers.scorer_factory import ScorerFactory
    ScorerFactory.load_scorers("./scorers/scores_info.json")

    print('Add evaluation keys to dataset')
    add_evaluation_keys_to_jsonl(dataset_path, f"{temp_dir}/data.jsonl")
    dataset_path = f"{temp_dir}/data.jsonl"

    scores_output_files: List[str] = []
    for sc in scorers_config:

        if requested_gpu > 1:
            result = evaluate_with_multi_gpu(
                sc, dataset_path, requested_gpu, temp_dir, "./scorers/scores_info.json")
        else:
            result = evaluate_scorer(sc, dataset_path)

        scorer_name = next(iter(result))
        final_score_file = os.path.join(temp_dir, f"{scorer_name}.jsonl")
        save_jsonl(result[scorer_name], final_score_file)
        scores_output_files.append(final_score_file)

    merge_multiple_scores(dataset_path, scores_output_files,
                          f"{output_path}/output.jsonl")
    print(f"FINISH: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
