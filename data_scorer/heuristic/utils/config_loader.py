import yaml
from typing import Dict, List, Any
from pathlib import Path
import os


class ConfigLoader:
    """
    Read the YAML config and provide a unified access interface:
        input_path      -> Dataset file
        output_path     -> Result directory
        num_gpu         -> String like '1-8' or '0'; can be converted to a list or count
        scorers         -> List of scorers, execution order follows the list
    """

    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg

    @staticmethod
    def get_scorer_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        scorers = cfg.get("scorers", [])
        wrapped = []
        for sc in scorers:
            wrapped.append({
                "type": sc["name"],
                "params": sc
            })
        return wrapped

    @staticmethod
    def get_dataset_path(cfg: Dict[str, Any]) -> str:
        return cfg["input_path"]

    @staticmethod
    def get_output_path(cfg: Dict[str, Any]) -> str:
        return cfg["output_path"]

    @staticmethod
    def get_num_gpu(cfg: Dict[str, Any]) -> int:
        """
        Support two formats: 'N' or 'M-K':
        - '4'    -> 4 GPUs
        - '1-8'  -> 8 GPUs (from 1 to 8, inclusive)
        """
        gpu_field = str(cfg.get("num_gpu", "1")).strip()
        if "-" in gpu_field:
            start, end = map(int, gpu_field.split("-", 1))
            return end - start + 1
        return int(gpu_field)
    
    @staticmethod
    def get_data_parallel(cfg: Dict[str, Any]) -> int:
        """
        Get data parallelism degree (how many splits to divide the data into)
        Default is 1 (no data parallelism)
        """
        return cfg.get("data_parallel", 1)
    
    @staticmethod
    def get_num_gpu_per_job(cfg: Dict[str, Any]) -> int:
        """
        Get the number of GPUs to use per sub-task
        Default is 1
        """
        return cfg.get("num_gpu_per_job", 1)

    @staticmethod
    def get_data_with_id(cfg: Dict[str, Any]) -> bool:
        """
        Indicates whether the input dataset already contains 'id' fields.
        Default is False.
        """
        return cfg.get("data_with_id", False)

    @staticmethod
    def get_scorer_parallel_config(
        scorer: Dict[str, Any], 
        global_data_parallel: int = 1, 
        global_num_gpu_per_job: int = 1
    ) -> tuple:
        """
        Get parallelism configuration for a single scorer
        Prioritize scorer's own configuration, otherwise use global defaults
        
        Args:
            scorer: scorer configuration dictionary
            global_data_parallel: global default data_parallel
            global_num_gpu_per_job: global default num_gpu_per_job
            
        Returns:
            (data_parallel, num_gpu_per_job) tuple
        """
        data_parallel = scorer.get("data_parallel", global_data_parallel)
        num_gpu_per_job = scorer.get("num_gpu_per_job", global_num_gpu_per_job)
        return (data_parallel, num_gpu_per_job)