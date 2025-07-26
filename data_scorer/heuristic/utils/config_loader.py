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
