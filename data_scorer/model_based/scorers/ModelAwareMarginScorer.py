import json
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines


class ModelAwareMarginScorer(BaseScorer):
    def _validate_config(self):
        if (
            "model" not in self.config
            or not isinstance(self.config["model"], str)
            or not self.config["model"].strip()
        ):
            print("Warning: No model specified. Using default: Qwen/Qwen3-Embedding-8B")
            self.config["model"] = "Qwen/Qwen3-Embedding-8B"
        else:
            print(f"Using embedding model: {self.config['model']}")

        if (
            "k" not in self.config
            or not isinstance(self.config["k"], int)
            or self.config["k"] <= 0
        ):
            print("Warning: No/invalid k specified. Using default k=5.")
            self.config["k"] = 5
        else:
            print(f"Using k={self.config['k']}")

        if (
            "batch_size" not in self.config
            or not isinstance(self.config["batch_size"], int)
            or self.config["batch_size"] <= 0
        ):
            print("Warning: No/invalid batch_size, using default 64.")
            self.config["batch_size"] = 64
        else:
            print(f"Using batch_size={self.config['batch_size']}")

        if "max_length" in self.config:
            if (
                not isinstance(self.config["max_length"], int)
                or self.config["max_length"] <= 0
            ):
                print("Warning: Invalid max_length. Ignoring and using model default.")
                self.config.pop("max_length", None)
            else:
                print(f"Using max_length={self.config['max_length']}")

    def _setup(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "sentence_transformers is required for ModelAwareMarginScorer. Please install it via `pip install sentence-transformers`."
            ) from e

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        model_id: str = self.config["model"]
        try:
            # Prefer enabling flash attention v2 for acceleration and memory savings
            self.model = SentenceTransformer(
                model_id,
                model_kwargs={"attn_implementation": "flash_attention_2"},
            )
        except Exception as e:
            print(
                f"Warning: Failed to enable flash_attention_2 for '{model_id}' ({e}). Retrying without it."
            )
            try:
                self.model = SentenceTransformer(model_id)
            except Exception as e2:
                print(
                    f"Warning: Failed to load specified model '{model_id}' without flash_attention_2 ({e2}). Trying default Qwen/Qwen3-Embedding-8B"
                )
                self.model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

        if "max_length" in self.config:
            try:
                self.model.max_seq_length = int(self.config["max_length"])  # type: ignore[attr-defined]
            except Exception:
                pass

        self.model.eval()
        print("Setting up ModelAwareMarginScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        raise NotImplementedError(
            "ModelAwareMarginScorer does not support single item scoring. Use evaluate() for batch processing."
        )

    def _extract_text(self, item: Dict[str, Any]) -> Optional[str]:
        val = str(item.get("instruction")).strip()
        return val if val else None

    def evaluate(self, dataset: str) -> List[Dict]:
        try:
            from sentence_transformers import util
        except Exception as e:
            raise RuntimeError(
                "sentence_transformers is required for ModelAwareMarginScorer. Please install it via `pip install sentence-transformers`."
            ) from e

        num_lines = get_total_lines(dataset)

        all_items: List[Dict[str, Any]] = []
        all_ids: List[Any] = []
        texts: List[str] = []
        valid_indices: List[int] = []

        with open(dataset, "r", encoding="utf-8") as f:
            pbar = tqdm(
                total=num_lines, desc=self.config.get("name", "ModelAwareMarginScorer")
            )
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                all_items.append(item)
                all_ids.append(item.get("id", ""))
                text = self._extract_text(item)
                if text is not None and len(text) > 0:
                    valid_indices.append(idx)
                    texts.append(text)
                pbar.update(1)
            pbar.close()

        results: List[Dict] = [
            {"id": _id, "Model_Aware_Margin": None} for _id in all_ids
        ]

        if len(valid_indices) <= 1:
            return results

        batch_size = int(self.config["batch_size"]) or 64

        with torch.no_grad():
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=True,
                )
            except TypeError:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=True,
                )

        k: int = int(self.config["k"])  # guaranteed >0
        effective_k = min(k, embeddings.size(0) - 1)
        if effective_k <= 0:
            return results

        hits = util.semantic_search(
            query_embeddings=embeddings,
            corpus_embeddings=embeddings,
            top_k=effective_k + 1,  # include self then drop
            score_function=util.cos_sim,
        )

        for local_idx, global_idx in enumerate(valid_indices):
            neighbors = hits[local_idx]
            filtered = []
            for h in neighbors:
                if h["corpus_id"] != local_idx:
                    filtered.append(h)
                if len(filtered) == effective_k:
                    break
            if not filtered:
                continue
            neighbor_scores = []
            for h in filtered:
                sc = h.get("score", 0.0)
                if isinstance(sc, torch.Tensor):
                    sc = sc.item()
                neighbor_scores.append(float(sc))
            avg_score = sum(neighbor_scores) / len(neighbor_scores)
            results[global_idx] = {
                "id": all_ids[global_idx],
                "Model_Aware_Margin": avg_score,
            }

        return results
