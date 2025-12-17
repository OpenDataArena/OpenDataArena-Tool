import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from bisect import bisect_left, bisect_right
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines


class AnswerProbScorer(BaseScorer):
    """
    Configuration example:
    {
        "name": "AnswerProbScorer",
        "model": "Qwen/Qwen2.5-7B",
        "case_sensitive": true,
        "batch_size": 16,
        "max_length": 2048
    }
    """

    # ========= Basic Configuration =========
    def _validate_config(self):
        if "model" not in self.config:
            print("Warning: No model specified. Use default 'Qwen/Qwen3-8B'.")
            self.config["model"] = "Qwen/Qwen3-8B"

        # Case sensitivity
        if "case_sensitive" not in self.config:
            self.config["case_sensitive"] = True

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 1
            print("Warning: No/invalid batch_size, use default 1.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

        # max_length optional, default to 2048
        if "max_length" not in self.config:
            self.config["max_length"] = 2048
            print("Warning: No max_length specified, use default 2048.")
        elif not isinstance(self.config["max_length"], int) or self.config["max_length"] <= 0:
            self.config["max_length"] = 2048
            print("Warning: Invalid max_length, use default 2048.")
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")

    def _setup(self):
        # -------- Device and precision strategy --------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Choose memory-efficient weight precision (CUDA: bf16 first, then fp16; CPU: fp32)
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                load_dtype = torch.bfloat16
            else:
                load_dtype = torch.float16
            self.autocast_dtype = load_dtype
        else:
            load_dtype = torch.float32
            self.autocast_dtype = None  # No autocast for CPU

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"], use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model weights with low memory mode at target precision
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"],
            trust_remote_code=True,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Disable KV cache to avoid unnecessary generation cache
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        # Enable TF32 (if hardware supports) for better speed/memory
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        self.case_sensitive = bool(self.config.get("case_sensitive", True))
        self.max_length = int(self.config.get("max_length", 2048))

        print(f"AnswerProbScorer set up on {self.device} with dtype={load_dtype}.")

    # ========= Utility: BOS-related =========
    def _get_bos_like_id(self) -> int:
        """
        Select an available "BOS-like" token id:
        Priority: bos_token_id, then eos_token_id, then pad_token_id, finally fallback to 0.
        """
        for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
            tid = getattr(self.tokenizer, k, None)
            if tid is not None:
                return tid
        return 0

    @staticmethod
    def _prepend_bos_like(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bos_id: int,
        offsets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepend a BOS-like token to the beginning of each row in the batch.
        - input_ids: [B, T]
        - attention_mask: [B, T]
        - offsets: Optional [B, T, 2] (if provided, will prepend (0,0))
        Returns:
        - new_input_ids: [B, T+1]
        - new_attention_mask: [B, T+1]
        - new_offsets: [B, T+1, 2] or None
        """
        B, T = input_ids.size()
        device = input_ids.device

        bos_col = torch.full((B, 1), bos_id, dtype=input_ids.dtype, device=device)
        att_col = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)

        new_input_ids = torch.cat([bos_col, input_ids], dim=1)
        new_attention = torch.cat([att_col, attention_mask], dim=1)

        if offsets is not None:
            # Offsets are typically CPU tensors in HF; move to same device as input
            off = offsets
            if off.device != device:
                off = off.to(device)
            bos_off = torch.zeros((B, 1, 2), dtype=off.dtype, device=device)  # (0,0)
            new_offsets = torch.cat([bos_off, off], dim=1)
            # Move back to CPU for subsequent list operations
            return new_input_ids, new_attention, new_offsets.to("cpu")
        else:
            return new_input_ids, new_attention, None

    # ========= Memory-efficient Forward =========
    def _forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass returning logits, with autocast + inference_mode and cache disabled.
        """
        amp_ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype) \
            if (self.device.type == "cuda" and self.autocast_dtype is not None) else nullcontext()
        with torch.inference_mode():
            with amp_ctx:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                logits = outputs.logits  # [B, T, V]
        return logits

    # ========= Public Interface =========
    def score_item(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        A) Question+Answer → average log conditional probability of answer token segment
        B) Answer only → average log conditional probability (baseline)
        C) normalized = A - B = log(P_A / P_B)  (compute probability ratio in log space)
        """
        # 1) Extract and concatenate
        answers_list: List[List[str]] = []
        answer_str_list: List[str] = []
        concat_texts: List[str] = []
        ids: List[Any] = []

        for item in data_items:
            instruction = item["instruction"]
            input_text = item.get("input", "")
            if input_text:
                instruction += "\n" + input_text
            output = item["output"]
            
            # If "answer" key exists and is not empty, use it directly; otherwise extract from output
            if "answer" in item and item["answer"]:
                answer_value = item["answer"]
                # Convert to list if it's a string
                if isinstance(answer_value, str):
                    answers = [answer_value]
                elif isinstance(answer_value, list):
                    answers = answer_value
                else:
                    answers = [str(answer_value)]
            else:
                answers = self._extract_boxed(output)
            
            answer_str = ",".join(answers) if answers else ""
            concat_text = instruction + "\n" + answer_str

            answers_list.append(answers)
            answer_str_list.append(answer_str)
            concat_texts.append(concat_text)
            ids.append(item.get("id", ""))

        has_answer = [bool(s) for s in answer_str_list]

        # 2) tokenizer
        enc = self.tokenizer(
            concat_texts,
            return_offsets_mapping=True,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length - 1,  # Reserve 1 position for BOS token
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets_batch = enc["offset_mapping"]  # CPU Tensor

        # 2.1 Check for truncation before prepending BOS
        for i in range(input_ids.size(0)):
            seq_len = int(attention_mask[i].sum().item())
            if seq_len >= self.max_length - 1:
                print(f"Warning: Sample {i} (id={ids[i]}) was truncated. Original text length may exceed max_length ({self.max_length}).")

        # 2.2 Prepend BOS-like token
        bos_id = self._get_bos_like_id()
        input_ids, attention_mask, offsets_batch = self._prepend_bos_like(
            input_ids, attention_mask, bos_id, offsets_batch
        )

        Bsz, T_max = input_ids.size()

        # 3) Forward pass (question+answer)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        logits = self._forward_logits(input_ids, attention_mask)  # [B, T, V]
        logits = logits.to("cpu")  # Move back to CPU for subsequent computation

        # 4) Compute average log probability for answer segment (A) - all on CPU to reduce memory
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids.to("cpu")[:, 1:]
        shift_mask = attention_mask.to("cpu")[:, 1:]

        logprobs_all = F.log_softmax(shift_logits, dim=-1)  # CPU
        token_logprobs_all = logprobs_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        A_mean_logprobs: List[Optional[float]] = [None] * Bsz
        A_token_counts: List[int] = [0] * Bsz

        for i in range(Bsz):
            T_i = int(attention_mask[i].sum().item())
            # Convenient structure aligned to token indices: index 0 is placeholder (for BOS), starting from 1 is log prob of first real token
            logs_full: List[Optional[float]] = [None]
            if T_i > 1:
                # token_logprobs_all is on CPU, directly convert to list
                logs_full += token_logprobs_all[i, :T_i-1].tolist()

            if not has_answer[i] or T_i <= 1:
                A_mean_logprobs[i] = None
                A_token_counts[i] = 0
                continue

            A_text = concat_texts[i]
            B_text = answer_str_list[i]
            offsets = offsets_batch[i, :T_i].tolist()  # CPU list [[s, e], ...]

            spans = self._map_char_to_token_spans(A_text, B_text, offsets)
            if not spans:
                A_mean_logprobs[i] = None
                A_token_counts[i] = 0
                continue

            last_span = max(spans, key=lambda s: s["char_span"][0])
            tok_start, tok_end = last_span["token_span"]  # Left-closed, right-open

            slice_logs = [v for v in logs_full[tok_start:tok_end] if v is not None]
            if len(slice_logs) == 0:
                A_mean_logprobs[i] = None
                A_token_counts[i] = 0
            else:
                A_mean_logprobs[i] = sum(slice_logs) / len(slice_logs)  # Average log probability
                A_token_counts[i] = len(slice_logs)

        # 5) Answer only (B, as denominator) - average log probability (micro-batch + back to CPU)
        idx_with_ans = [i for i, ok in enumerate(has_answer) if ok]
        ans_only_texts = [answer_str_list[i] for i in idx_with_ans]

        B_mean_logprobs_map: Dict[int, Optional[float]] = {}
        B_token_counts_map: Dict[int, int] = {}

        if len(ans_only_texts) > 0:
            enc_ans = self.tokenizer(
                ans_only_texts,
                return_offsets_mapping=False,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.max_length - 1,  # Reserve 1 position for BOS token
                return_tensors="pt",
            )
            ans_input_ids = enc_ans["input_ids"]
            ans_attention = enc_ans["attention_mask"]

            # Check for truncation in answer-only sequences
            for j in range(ans_input_ids.size(0)):
                seq_len = int(ans_attention[j].sum().item())
                if seq_len >= self.max_length - 1:
                    sample_idx = idx_with_ans[j]
                    print(f"Warning: Answer-only for sample {sample_idx} (id={ids[sample_idx]}) was truncated. Answer text may exceed max_length ({self.max_length}).")

            ans_input_ids, ans_attention, _ = self._prepend_bos_like(
                ans_input_ids, ans_attention, bos_id, offsets=None
            )

            # Forward computation
            ans_input_ids = ans_input_ids.to(self.device)
            ans_attention = ans_attention.to(self.device)
            ans_logits = self._forward_logits(ans_input_ids, ans_attention)  # [N, L, V]
            ans_logits = ans_logits.to("cpu")  # Move back to CPU

            ans_shift_logits = ans_logits[:, :-1, :]
            ans_shift_labels = ans_input_ids.to("cpu")[:, 1:]
            # ans_shift_mask = ans_attention.to("cpu")[:, 1:]  # Only need length, not involved in gather

            ans_logprobs_all = F.log_softmax(ans_shift_logits, dim=-1)
            ans_token_logprobs = ans_logprobs_all.gather(-1, ans_shift_labels.unsqueeze(-1)).squeeze(-1)  # [N, L-1]

            N = ans_input_ids.size(0)
            for j in range(N):
                L_j = int(ans_attention[j].sum().item())
                if L_j <= 1:
                    B_mean_logprobs_map[idx_with_ans[j]] = None
                    B_token_counts_map[idx_with_ans[j]] = 0
                    continue
                seq_logs = ans_token_logprobs[j, :L_j-1]  # CPU
                mean_log = float(seq_logs.mean().item()) if seq_logs.numel() > 0 else None
                B_mean_logprobs_map[idx_with_ans[j]] = mean_log
                B_token_counts_map[idx_with_ans[j]] = int(seq_logs.numel())

        # 6) Assemble results (with normalization)
        results: List[Dict[str, Any]] = []
        for i in range(Bsz):
            mean_log_A = A_mean_logprobs[i]
            mean_log_B = B_mean_logprobs_map.get(i, None) if has_answer[i] else None

            if (mean_log_A is None) or (mean_log_B is None):
                normalized = None
            else:
                # Correct normalization: log(P_A) - log(P_B) = log(P_A / P_B)
                # Compute probability ratio in log space to avoid numerical underflow
                normalized = mean_log_A - mean_log_B

            results.append({
                "id": ids[i],
                "mean_prob": mean_log_A,                     # Average log probability
                "token_count": A_token_counts[i],
                "answers": answers_list[i],
                "answer_str": answer_str_list[i],
                "mean_prob_answer_only": mean_log_B,         # Average log probability
                "score": normalized,          # = log(P_A / P_B), log of probability ratio
                "answer_only_token_count": B_token_counts_map.get(i, 0) if has_answer[i] else 0,
            })

        return results

    def evaluate(self, dataset_path: str) -> List[Dict[str, Any]]:
        num_lines = get_total_lines(dataset_path)
        results: List[Dict[str, Any]] = []

        bs = self.config.get("batch_size", 32)
        buf: List[Dict[str, Any]] = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            pbar = tqdm(total=num_lines, desc=self.config.get("name", "AnswerProbScorer"))
            for line in f:
                item = json.loads(line.strip())
                buf.append(item)
                if len(buf) == bs:
                    results.extend(self.score_batch(buf))
                    buf.clear()
                pbar.update(1)
            if buf:
                results.extend(self.score_batch(buf))
                buf.clear()
            pbar.close()
        return results

    # ========= Internal Utilities =========
    @staticmethod
    def _extract_boxed(text: str) -> List[str]:
        """Extract content from \\boxed{...}, supporting nested braces."""
        results = []
        i = 0
        while i < len(text):
            if text[i:i+7] == "\\boxed{":
                i += 7
                brace = 1
                start = i
                while i < len(text) and brace > 0:
                    if text[i] == "{":
                        brace += 1
                    elif text[i] == "}":
                        brace -= 1
                    i += 1
                if brace == 0:
                    content = text[start:i-1].strip()
                    if content:
                        results.append(content)
            else:
                i += 1
        return results

    def _map_char_to_token_spans(self, A: str, B: str, offsets: List[List[int]]) -> List[Dict[str, Any]]:
        """Map all character-level occurrences of B in A to token ranges (based on offsets)."""
        if not B:
            return []

        starts = [s for (s, e) in offsets]
        ends   = [e for (s, e) in offsets]

        haystack = A if self.case_sensitive else A.lower()
        needle   = B if self.case_sensitive else B.lower()

        spans = []
        for m in re.finditer(re.escape(needle), haystack):
            c0 = m.start()
            c1 = c0 + len(B)
            tok_start = bisect_right(ends, c0)
            tok_end   = bisect_left(starts, c1)
            if tok_start > tok_end:
                # Extreme fallback tolerance
                tok_start = next((i for i, (s, e) in enumerate(offsets) if e > c0), None)
                tok_end = next((i for i, (s, e) in enumerate(offsets) if s >= c1), len(offsets))
                if tok_start is None:
                    continue
            spans.append({
                "char_span": (c0, c1),
                "token_span": (tok_start, tok_end),  # Left-closed, right-open
            })
        return spans