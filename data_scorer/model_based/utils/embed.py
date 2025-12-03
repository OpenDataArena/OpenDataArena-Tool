import json
import argparse
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from vllm import LLM


def tokenize_and_truncate(
    texts: List[str],
    tokenizer,
    max_tokens: int = 32768,
    batch_size: int = 2048,
) -> Tuple[List[str], List[int]]:
    """
    Tokenize texts in batches using the given tokenizer, identify samples exceeding max_tokens, and truncate them.
    Returns:
        - List of truncated texts (aligned with original texts)
        - List of original indices of truncated samples
    Notes:
        - Uses batch processing (encode_batch), significantly faster than encoding one by one
        - Truncation is done by slicing token ids and decoding back to strings
    """
    truncated_indices: List[int] = []
    out_texts: List[str] = list(texts)  # Make a copy to modify as needed

    for start in tqdm(range(0, len(texts), batch_size), desc="Tokenizing (batch)", unit="batch"):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        # Batch tokenization; add_special_tokens=False avoids introducing model-specific special tokens that may cause length discrepancies
        tokenized = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids_batch = tokenized["input_ids"]

        # Check each item for length overflow; truncate and decode back to text if necessary
        for j, ids in enumerate(input_ids_batch):
            if len(ids) > max_tokens:
                global_idx = start + j
                truncated_indices.append(global_idx)
                truncated_ids = ids[:max_tokens]
                # Also avoid adding special tokens when decoding
                out_texts[global_idx] = tokenizer.decode(truncated_ids)

    return out_texts, truncated_indices


def get_embeddings(sentences: List[str], model: LLM, batch_size: int = 16384) -> np.ndarray:
    """
    Compute embedding vectors for a list of sentences.
        - Parameter model: An initialized vLLM.LLM object with task="embed" specified during construction
        - Returns: np.ndarray with shape (N, D)
    """
    if not sentences:
        return np.empty((0, 0), dtype=np.float32)

    all_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Computing embeddings", unit="batch"):
        batch_texts = sentences[i:i + batch_size]
        # vLLM embed interface: returns a list of objects, each object's o.outputs.embedding is the vector
        outputs = model.embed(batch_texts)
        # Note: Different vLLM versions may have slightly different object structures; this follows your original pattern
        batch_vecs = [np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs]
        all_vecs.extend(batch_vecs)

    return np.vstack(all_vecs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sentence embeddings and save as .npy file (with token truncation preprocessing).")
    parser.add_argument(
        "--embedder_model",
        default="/mnt/petrelfs/gaoxin/local_models/all-MiniLM-L6-v2",
        help="Path or name of the vLLM model for computing embeddings (task=embed)."
    )
    parser.add_argument(
        "--input_path",
        help="Path to the input JSONL file.",
        default="/mnt/hwfile/opendatalab/air/data/data-leaderboard/dolly/train.jsonl"
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the output .npy file.",
        default="./large_embeddings.npy"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens allowed per text; texts exceeding this will be truncated. Default: 32768."
    )
    parser.add_argument(
        "--truncate_report_path",
        type=str,
        default="",
        help="Optional: Write line numbers of truncated samples to this text file (one line number per line). Leave empty to skip."
    )
    parser.add_argument(
        "--tokenize_batch_size",
        type=int,
        default=16384,
        help="Batch size for tokenization (encode_batch). Default: 2048, can be adjusted based on memory."
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=16384,
        help="Batch size for embedding computation. Default: 16384, can be adjusted based on GPU/memory."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embedder = LLM(model=args.embedder_model, task="embed", tensor_parallel_size=4)
    # ========== 1) Initialize tokenizer (using the embedder model for token counting) ==========
    tokenizer = embedder.get_tokenizer()

    # ========== 2) Read JSONL and assemble texts to encode ==========
    global_data: List[str] = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:

            data = json.loads(line)
            if data.get("input", "") == "":
                global_data.append(data["instruction"])
            else:
                global_data.append(data["instruction"] + "\n" + data["input"])

    # ========== 3) Tokenize and truncate long samples first ==========
    global_data_trunc, truncated_indices = tokenize_and_truncate(
        global_data,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        batch_size=args.tokenize_batch_size,
    )
    print(f"[Info] Number of samples exceeding {args.max_tokens} tokens: {len(truncated_indices)}")

    # Optional: Write line numbers of truncated samples to file
    if args.truncate_report_path:
        with open(args.truncate_report_path, "w", encoding="utf-8") as fw:
            for idx in truncated_indices:
                fw.write(str(idx) + "\n")
        print(f"[Info] Line numbers of truncated samples written to: {args.truncate_report_path}")

    # ========== 4) Initialize vLLM model for computing embeddings ==========

    print("max_model_len (embedder):", embedder.llm_engine.model_config.max_model_len)

    # ========== 5) Compute embeddings ==========
    embeddings = get_embeddings(global_data_trunc, embedder, batch_size=args.embed_batch_size)

    # ========== 6) Save as float64 .npy ==========
    embeddings_float64 = embeddings.astype(np.float64)
    np.save(args.output_path, embeddings_float64)

    print(f"✅ Embeddings successfully saved to {args.output_path}")
    print(f"📐 Array shape: {embeddings_float64.shape}")
    print(f"🧪 Data type: {embeddings_float64.dtype}")