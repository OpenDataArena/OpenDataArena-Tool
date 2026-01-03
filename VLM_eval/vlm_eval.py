#!/usr/bin/env python3
"""
Script to evaluate VLM predictions using CompassVerifier model via vLLM.
Reads xlsx files from a directory and evaluates predictions.
"""

import argparse
import os
import glob
import pandas as pd
from typing import List
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# GRADER_TEMPLATE from hmmt2025_repeat8_cver_gen.py
GRADER_TEMPLATE = """
    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
    2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
    3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
    4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
    5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
    6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT 
    B: INCORRECT
    C: INVALID
    Just return the letters "A", "B", or "C", with no text around it.
    Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
    <Original Question Begin>:
    {problem}
    <Original Question End>
    <Standard Answer Begin>:
    {answer}
    <Standard Answer End>
    <Candidate's Answer Begin>: 
    {prediction}
    <Candidate's Answer End>
    Judging the correctness of the candidate's answer: 
""".strip()

SYSTEM_PROMPT = "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly."


def load_model(model_path: str, tensor_parallel_size: int = 1):
    """Load the CompassVerifier model using vLLM."""
    import os
    
    # Print GPU information
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch detected {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            print("PyTorch CUDA not available")
    except ImportError:
        print("PyTorch not available for GPU info")
    
    print(f"Loading model from: {model_path}")
    
    # Initialize Ray if needed (vLLM uses Ray for distributed inference)
    import ray
    if ray.is_initialized():
        print("Ray is already initialized, shutting down...")
        ray.shutdown()
    
    # Load model with vLLM
    # Note: max_model_len is set to 32768 (model's max_position_embeddings)
    # If you need longer sequences, set env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=32768,  # Changed from 40960 to match model's max_position_embeddings
    )
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    print("Model loaded successfully!")
    return model, tokenizer


def format_prompt(question: str, answer: str, prediction: str) -> List[dict]:
    """Format the prompt for the verifier model as chat messages."""
    prompt = GRADER_TEMPLATE.format(
        problem=question,
        answer=answer,
        prediction=prediction
    )
    
    # Format as chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    return messages


def read_xlsx_files(folder_path: str, filter_string: str = None) -> List[str]:
    """Find all xlsx files in the folder, excluding _acc.xlsx and _results.xlsx files.
    
    Args:
        folder_path: Path to folder containing xlsx files
        filter_string: Optional string to filter files by name. Only files containing this string will be returned.
    
    Returns:
        List of xlsx file paths
    """
    xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    # Filter out files ending with _acc.xlsx or _results.xlsx
    filtered_files = [
        f for f in xlsx_files 
        if not (f.endswith('_acc.xlsx') or f.endswith('_results.xlsx'))
    ]
    
    # Apply filter_string if provided
    if filter_string:
        filtered_files = [
            f for f in filtered_files 
            if filter_string in os.path.basename(f)
        ]
    
    return sorted(filtered_files)


def process_xlsx_file(file_path: str, model: LLM, tokenizer) -> pd.DataFrame:
    """Process a single xlsx file and add verification results."""
    print(f"\nProcessing file: {file_path}")
    
    # Read the xlsx file
    df = pd.read_excel(file_path)
    
    # Normalize column names to lowercase for case-insensitive matching
    df.columns = df.columns.str.lower().str.strip()
    
    # Check if required columns exist
    required_columns = ['question', 'answer', 'prediction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns {missing_columns} in {file_path}. Skipping...")
        return df
    
    # Check if this is a MathVista_MINI file
    file_basename = os.path.basename(file_path)
    is_mathvista_mini = 'MathVista_MINI' in file_basename
    
    # Prepare prompts and format them using tokenizer's chat template
    formatted_prompts = []
    for idx, row in df.iterrows():
        question = str(row['question']) if pd.notna(row['question']) else ""
        answer = str(row['answer']) if pd.notna(row['answer']) else ""
        prediction = str(row['prediction']) if pd.notna(row['prediction']) else ""
        
        # Special handling for MathVista_MINI files: use answer_option if available
        if is_mathvista_mini and 'answer_option' in df.columns:
            answer_option = row['answer_option']
            if pd.notna(answer_option) and str(answer_option).strip():
                # Use answer_option if it's not empty (e.g., A, B, C, D, E)
                answer = str(answer_option).strip()
        
        messages = format_prompt(question, answer, prediction)
        # Apply chat template to format messages
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Remove bos_token if present (vLLM handles this automatically)
        if tokenizer.bos_token and formatted_prompt.startswith(tokenizer.bos_token):
            formatted_prompt = formatted_prompt.removeprefix(tokenizer.bos_token)
        
        formatted_prompts.append(formatted_prompt)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=1e-6,
        top_p=0.9,
        top_k=1,
        max_tokens=32768,
    )
    
    # Generate predictions in batches
    batch_size = 16
    all_results = []
    
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(formatted_prompts) + batch_size - 1)//batch_size} ({len(batch)} items)")
        
        try:
            # Call vLLM model
            outputs = model.generate(batch, sampling_params)
            
            # Extract text from outputs
            batch_results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                batch_results.append(generated_text)
            
            all_results.extend(batch_results)
        except Exception as e:
            print(f"  Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            # Add empty results for failed batch
            all_results.extend([""] * len(batch))
    
    # Parse results to extract judgment (A/B/C)
    def extract_judge(result: str) -> str:
        """Extract A, B, or C from model output."""
        if not result:
            return "UNKNOWN"
        result_upper = result.strip().upper()
        # Look for A, B, C at the beginning or in the text
        if result_upper.startswith('A'):
            return "A"
        elif result_upper.startswith('B'):
            return "B"
        elif result_upper.startswith('C'):
            return "C"
        # If not found at start, search for single letter A/B/C
        elif ' A ' in result_upper or result_upper.endswith(' A'):
            return "A"
        elif ' B ' in result_upper or result_upper.endswith(' B'):
            return "B"
        elif ' C ' in result_upper or result_upper.endswith(' C'):
            return "C"
        else:
            # Fallback: try to find CORRECT/INCORRECT/INVALID
            if 'CORRECT' in result_upper:
                return "A"
            elif 'INCORRECT' in result_upper:
                return "B"
            elif 'INVALID' in result_upper:
                return "C"
            else:
                return result_upper[:10]  # Return first 10 chars if unclear
    
    # Extract judge results (A/B/C)
    df['judge'] = [extract_judge(result) for result in all_results]
    
    # Add judge_prompt column with the full formatted prompt sent to the model
    df['judge_prompt'] = formatted_prompts
    
    # Check if this is MMMU_DEV_VAL file and needs to be split
    is_mmmu_dev_val = 'MMMU_DEV_VAL' in file_basename
    has_split_column = 'split' in df.columns
    
    if is_mmmu_dev_val and has_split_column:
        # Split MMMU_DEV_VAL into DEV and VAL
        print(f"  Detected MMMU_DEV_VAL file with split column, splitting...")
        
        # Split data by split column
        split_mapping = {
            'dev': 'MMMU_DEV',
            'validation': 'MMMU_VAL'
        }
        
        for split_value, split_name in split_mapping.items():
            df_split = df[df['split'] == split_value].copy()
            
            if len(df_split) == 0:
                print(f"  Warning: No data found for split '{split_value}', skipping...")
                continue
            
            # Create results file for this split
            base_name = file_basename.replace('MMMU_DEV_VAL', split_name).replace('.xlsx', '')
            results_path = os.path.join(os.path.dirname(file_path), f"{base_name}_results.xlsx")
            df_split.to_excel(results_path, index=False)
            print(f"  Saved {split_name} results to: {results_path}")
            
            # Calculate accuracy for this split
            total_count = len(df_split)
            a_count = (df_split['judge'] == 'A').sum()
            accuracy = a_count / total_count if total_count > 0 else 0.0
            
            # Create accuracy file for this split
            acc_data = {
                'metric': ['total', 'A_count', 'accuracy'],
                'value': [total_count, a_count, accuracy]
            }
            df_acc = pd.DataFrame(acc_data)
            acc_path = os.path.join(os.path.dirname(file_path), f"{base_name}_acc.xlsx")
            df_acc.to_excel(acc_path, index=False)
            print(f"  Saved {split_name} accuracy to: {acc_path}")
            
            # Print summary for this split
            judge_counts = df_split['judge'].value_counts()
            print(f"  {split_name} Summary: {dict(judge_counts)}")
            print(f"  {split_name} Accuracy: {accuracy:.4f} ({a_count}/{total_count})")
        
        # Return the full dataframe
        return df
    else:
        # Normal processing for other files
        # Create results file: copy original content and add judge column
        df_results = df.copy()
        results_path = file_path.replace('.xlsx', '_results.xlsx')
        df_results.to_excel(results_path, index=False)
        print(f"  Saved results to: {results_path}")
        
        # Calculate accuracy (A / total)
        total_count = len(df_results)
        a_count = (df_results['judge'] == 'A').sum()
        accuracy = a_count / total_count if total_count > 0 else 0.0
        
        # Create accuracy file
        acc_data = {
            'metric': ['total', 'A_count', 'accuracy'],
            'value': [total_count, a_count, accuracy]
        }
        df_acc = pd.DataFrame(acc_data)
        acc_path = file_path.replace('.xlsx', '_acc.xlsx')
        df_acc.to_excel(acc_path, index=False)
        print(f"  Saved accuracy to: {acc_path}")
        
        # Print summary
        judge_counts = df_results['judge'].value_counts()
        print(f"  Summary: {dict(judge_counts)}")
        print(f"  Accuracy: {accuracy:.4f} ({a_count}/{total_count})")
        
        return df_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM predictions using CompassVerifier')
    parser.add_argument('folder', type=str, help='Path to folder containing xlsx files')
    parser.add_argument(
        '--model_path',
        type=str,
        default='opencompass/CompassVerifier-7B',
        help='Path to CompassVerifier model'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Filter string: only process xlsx files whose filename contains this string'
    )
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        return
    
    # Load model
    model, tokenizer = load_model(args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    
    # Find all xlsx files
    xlsx_files = read_xlsx_files(args.folder, filter_string=args.filter)
    
    if not xlsx_files:
        if args.filter:
            print(f"No xlsx files found in {args.folder} matching filter '{args.filter}'")
        else:
            print(f"No xlsx files found in {args.folder}")
        return
    
    if args.filter:
        print(f"\nFound {len(xlsx_files)} xlsx file(s) matching filter '{args.filter}'")
    else:
        print(f"\nFound {len(xlsx_files)} xlsx file(s)")
    
    # Process each file
    all_results = []
    for xlsx_file in xlsx_files:
        try:
            df_result = process_xlsx_file(xlsx_file, model, tokenizer)
            all_results.append(df_result)
        except Exception as e:
            print(f"Error processing {xlsx_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("All files processed!")


if __name__ == "__main__":
    main()
