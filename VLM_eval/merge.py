#!/usr/bin/env python3
"""
Script to merge all accuracy results from *_acc.xlsx files into a single results.xlsx file.
"""

import argparse
import os
import glob
import pandas as pd
from typing import Dict, List, Optional
import re

# Define the order of test sets
TEST_SET_ORDER = [
    'CV-Bench-2D',
    'CV-Bench-3D',
    'MathVista_MINI',
    'MathVision_MINI',
    'MathVerse_MINI',
    'Dynamath',
    'LogicVista',
    'VisuLogic',
    'AI2D',
    'ScienceQA',
    'SEEDBench2',
    'MMBench_DEV_EN_V11',
    'RealWorldQA',
    'MMStar',
    'MMMU_VAL',
    'MMMU_DEV',
    'ChartQA_TEST',
    'OCRBench',
    'CharXiv_reasoning_val',
    'CharXiv_descriptive_val',
]


def extract_test_set_name(filename: str, model_prefix: Optional[str] = None) -> Optional[str]:
    """
    Extract test set name from filename.
    Example: Qwen3-VL-8B-Instruct_AI2D_TEST_acc.xlsx -> AI2D_TEST
    
    Args:
        filename: The filename to extract from
        model_prefix: Model name prefix to remove (e.g., "Qwen3-VL-8B-Instruct")
    """
    # Remove .xlsx extension and _acc suffix
    basename = os.path.basename(filename).replace('_acc.xlsx', '')
    
    # If model prefix is provided, try to remove it first
    if model_prefix:
        if basename.startswith(model_prefix + '_'):
            test_set_name = basename[len(model_prefix) + 1:]  # Remove prefix and underscore
            return test_set_name if test_set_name else None
    
    # Fallback: try common model name patterns
    common_prefixes = [
        'Qwen3-VL-8B-Instruct',
        'Qwen3-VL-8B',
        'Qwen3-VL',
    ]
    
    # Try to remove model prefix
    test_set_name = basename
    for prefix in common_prefixes:
        if basename.startswith(prefix + '_'):
            test_set_name = basename[len(prefix) + 1:]  # Remove prefix and underscore
            break
    
    # If still contains model name pattern, try splitting by underscore
    if test_set_name == basename:
        parts = basename.split('_')
        # Model names typically have multiple parts, test set name usually starts after "Instruct"
        # Find "Instruct" and take everything after it
        if 'Instruct' in parts:
            idx = parts.index('Instruct')
            if idx + 1 < len(parts):
                test_set_name = '_'.join(parts[idx + 1:])
        else:
            # Fallback: take last 2-4 parts (most test sets have 1-3 parts)
            # Try from the end
            for i in range(len(parts) - 4, len(parts)):
                if i >= 0:
                    potential = '_'.join(parts[i:])
                    # Check if it matches a known test set
                    if potential in TEST_SET_ORDER or potential.endswith('_TEST'):
                        test_set_name = potential
                        break
    
    return test_set_name if test_set_name else None


def normalize_test_set_name(name: str) -> str:
    """Normalize test set name to match the order list."""
    # Special handling for variations
    name_mapping = {
        'AI2D_TEST': 'AI2D',
        'ScienceQA_TEST': 'ScienceQA',
        'MMMU_DEV_VAL': 'MMMU_VAL',  # Map DEV_VAL to VAL (for backward compatibility)
        'MMMU_DEV': 'MMMU_DEV',  # Keep as is
        'MMMU_VAL': 'MMMU_VAL',  # Keep as is
    }
    
    # Check mapping first
    if name in name_mapping:
        return name_mapping[name]
    
    # Check if name exists in order list
    if name in TEST_SET_ORDER:
        return name
    
    # Remove _TEST suffix for matching
    if name.endswith('_TEST'):
        base = name.replace('_TEST', '')
        # Check if base exists in order list
        if base in TEST_SET_ORDER:
            return base
        # Otherwise keep _TEST suffix if it's in the list
        if name in TEST_SET_ORDER:
            return name
    
    return name


def read_acc_file(file_path: str) -> Dict[str, float]:
    """Read accuracy file and return metrics as dictionary."""
    try:
        df = pd.read_excel(file_path)
        # Convert to dictionary: metric -> value
        metrics = {}
        for _, row in df.iterrows():
            metric = row['metric']
            value = row['value']
            metrics[metric] = value
        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}


def find_test_set_in_order(test_set_name: str) -> int:
    """Find the position of test set in the ordered list."""
    normalized = normalize_test_set_name(test_set_name)
    
    # Handle MMMU_DEV_VAL: if it's still in the name, map to MMMU_VAL position
    # (though it should already be split by vlm_eval.py)
    if 'MMMU_DEV_VAL' in normalized:
        normalized = 'MMMU_VAL'
    
    # Try exact match first
    if normalized in TEST_SET_ORDER:
        return TEST_SET_ORDER.index(normalized)
    
    # Try matching without _TEST suffix
    base_name = normalized.replace('_TEST', '')
    if base_name in TEST_SET_ORDER:
        return TEST_SET_ORDER.index(base_name)
    
    # Try matching with _TEST suffix
    test_name = f"{normalized}_TEST"
    if test_name in TEST_SET_ORDER:
        return TEST_SET_ORDER.index(test_name)
    
    # Try partial match (e.g., MMMU_DEV_VAL -> MMMU_VAL)
    for i, ordered_name in enumerate(TEST_SET_ORDER):
        if normalized.startswith(ordered_name) or ordered_name in normalized:
            return i
    
    # Not found, return a large number to put it at the end
    return len(TEST_SET_ORDER)


def merge_acc_files(folder_path: str, output_file: str = 'results.xlsx', model_prefix: Optional[str] = None):
    """Merge all *_acc.xlsx files into a single results.xlsx file.
    
    Args:
        folder_path: Path to folder containing *_acc.xlsx files
        output_file: Output filename (default: results.xlsx)
        model_prefix: Model name prefix to remove from filenames (e.g., "Qwen3-VL-8B-Instruct")
    """
    print(f"Scanning folder: {folder_path}")
    if model_prefix:
        print(f"Using model prefix: {model_prefix}")
    
    # Find all acc files
    acc_files = glob.glob(os.path.join(folder_path, "*_acc.xlsx"))
    
    if not acc_files:
        print(f"No *_acc.xlsx files found in {folder_path}")
        return
    
    print(f"Found {len(acc_files)} acc files")
    
    # Check if MMMU_DEV_VAL_acc.xlsx exists and needs to be split
    mmmu_dev_val_acc = None
    mmmu_dev_val_results = None
    for acc_file in acc_files:
        if 'MMMU_DEV_VAL' in os.path.basename(acc_file) and '_acc.xlsx' in acc_file:
            mmmu_dev_val_acc = acc_file
            # Find corresponding results file
            results_file = acc_file.replace('_acc.xlsx', '_results.xlsx')
            if os.path.exists(results_file):
                mmmu_dev_val_results = results_file
            break
    
    # If MMMU_DEV_VAL exists, try to split it
    if mmmu_dev_val_results and os.path.exists(mmmu_dev_val_results):
        print(f"\nFound MMMU_DEV_VAL files, attempting to split...")
        try:
            df_results = pd.read_excel(mmmu_dev_val_results)
            if 'split' in df_results.columns:
                split_mapping = {
                    'dev': 'MMMU_DEV',
                    'validation': 'MMMU_VAL'
                }
                
                for split_value, split_name in split_mapping.items():
                    df_split = df_results[df_results['split'] == split_value].copy()
                    
                    if len(df_split) == 0:
                        print(f"  Warning: No data found for split '{split_value}', skipping...")
                        continue
                    
                    # Calculate accuracy for this split
                    total_count = len(df_split)
                    if 'judge' in df_split.columns:
                        a_count = (df_split['judge'] == 'A').sum()
                    else:
                        print(f"  Warning: No 'judge' column found in results file, skipping split...")
                        continue
                    
                    accuracy = a_count / total_count if total_count > 0 else 0.0
                    
                    # Create accuracy file for this split
                    acc_data = {
                        'metric': ['total', 'A_count', 'accuracy'],
                        'value': [total_count, a_count, accuracy]
                    }
                    df_acc = pd.DataFrame(acc_data)
                    
                    # Generate acc file name
                    base_name = os.path.basename(mmmu_dev_val_results).replace('MMMU_DEV_VAL', split_name).replace('_results.xlsx', '')
                    acc_path = os.path.join(folder_path, f"{base_name}_acc.xlsx")
                    df_acc.to_excel(acc_path, index=False)
                    print(f"  Created {split_name} acc file: {acc_path}")
                    
                    # Also create results file
                    results_path = os.path.join(folder_path, f"{base_name}_results.xlsx")
                    df_split.to_excel(results_path, index=False)
                    print(f"  Created {split_name} results file: {results_path}")
                
                # Remove MMMU_DEV_VAL from acc_files list so it won't be processed again
                if mmmu_dev_val_acc in acc_files:
                    acc_files.remove(mmmu_dev_val_acc)
                    print(f"  Removed MMMU_DEV_VAL_acc.xlsx from processing list")
                    
        except Exception as e:
            print(f"  Error splitting MMMU_DEV_VAL: {e}")
            import traceback
            traceback.print_exc()
    
    # Extract test set names and read data
    results = []
    for acc_file in acc_files:
        test_set_name = extract_test_set_name(acc_file, model_prefix)
        if test_set_name is None:
            print(f"Warning: Could not extract test set name from {acc_file}")
            # Use filename as fallback
            basename = os.path.basename(acc_file).replace('_acc.xlsx', '')
            # Try to extract after last underscore
            parts = basename.split('_')
            if len(parts) > 1:
                test_set_name = '_'.join(parts[-2:])  # Take last two parts
            else:
                test_set_name = basename
        
        metrics = read_acc_file(acc_file)
        if metrics:
            results.append({
                'test_set': test_set_name,
                'total': metrics.get('total', 0),
                'A_count': metrics.get('A_count', 0),
                'accuracy': metrics.get('accuracy', 0.0),
                'file': os.path.basename(acc_file)
            })
    
    if not results:
        print("No valid results found")
        return
    
    # Sort by order
    results.sort(key=lambda x: find_test_set_in_order(x['test_set']))
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns
    df_results = df_results[['test_set', 'total', 'A_count', 'accuracy']]
    
    # Transpose: test sets as columns, metrics as rows
    # Set test_set as index first
    df_transposed = df_results.set_index('test_set').T
    # Reset index to make metrics a column
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'metric'}, inplace=True)
    
    # Save to Excel
    output_path = os.path.join(folder_path, output_file)
    df_transposed.to_excel(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"\nSummary (transposed format):")
    print(df_transposed.to_string(index=False))
    
    # Also print original format for reference
    print(f"\nOriginal format (for reference):")
    print(df_results.to_string(index=False))
    
    # Calculate overall statistics
    total_samples = df_results['total'].sum()
    total_correct = df_results['A_count'].sum()
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    print(f"\nOverall Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Total correct (A): {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Merge accuracy results from multiple acc files')
    parser.add_argument('folder', type=str, help='Path to folder containing *_acc.xlsx files')
    parser.add_argument(
        '--output',
        type=str,
        default='results.xlsx',
        help='Output filename (default: results.xlsx)'
    )
    parser.add_argument(
        '--model_prefix',
        type=str,
        default=None,
        help='Model name prefix to remove from filenames (e.g., "Qwen3-VL-8B-Instruct"). '
             'If not provided, will try to auto-detect.'
    )
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        return
    
    merge_acc_files(args.folder, args.output, args.model_prefix)


if __name__ == "__main__":
    main()

