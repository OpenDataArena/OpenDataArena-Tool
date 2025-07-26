import pathlib
import pandas as pd
import argparse # 导入 argparse 库

# --- 1. 配置 (常量部分) ---

# 定义常量以避免拼写错误
DATASET_COL = 'dataset'
METRIC_COL = 'metric'

# --- 2. 辅助函数 (保持不变) ---

def get_value_from_df(df, value_col, dataset_filter_value, metric_filter_value=None, 
                      dataset_is_prefix=False, average=False, default_value="N/A"):
    """
    根据条件从 DataFrame 中提取单个值或平均值。
    """
    if value_col not in df.columns:
        return f"列缺失"

    filtered_df = df.copy()

    if dataset_is_prefix:
        if pd.isna(dataset_filter_value) or dataset_filter_value == "":
             return default_value
        filtered_df = filtered_df[filtered_df[DATASET_COL].astype(str).str.startswith(str(dataset_filter_value), na=False)]
    else:
        filtered_df = filtered_df[filtered_df[DATASET_COL] == dataset_filter_value]

    if metric_filter_value:
        filtered_df = filtered_df[filtered_df[METRIC_COL] == metric_filter_value]

    if filtered_df.empty:
        return default_value

    values = pd.to_numeric(filtered_df[value_col], errors='coerce')

    if average:
        if values.isna().all():
            return "error" 
        mean_val = values.mean()
        return mean_val if not pd.isna(mean_val) else "error"
    else:
        valid_values = values.dropna()
        if valid_values.empty:
            original_values = filtered_df[value_col].dropna()
            if not original_values.empty:
                return "error"
            return default_value
        
        return valid_values.iloc[0]

# --- 3. 主逻辑函数 (现在接收参数) ---

def process_and_summarize(source_dir: pathlib.Path, dest_dir: pathlib.Path, model_column: str):
    """
    主函数，执行查找、解析、提取和汇总操作。

    Args:
        source_dir (pathlib.Path): 包含原始结果的源目录。
        dest_dir (pathlib.Path): 用于存放最终 summary.csv 文件的目标目录。
        model_column (str): CSV 文件中包含模型分数的列名。
    """
    # 步骤 A: 检查和设置目录
    if not source_dir.is_dir():
        print(f"❌ 错误：源目录 '{source_dir}' 不存在，请检查路径。")
        return

    dest_dir.mkdir(exist_ok=True)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {dest_dir}")
    print(f"目标模型列: {model_column}")
    print("---")

    # 步骤 B: 递归查找所有符合条件的 csv 文件
    csv_files_found = list(source_dir.glob("**/summary/summary_*.csv"))

    if not csv_files_found:
        print("🟡 未在源目录中找到任何 'summary/summary_*.csv' 文件。")
        return

    # 步骤 C: 定义数据提取规则
    processing_rules = [
        ('drop', 'drop', 'accuracy', False, False),
        ('IFEval', 'IFEval', None, False, True),
        ('agieval', 'agieval', 'naive_average', False, False),
        ('mmlu_pro', 'mmlu_pro','accuracy', True, True),
        ('gsm8k', 'gsm8k', 'accuracy', False, False),
        ('math', 'math', 'accuracy', False, False),
        ('math_prm800k_500', 'math_prm800k_500', 'accuracy', False, False),
        ('OmniMath', 'OmniMath', None, False, False),
        ('OlympiadBench_OE_TO_maths_en_COMP', 'OlympiadBench_OE_TO_maths_en_COMP', 'accuracy', False, False)
        ('aime2024_accuracy', 'aime2024', 'accuracy', True, True),
        ('openai_humaneval', 'openai_humaneval', None, False, False),
        ('sanitized_mbpp_score', 'sanitized_mbpp', 'score', False, False),
        ('lcb_code_generation', 'lcb_code_generation', None, False, False),
        ('humaneval_plus', 'humaneval_plus', None, False, False),
        ('ARC-c', 'ARC-c', 'accuracy', False, False),
        ('bbh_naive_average', 'bbh', 'naive_average', False, False),
        ('GPQA_diamond', 'GPQA_diamond', 'accuracy', False, False),
        ('calm_Accuracy', 'calm', 'Accuracy', True, True),
        ('korbench', 'korbench', 'accuracy', True, True),
    ]
    
    all_results = []
    processed_count = 0

    # 步骤 D: 遍历文件，生成新文件名并提取数据
    for csv_path in csv_files_found:
        print(f"🔎 正在处理文件: {csv_path}")
        
        try:
            summary_dir = csv_path.parent
            ts_dir = summary_dir.parent
            top_dir = ts_dir.parent
            top_dir_name = top_dir.name
            ts_dir_name = ts_dir.name
            
            parts = top_dir_name.rsplit('_', 1)
            main_part, suffix = parts[0], parts[1]
            prefix = main_part.rsplit('_', 1)[0]
            
            new_filename = f"{prefix}-{ts_dir_name}-{suffix}.csv"
            
        except IndexError:
            print(f"  ⚠️ 跳过文件，其父目录命名格式不符合预期: {top_dir.name}")
            print("---")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            
            if model_column not in df.columns:
                print(f"  ⚠️ 跳过文件，因缺少目标列 '{model_column}'。")
                print("---")
                continue
            
            if DATASET_COL not in df.columns or METRIC_COL not in df.columns:
                print(f"  ⚠️ 跳过文件，因缺少 '{DATASET_COL}' 或 '{METRIC_COL}' 列。")
                print("---")
                continue

            file_summary = {'filename': new_filename}

            for stat_key, dataset_val, metric_val, is_prefix, needs_avg in processing_rules:
                value = get_value_from_df(
                    df,
                    value_col=model_column, # 使用传入的参数
                    dataset_filter_value=dataset_val,
                    metric_filter_value=metric_val,
                    dataset_is_prefix=is_prefix,
                    average=needs_avg,
                    default_value="N/A"
                )
                file_summary[stat_key] = value
            
            all_results.append(file_summary)
            processed_count += 1
            print(f"  -> 成功提取数据，标识为: {new_filename}")
            print("---")

        except pd.errors.EmptyDataError:
            print(f"  ⚠️ 跳过文件，因为文件为空: {csv_path}")
            print("---")
            continue
        except Exception as e:
            print(f"  ❌ 处理文件 {csv_path} 时发生错误: {e}")
            print("---")
            continue

    # 步骤 E: 创建并保存最终的 summary 文件
    if not all_results:
        print("🟡 未能从任何文件中成功提取数据。")
        return

    summary_df = pd.DataFrame(all_results)
    
    if 'filename' in summary_df.columns:
        summary_df.set_index('filename', inplace=True)
    
    output_columns = [rule[0] for rule in processing_rules]
    summary_df = summary_df.reindex(columns=output_columns) 

    # 使用传入的参数来构建输出文件名
    output_file_path = dest_dir / f"summary_{model_column}.csv"
    try:
        summary_df.to_csv(output_file_path)
        print(f"\n🎉 脚本执行完毕！共处理了 {processed_count} 个文件。")
        print(f"最终汇总文件已保存至: {output_file_path}")
    except Exception as e:
        print(f"\n❌ 写入最终汇总文件时出错: {e}")


# --- 4. 命令行参数解析和主程序入口 ---
def main():
    """
    解析命令行参数并启动主处理函数。
    """
    parser = argparse.ArgumentParser(
        description="查找、处理并汇总模型评估的 CSV 结果文件。",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助文本格式
    )
    
    parser.add_argument(
        "-s", "--source-dir",
        type=str,
        required=True,
        help="必须：包含原始结果的源目录路径。\n例如: outputs"
    )
    
    parser.add_argument(
        "-d", "--dest-dir",
        type=str,
        required=True,
        help="必须：用于存放最终 summary.csv 文件的目标目录路径。\n例如: res"
    )
    
    parser.add_argument(
        "-m", "--model-column",
        type=str,
        required=True,
        help="必须：在 CSV 文件中包含模型分数的列名。\n例如: qwen2.5-7b-instruct-vllm"
    )
    
    args = parser.parse_args()
    
    # 将字符串路径转换为 pathlib.Path 对象
    source_path = pathlib.Path(args.source_dir)
    dest_path = pathlib.Path(args.dest_dir)
    
    # 使用解析到的参数调用主函数
    process_and_summarize(source_path, dest_path, args.model_column)


if __name__ == "__main__":
    main()

