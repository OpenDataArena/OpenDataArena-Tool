#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量将目录下所有后缀为InfoVQA_TEST.xlsx的文件（包括子目录）转换为符合InfoVQA提交要求的JSON格式
输入：目录路径
输出：JSON文件（保存在原始目录，扩展名为.json，且文件名前加a_）
"""

import json
import os
import sys
import openpyxl
from pathlib import Path
import glob

def convert_excel_to_json(excel_path):
    """
    将Excel文件转换为JSON格式

    Args:
        excel_path: Excel文件路径

    Returns:
        JSON文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"文件不存在: {excel_path}")

    print(f"正在读取文件: {excel_path}")

    # 读取Excel文件
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb.active

    # 读取所有行
    rows = list(ws.iter_rows(values_only=True))

    if len(rows) == 0:
        raise ValueError("Excel文件为空")

    # 获取表头
    headers = rows[0]
    print(f"表头: {headers}")

    # 查找需要的列索引
    try:
        index_col = headers.index('index')
        prediction_col = headers.index('prediction')
    except ValueError as e:
        raise ValueError(f"未找到必需的列。当前列名: {headers}")

    # 构建JSON数据
    json_data = []
    for row in rows[1:]:  # 跳过表头
        if row[index_col] is not None:  # 确保questionId不为空
            item = {
                "questionId": int(row[index_col]),  # 确保questionId是整数
                "answer": str(row[prediction_col]) if row[prediction_col] is not None else ""
            }
            json_data.append(item)

    print(f"共处理 {len(json_data)} 条数据")

    # 生成输出文件路径，文件名最前面加a_
    excel_file = Path(excel_path)
    parent = excel_file.parent
    orig_name = excel_file.name
    json_name = f"a_{orig_name.rsplit('.', 1)[0]}.json"
    json_path = parent / json_name

    # 保存JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"JSON文件已保存到: {json_path}")
    return json_path

def main():
    if len(sys.argv) != 2:
        print("使用方法: python convert_to_json.py <包含Excel的目录路径>")
        print("示例: python convert_to_json.py ./data/results/")
        sys.exit(1)

    dir_path = sys.argv[1]
    if not os.path.isdir(dir_path):
        print(f"错误：{dir_path} 不是有效的目录")
        sys.exit(1)

    # 查找所有后缀为InfoVQA_TEST.xlsx的文件（包括子目录）
    excel_files = []
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            if fname.endswith('InfoVQA_TEST.xlsx'):
                excel_files.append(os.path.join(root, fname))

    if not excel_files:
        print(f"目录 {dir_path} 下未找到任何后缀为 InfoVQA_TEST.xlsx 的文件。")
        sys.exit(1)

    print(f"在目录 {dir_path} (及其子目录) 下共找到 {len(excel_files)} 个 InfoVQA_TEST.xlsx 文件。\n")

    success_count = 0
    for i, excel_path in enumerate(excel_files, 1):
        try:
            print(f"========== 正在转换({i}/{len(excel_files)})：{excel_path} ==========")
            json_path = convert_excel_to_json(excel_path)
            print(f"✓ 转换成功！JSON文件位置: {json_path}\n")
            success_count += 1
        except Exception as e:
            print(f"\n✗ 转换失败: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\n")

    print(f"全部处理完成：成功 {success_count}/{len(excel_files)} 个文件。")

if __name__ == "__main__":
    main()
