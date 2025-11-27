from .base_scorer import BaseScorer
from .utils import get_total_lines
from typing import Dict
import numpy as np
import json


class PartitionEntropyScorer(BaseScorer):
    def _validate_config(self):
        """验证配置参数"""
        # 验证全局簇数
        if "num_clusters" not in self.config:
            raise ValueError("num_clusters (number of global clusters) is required in config.")
        
        if not isinstance(self.config["num_clusters"], int) or self.config["num_clusters"] <= 0:
            raise ValueError(f"num_clusters must be a positive integer, got: {self.config['num_clusters']}")
        
        print(f"Global number of clusters: {self.config['num_clusters']}")

    def _setup(self):
        """初始化scorer"""
        # 获取全局聚类数量
        self.num_clusters = self.config["num_clusters"]
        print(f"Number of global clusters: {self.num_clusters}")
        
        print("Setting up PartitionEntropyScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """PartitionEntropyScorer 对整个数据集打分，不对单个样本打分"""
        raise NotImplementedError(
            "PartitionEntropyScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """评估整个数据集，计算分区熵（Partition Entropy）
        
        该方法计算子集数据在全局聚类簇中的分布熵。
        
        分区熵定义为：H = -Σ(p_i * log(p_i))
        其中 p_i 是第 i 个簇在子集中样本的比例
        
        熵越高表示子集数据在全局簇中分布越均匀，多样性越高
        熵越低表示子集数据在全局簇中分布越集中，多样性越低
        
        Args:
            dataset: 数据集文件路径（jsonl格式），每个数据项必须包含 cluster_id 字段
        
        Returns:
            包含 Partition Entropy 分数的字典
        """
        num_lines = get_total_lines(dataset)
        print(f"Computing Partition Entropy for {num_lines} samples in subset...")
        print(f"Global number of clusters: {self.num_clusters}")
        
        # 读取数据集，统计每个簇的样本数量
        cluster_counts = {}
        total_samples = 0
        missing_cluster_id_count = 0
        
        with open(dataset, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # 检查是否有 cluster_id 字段
                    if "cluster_id" not in item:
                        missing_cluster_id_count += 1
                        continue
                    
                    cluster_id = int(item["cluster_id"])
                    
                    # 统计簇计数
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                    total_samples += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Failed to extract cluster_id from line {line_num}: {e}")
                    continue
        
        if missing_cluster_id_count > 0:
            print(f"Warning: {missing_cluster_id_count} samples missing cluster_id field")
        
        if total_samples == 0:
            print("Error: No valid samples found with cluster_id")
            return {
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "max_entropy": 0.0,
                "num_samples": 0,
                "num_clusters_global": self.num_clusters,
                "num_clusters_in_subset": 0,
                "cluster_counts": {},
                "cluster_probabilities": {}
            }
        
        print(f"Successfully loaded {total_samples} samples with cluster_id")
        print(f"Number of unique clusters in subset: {len(cluster_counts)}")
        
        # 计算每个簇的概率分布
        cluster_probabilities = {}
        for cluster_id, count in cluster_counts.items():
            cluster_probabilities[cluster_id] = count / total_samples
        
        # 计算分区熵
        # H = -Σ(p_i * log(p_i))
        # 注意：只对子集中实际出现的簇计算熵
        entropy = 0.0
        for cluster_id, prob in cluster_probabilities.items():
            if prob > 0:  # 避免 log(0)
                entropy -= prob * np.log(prob)
        
        # 计算归一化熵（除以全局最大可能熵 log(num_clusters)）
        # 使用全局簇数来计算最大熵，这样可以反映子集相对于全局的多样性
        max_entropy = np.log(self.num_clusters) if self.num_clusters > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        print(f"Partition Entropy: {entropy:.4f}")
        print(f"Normalized Entropy: {normalized_entropy:.4f}")
        print(f"Max possible entropy (based on global clusters): {max_entropy:.4f}")
        print(f"Cluster distribution in subset: {cluster_counts}")
        
        return {
            "entropy": float(entropy),  # 分区熵
            "normalized_entropy": float(normalized_entropy),  # 归一化分区熵（相对于全局簇数）
            "max_entropy": float(max_entropy),  # 最大可能分区熵（基于全局簇数）
            "num_samples": total_samples,  # 子集中的样本数量
            "num_clusters_global": self.num_clusters,  # 全局聚类簇数量
            "num_clusters_in_subset": len(cluster_counts),  # 子集中实际出现的簇数量
            "cluster_counts": cluster_counts,  # 聚类计数
            "cluster_probabilities": {int(k): float(v) for k, v in cluster_probabilities.items()}  # 聚类概率分布
        }

