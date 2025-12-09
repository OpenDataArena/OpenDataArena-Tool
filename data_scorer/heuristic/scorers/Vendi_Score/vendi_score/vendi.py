import numpy as np
import scipy
import scipy.linalg
from sklearn import preprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm


def weight_K(K, p=None, sqrt_p=None):
    # 如果 p 为 None，直接返回按行数缩放的 K 矩阵
    if p is None:
        return K / K.shape[0]

    # 如果 sqrt_p 没有预先传递，计算 sqrt(p) 并缓存
    if sqrt_p is None:
        sqrt_p = np.sqrt(p)

    # 计算外积矩阵
    return K * np.outer(sqrt_p, sqrt_p)


def normalize_K(K):
    # 获取对角线元素的平方根
    d = np.sqrt(np.diagonal(K))

    # 定义标准化计算函数
    def normalize_block(start_row, end_row):
        return K[start_row:end_row] / (d[start_row:end_row, None] * d)

    # 动态调整并行块数，根据 CPU 核心数
    num_chunks = os.cpu_count()
    chunk_size = K.shape[0] // num_chunks if K.shape[0] >= num_chunks else K.shape[0]

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = []
        for i in range(num_chunks):
            start_row = i * chunk_size
            # 最后一个块保证覆盖剩余行
            end_row = K.shape[0] if i == num_chunks - \
                1 else (i + 1) * chunk_size
            futures.append(executor.submit(
                normalize_block, start_row, end_row))

        # 合并计算结果
        results = [future.result() for future in futures]

    return np.vstack(results)


def _entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_ ** q).sum()) / (1 - q)


def entropy_q(p, q=1, num_workers=96):
    p = np.asarray(p).flatten()

    # 拆分 p 为多个块nproc
    chunk_size = len(p) // num_workers or 1
    chunks = [p[i:i + chunk_size] for i in range(0, len(p), chunk_size)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_entropy_q, chunk, q) for chunk in chunks]

        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Entropy calculation"):
            results.append(f.result())

    return sum(results)


def score_K(K, q=1, p=None, normalize=False):
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    # 判断矩阵类型
    if isinstance(K_, scipy.sparse.csr.csr_matrix):
        w, _ = scipy.sparse.linalg.eigsh(K_)
    else:
        w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))

##############################################
# 以下为核矩阵计算部分的优化
##############################################


def _compute_kernel_block(args):
    """
    计算从 start 行到 end 行 的核矩阵（仅计算上三角部分）
    返回 (start, block)，其中 block 的 shape 为 (end-start, n)，n为样本数
    """
    start, end, samples, k = args
    n = len(samples)
    block = np.zeros((end - start, n))
    for i_local, i in enumerate(range(start, end)):
        # 只计算上三角部分
        for j in range(i, n):
            block[i_local, j] = k(samples[i], samples[j])
    return start, block


def score(samples, k, q=1, p=None, normalize=False, block_size=None):
    """
    对样本 samples 计算核矩阵，并评估 score。

    参数：
      samples: 可迭代样本
      k: 两个样本间计算核值的函数
      q: 熵计算相关参数
      p: 可选的权重参数
      normalize: 是否标准化矩阵K
      block_size: 分块大小（如果为 None，则根据 CPU 核数自动划分块）

    返回：
      基于核矩阵的 score。
    """
    n = len(samples)
    K = np.zeros((n, n))

    # 采用进程池解决 CPU 密集型核计算，分块任务减少调度开销
    # cpu_count = os.cpu_count()
    cpu_count = 96
    if block_size is None:
        # 根据 CPU 核数划分块，每块大概包含几行
        block_size = max(1, n // cpu_count)

    # 构造任务列表，每个任务计算一个块
    tasks = []
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        tasks.append((start, end, samples, k))

    # 使用 ProcessPoolExecutor 并行计算
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [executor.submit(_compute_kernel_block, task)
                   for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing kernel matrix"):
            start, block = future.result()
            rows = block.shape[0]
            # 填充 K：利用计算出的上三角结果填充对称矩阵
            for i_local, i in enumerate(range(start, start + rows)):
                K[i, i:] = block[i_local, i:]
                K[i+1:, i] = block[i_local, i+1:]

    return score_K(K, p=p, q=q, normalize=normalize)
