import numpy as np
import scipy
import scipy.linalg
from sklearn import preprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def weight_K(K, p=None, sqrt_p=None):
    # 如果p为None，直接返回按行数缩放的K矩阵
    if p is None:
        return K / K.shape[0]

    # 如果sqrt_p没有预先传递，计算sqrt(p)并缓存
    if sqrt_p is None:
        sqrt_p = np.sqrt(p)

    # 计算外积矩阵
    return K * np.outer(sqrt_p, sqrt_p)


def normalize_K(K):
    # 获取对角线元素的平方根
    d = np.sqrt(np.diagonal(K))

    # 定义标准化计算函数
    def normalize_block(start_row, end_row):
        # 对每行进行标准化
        return K[start_row:end_row] / (d[start_row:end_row, None] * d)

    # 动态调整并行块数，根据CPU核心数
    num_chunks = os.cpu_count()  # 根据系统的CPU核心数设置
    chunk_size = K.shape[0] // num_chunks

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = []
        for i in range(num_chunks):
            start_row = i * chunk_size
            end_row = (i + 1) * chunk_size if i != num_chunks - \
                1 else K.shape[0]
            futures.append(executor.submit(
                normalize_block, start_row, end_row))

        # 使用 `as_completed` 来减少等待时间
        results = [future.result() for future in futures]

    # 合并并返回标准化后的矩阵
    return np.vstack(results)


def _entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_ ** q).sum()) / (1 - q)


# def entropy_q(p, q=1, num_workers=8):
#     p = np.asarray(p).flatten()

#     # Split the array into chunks for parallel processing
#     chunk_size = len(p) // num_workers
#     chunks = [p[i:i + chunk_size] for i in range(0, len(p), chunk_size)]

#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         results = list(executor.map(_entropy_q, chunks, [q]*len(chunks)))

#     # Combine results from all chunks
#     return sum(results)

def entropy_q(p, q=1, num_workers=8):
    p = np.asarray(p).flatten()

    # Split the array into chunks
    chunk_size = len(p) // num_workers or 1
    chunks = [p[i:i + chunk_size] for i in range(0, len(p), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_entropy_q, chunk, q) for chunk in chunks]

        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Entropy calculation"):
            results.append(f.result())

    return sum(results)


def score_K(K, q=1, p=None, normalize=False):
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    if type(K_) == scipy.sparse.csr.csr_matrix:
        w, _ = scipy.sparse.linalg.eigsh(K_)
    else:
        w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))


def _compute_kernel_row(i, samples, k):
    n = len(samples)
    row = np.zeros(n)
    for j in range(i, n):
        val = k(samples[i], samples[j])
        row[j] = val
    return i, row  # 返回行号和计算结果


def score(samples, k, q=1, p=None, normalize=False):
    n = len(samples)
    K = np.zeros((n, n))

    # 控制线程数，使用 CPU 核心数
    max_workers = os.cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(n):
            futures.append(executor.submit(_compute_kernel_row, i, samples, k))

        # 用 tqdm 跟踪完成的任务
        for future in tqdm(as_completed(futures), total=n, desc="Computing kernel matrix"):
            i, row = future.result()
            K[i, i:] = row[i:]
            K[i+1:, i] = row[i+1:]

    return score_K(K, p=p, q=q, normalize=normalize)

# def score(samples, k, q=1, p=None, normalize=False):
#     n = len(samples)
#     K = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i, n):
#             K[i, j] = K[j, i] = k(samples[i], samples[j])
#     return score_K(K, p=p, q=q, normalize=normalize)
