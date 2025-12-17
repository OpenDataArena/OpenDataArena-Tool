import numpy as np


def cosine_similarity_safe(a, b):
    """Cosine similarity (safe version, handles zero vectors)"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def euclidean_similarity(a, b):
    """Similarity based on Euclidean distance (smaller distance means higher similarity)"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    distance = np.linalg.norm(a - b)
    # Convert distance to similarity: 1 / (1 + distance)
    return 1.0 / (1.0 + distance)


def manhattan_similarity(a, b):
    """Similarity based on Manhattan distance"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    distance = np.sum(np.abs(a - b))
    return 1.0 / (1.0 + distance)


def dot_product_similarity(a, b):
    """Dot product similarity"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    return np.dot(a, b)


def pearson_correlation(a, b):
    """Pearson correlation coefficient"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    numerator = np.sum((a - a_mean) * (b - b_mean))
    denominator = np.sqrt(np.sum((a - a_mean)**2) * np.sum((b - b_mean)**2))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def rbf_kernel(x, y, gamma=1.0):
    """RBF (Radial Basis Function) kernel / Gaussian kernel
    
    K(x, y) = exp(-gamma * ||x - y||^2)
    
    This is a standard positive definite kernel function that guarantees the kernel matrix is positive semi-definite.
    
    Args:
        x, y: Input vectors
        gamma: Kernel parameter controlling the width of the kernel. Larger gamma means narrower kernel
               Common setting: gamma = 1 / (2 * sigma^2)
        
    Returns:
        Kernel value, range (0, 1], equals 1 when x=y
    """
    x = np.asarray(x)
    y = np.asarray(y)
    squared_distance = np.sum((x - y) ** 2)
    return np.exp(-gamma * squared_distance)


def polynomial_kernel(x, y, degree=3, coef0=1.0):
    """Polynomial kernel function
    
    K(x, y) = (x·y + coef0)^degree
    
    Args:
        x, y: Input vectors
        degree: Polynomial degree
        coef0: Constant term (needs to be > 0 to ensure positive definiteness)
        
    Returns:
        Kernel value
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return (np.dot(x, y) + coef0) ** degree


def linear_kernel(x, y):
    """Linear kernel function
    
    K(x, y) = x·y
    
    Note: Linear kernel only guarantees positive definiteness under specific conditions (e.g., data in the same quadrant)
    
    Args:
        x, y: Input vectors
        
    Returns:
        Kernel value (dot product)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.dot(x, y)


def get_similarity_function(metric_name: str):
    """Get similarity calculation function by name
    
    Args:
        metric_name: Name of the similarity metric
        
    Returns:
        Similarity calculation function
        
    Available metrics:
        - cosine: Cosine similarity (default)
        - euclidean: Euclidean distance similarity
        - manhattan: Manhattan distance similarity
        - dot_product: Dot product similarity
        - pearson: Pearson correlation coefficient
    """
    similarity_functions = {
        "cosine": cosine_similarity_safe,
        "euclidean": euclidean_similarity,
        "manhattan": manhattan_similarity,
        "dot_product": dot_product_similarity,
        "pearson": pearson_correlation,
    }
    
    if metric_name not in similarity_functions:
        available = ", ".join(similarity_functions.keys())
        raise ValueError(
            f"Unknown similarity metric '{metric_name}'. "
            f"Available metrics: {available}"
        )
    
    return similarity_functions[metric_name]


def get_kernel_function(kernel_name: str, **kernel_params):
    """Get kernel function by name
    
    Args:
        kernel_name: Name of the kernel function
        kernel_params: Kernel function parameters (e.g., gamma, degree, coef0)
        
    Returns:
        Kernel function
        
    Available kernels:
        - rbf: RBF kernel/Gaussian kernel (recommended for Log-Det)
        - polynomial: Polynomial kernel
        - linear: Linear kernel
    """
    if kernel_name == "rbf":
        gamma = kernel_params.get("gamma", 1.0)
        return lambda x, y: rbf_kernel(x, y, gamma=gamma)
    elif kernel_name == "polynomial":
        degree = kernel_params.get("degree", 3)
        coef0 = kernel_params.get("coef0", 1.0)
        return lambda x, y: polynomial_kernel(x, y, degree=degree, coef0=coef0)
    elif kernel_name == "linear":
        return linear_kernel
    else:
        available = "rbf, polynomial, linear"
        raise ValueError(
            f"Unknown kernel function '{kernel_name}'. "
            f"Available kernels: {available}"
        )


def euclidean_distance(a, b):
    """Euclidean distance"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    return np.linalg.norm(a - b)


def squared_euclidean_distance(a, b):
    """Squared Euclidean distance"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    return np.sum((a - b) ** 2)


def manhattan_distance(a, b):
    """Manhattan distance (L1 distance)"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    return np.sum(np.abs(a - b))


def cosine_distance(a, b):
    """Cosine distance (1 - cosine similarity)"""
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0  # Completely dissimilar
    
    cosine_sim = dot_product / (norm_a * norm_b)
    return 1.0 - cosine_sim


def get_distance_function(metric_name: str):
    """Get distance calculation function by name
    
    Args:
        metric_name: Name of the distance metric
        
    Returns:
        Distance calculation function
        
    Available metrics:
        - euclidean: Euclidean distance
        - squared_euclidean: Squared Euclidean distance
        - manhattan: Manhattan distance (L1 distance)
        - cosine: Cosine distance (1 - cosine similarity)
    """
    distance_functions = {
        "euclidean": euclidean_distance,
        "squared_euclidean": squared_euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_distance,
    }
    
    if metric_name not in distance_functions:
        available = ", ".join(distance_functions.keys())
        raise ValueError(
            f"Unknown distance metric '{metric_name}'. "
            f"Available metrics: {available}"
        )
    
    return distance_functions[metric_name]


def get_total_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)
