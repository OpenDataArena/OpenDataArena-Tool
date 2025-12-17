import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm
import argparse

# --- 1. Perform Mini-Batch K-Means Clustering ---


def perform_mini_batch_kmeans(data_mmap, n_clusters, max_epochs=10, batch_size=4096, n_init=3, convergence_threshold=1e-4):
    """
    Perform batch-wise K-Means training on the entire dataset, with support for multiple iterations and early stopping.

    Args:
        data_mmap: Memory-mapped numpy array.
        n_clusters (int): Number of clusters.
        max_epochs (int): Maximum number of passes through the dataset (Epochs).
        convergence_threshold (float): Threshold for determining cluster center convergence.
    """
    print(f"\n--- Step 2: Starting Mini-Batch K-Means on full dataset (K={n_clusters}) ---")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=42,
        n_init=n_init,  # The algorithm will run independently 3 times and return the best result
        reassignment_ratio=0.01  # Helps handle empty cluster issues
    )

    total_size = data_mmap.shape[0]
    prev_centroids = None

    # *** Main modification: Added outer Epoch loop ***
    for epoch in range(max_epochs):
        epoch_start_time = time.time()

        # Inner loop: iterate through all batches once
        with tqdm(total=total_size, desc=f"Epoch {epoch + 1}/{max_epochs}") as pbar:
            for i in range(0, total_size, batch_size):
                end = min(i + batch_size, total_size)
                batch_data = data_mmap[i:end]
                kmeans.partial_fit(batch_data)
                pbar.update(len(batch_data))

        # *** Check convergence ***
        # Calculate sum of squared differences between old and new centroids
        current_centroids = kmeans.cluster_centers_
        if prev_centroids is not None:
            centroid_shift = np.sum((current_centroids - prev_centroids) ** 2)
            print(
                f"Epoch {epoch + 1} completed, time elapsed: {time.time() - epoch_start_time:.2f} seconds. Centroid shift: {centroid_shift:.6f}")
            if centroid_shift < convergence_threshold:
                print(f"Cluster centers have converged. Early stopping at Epoch {epoch + 1}.")
                break
        else:
            print(
                f"Epoch {epoch + 1} completed, time elapsed: {time.time() - epoch_start_time:.2f} seconds.")

        prev_centroids = np.copy(current_centroids)

        if epoch == max_epochs - 1:
            print("Maximum number of epochs reached. Training completed.")

    print("\nModel training completed!")
    return kmeans


# --- 2. Assign Labels to All Data and Save Results ---
def predict_and_save(model, data_mmap, output_labels_file, output_centroids_file, batch_size):
    """Predict labels for all data in batches and save the results."""
    print("\n--- Step 3: Starting cluster label assignment for all data points ---")

    total_size = data_mmap.shape[0]
    # Create an empty numpy array to store all labels
    all_labels = np.zeros(total_size, dtype=np.int32)

    with tqdm(total=total_size, desc="Predicting in batches") as pbar:
        for i in range(0, total_size, batch_size):
            end = min(i + batch_size, total_size)
            batch_data = data_mmap[i:end]
            labels = model.predict(batch_data)
            all_labels[i:end] = labels
            pbar.update(len(batch_data))

    print("Label prediction for all data completed.")

    # Save labels and centroids to files
    np.save(output_labels_file, all_labels)
    np.save(output_centroids_file, model.cluster_centers_)
    print(f"All cluster labels saved to: '{output_labels_file}'")
    print(f"Cluster centroids saved to: '{output_centroids_file}'")

    return all_labels, model.cluster_centers_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16384, type=int)
    parser.add_argument("--opt_k", default=2000, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)

    parser.add_argument(
        "--input_path", default="/mnt/petrelfs/gaoxin/fine_data_sel/minibatch-kmeans/gaussian_data_simple.npy")
    parser.add_argument("--output_dir", default="./")
    return parser.parse_args()


# --- Main Process ---
if __name__ == '__main__':
    # Record overall start time
    args = parse_args()
    overall_start_time = time.time()
    print(f"Opening file '{args.input_path}' in read-only memory-mapped mode...")
    embeddings_mmap = np.load(args.input_path, mmap_mode='r')

    # Step 2: Perform clustering training (using modified function)
    kmeans_model = perform_mini_batch_kmeans(
        embeddings_mmap,
        n_clusters=args.opt_k,
        max_epochs=args.max_epochs,      # <-- Adjustable: up to 10 training epochs
        batch_size=args.batch_size
    )

    # Step 3: Predict and save
    labels, centroids = predict_and_save(
        kmeans_model,
        embeddings_mmap,
        f'{args.output_dir}/cluster_labels.npy',
        f'{args.output_dir}/cluster_centroids.npy',
        batch_size=4096
    )

    # --- Cluster Results Analysis ---
    print("\n--- Cluster Results Analysis ---")
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Total number of clusters formed: {len(unique_labels)}.")
    print("Sample count distribution per cluster:")
    # To avoid screen overflow, only display info for first 10 and last 10 clusters
    if len(unique_labels) > 20:
        for i in range(10):
            print(f"  Cluster {unique_labels[i]:>3}: {counts[i]} samples")
        print("  ...")
        for i in range(len(unique_labels) - 10, len(unique_labels)):
            print(f"  Cluster {unique_labels[i]:>3}: {counts[i]} samples")
    else:
        for label, count in zip(unique_labels, counts):
            print(f"  Cluster {label}: {count} samples")

    overall_end_time = time.time()
    print(f"\nTotal time for entire process: {(overall_end_time - overall_start_time) / 60:.2f} minutes.")
