import numpy as np


def kmeans(
    embeddings,
    k,
    seed,
    max_iters,
    tol=1e-4,
):
    rng = np.random.default_rng(seed)
    n_samples = embeddings.shape[0]
    if k > n_samples:
        raise ValueError(f"k ({k}) cannot be larger than samples ({n_samples})")
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = embeddings[indices].copy()

    for _ in range(max_iters):
        distances = np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                new_centroids[idx] = embeddings[mask].mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids


def kmeans_std(embeddings, labels, centroids):
    distances = np.linalg.norm(embeddings - centroids[labels], axis=1)
    return float(np.std(distances))
