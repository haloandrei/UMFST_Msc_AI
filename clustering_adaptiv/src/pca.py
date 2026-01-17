import numpy as np


def pca_reduce(embeddings, n_components):
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return centered @ components.T
