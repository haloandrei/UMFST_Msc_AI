import os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_embeddings_3d(
    points,
    labels,
    out_path,
    title=None,
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=6, cmap="tab20")
    ax.set_title(title or "Embeddings 3D + K-means")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_feature_map(feature_map, out_path, title=None):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(feature_map, cmap="viridis")
    if title:
        plt.title(title)
    plt.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
