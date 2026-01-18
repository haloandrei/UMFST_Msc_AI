import argparse
import json
import os
import multiprocessing

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_ACTIVATION,
    DEFAULT_DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SEED,
)
from data import FruitDataset
from model import SimpleCNN
from pca import pca_reduce
from visualize import save_embeddings_3d
from utils import set_seed, get_device


def load_report(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        print(f"Report not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def report_value(report, key):
    if key in report:
        return report[key]
    best = report.get("best")
    if isinstance(best, dict) and key in best:
        return best[key]
    return None


def resolve_hparams(args, report):
    image_size = args.image_size
    if image_size is None:
        image_size = report_value(report, "image_size")
    if image_size is None:
        image_size = DEFAULT_IMAGE_SIZE

    embedding_dim = args.embedding_dim
    if embedding_dim is None:
        embedding_dim = report_value(report, "embedding_dim")
    if embedding_dim is None:
        embedding_dim = DEFAULT_EMBEDDING_DIM

    dropout = args.dropout
    if dropout is None:
        dropout = report_value(report, "dropout")
    if dropout is None:
        dropout = DEFAULT_DROPOUT

    activation = args.activation
    if activation is None:
        activation = report_value(report, "activation")
    if activation is None:
        activation = DEFAULT_ACTIVATION

    return {
        "image_size": int(image_size),
        "embedding_dim": int(embedding_dim),
        "dropout": float(dropout),
        "activation": str(activation),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Vizualizeaza clusterele reale dupa tipul de fruct")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Path catre Fruits-360 Training")
    parser.add_argument("--weights", default=None, help="Path catre model .pth (ex. outputs/baseline_model.pth)")
    parser.add_argument("--report", default=None, help="Optional JSON report pentru a prelua hiperparametrii")
    parser.add_argument("--output-path", default="outputs/true_clusters_3d.png")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", choices=["relu", "leaky_relu"], default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--encoding-index", type=int, default=0)
    parser.add_argument("--encoding-preview", type=int, default=12)
    return parser.parse_args()


def extract_embeddings(
    model,
    loader,
    device,
):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            _, embeds = model(images)
            all_embeddings.append(embeds.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def main():
    args = parse_args()
    if not os.path.isdir(args.data_root):
        print(f"Data root not found: {args.data_root}")
        print(f"Falling back to default: {DEFAULT_DATA_ROOT}")
        args.data_root = DEFAULT_DATA_ROOT

    if args.num_workers > 0:
        try:
            _ = multiprocessing.get_context().Lock()
        except Exception as exc:
            print(f"Disabling workers due to multiprocessing issue: {exc}")
            args.num_workers = 0

    report = load_report(args.report)
    weights_path = args.weights or report_value(report, "weights") or "outputs/baseline_model.pth"
    if not os.path.isfile(weights_path):
        print(f"Weights not found: {weights_path}")
        return

    hparams = resolve_hparams(args, report)
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Using weights: {weights_path}")
    print(
        "Hyperparams: "
        f"image_size={hparams['image_size']} "
        f"embedding_dim={hparams['embedding_dim']} "
        f"dropout={hparams['dropout']} "
        f"activation={hparams['activation']}"
    )

    dataset = FruitDataset(
        root=args.data_root,
        image_size=hparams["image_size"],
        max_samples=args.max_samples,
        seed=args.seed,
    )
    pin_memory = device.type != "cpu"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = SimpleCNN(
        num_classes=len(dataset.class_names),
        embedding_dim=hparams["embedding_dim"],
        dropout=hparams["dropout"],
        activation=hparams["activation"],
        image_size=hparams["image_size"],
    ).to(device)

    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        print(f"Failed to load weights: {exc}")
        return

    embeddings = extract_embeddings(model, loader, device)
    labels = np.array([label for _, label in dataset.samples], dtype=np.int64)

    if embeddings.shape[0] != labels.shape[0]:
        print("Mismatch between embeddings and labels; check data loader settings.")
        return

    index = args.encoding_index
    if 0 <= index < len(dataset.samples):
        path, label = dataset.samples[index]
        class_name = dataset.class_names[label] if dataset.class_names else str(label)
        preview_len = max(1, args.encoding_preview)
        preview_vals = ", ".join(f"{v:.4f}" for v in embeddings[index][:preview_len])
        if preview_len < embeddings.shape[1]:
            preview_vals += ", ..."
        print(f"Encoding sample idx={index} label={label} class={class_name}")
        print(f"Path: {path}")
        print(f"Embedding dim: {embeddings.shape[1]}")
        print(f"Embedding preview: [{preview_vals}]")
    else:
        print(f"Encoding index out of range: {index}")

    points_3d = pca_reduce(embeddings, 3)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_embeddings_3d(
        points_3d,
        labels,
        args.output_path,
        title="Embeddings 3D (true labels)",
    )
    print(f"Saved true clusters plot to {args.output_path}")


if __name__ == "__main__":
    main()
