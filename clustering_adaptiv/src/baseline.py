import argparse
import json
import os
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_ACTIVATION,
    DEFAULT_DATA_ROOT,
    DEFAULT_VAL_RATIO,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_K,
    DEFAULT_KMEANS_ITERS,
)
from data import FruitDataset
from model import SimpleCNN
from clustering import kmeans, kmeans_std
from pca import pca_reduce
from visualize import save_embeddings_3d, save_feature_map
from utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    set_seed,
    get_device,
    split_train_val,
)


def denormalize_image(tensor):
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = arr * std + mean
    return np.clip(arr, 0.0, 1.0)


def evaluate_model(
    model,
    loader,
    device,
    criterion,
    collect_predictions=False,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            if collect_predictions:
                all_labels.append(labels.cpu())
                all_preds.append(preds.cpu())
    avg_loss = total_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    if collect_predictions and all_labels and all_preds:
        return avg_loss, accuracy, torch.cat(all_labels), torch.cat(all_preds)
    return avg_loss, accuracy, None, None


def save_epoch_sample(
    model,
    loader,
    device,
    class_names,
    out_dir,
    epoch,
):
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            image = denormalize_image(images[0])
            true_idx = int(labels[0].item())
            pred_idx = int(preds[0].item())
            true_name = class_names[true_idx] if class_names else str(true_idx)
            pred_name = class_names[pred_idx] if class_names else str(pred_idx)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"Epoch {epoch}: true={true_name}, pred={pred_name}")
            fig.tight_layout()
            out_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            break


def save_loss_curves(
    train_losses,
    val_losses,
    val_accs,
    out_dir,
):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, label="train loss")
    if val_losses:
        ax.plot(epochs, val_losses, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    if val_accs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, val_accs, label="val acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "val_accuracy.png"), dpi=150)
        plt.close(fig)


def compute_confusion_matrix(
    labels,
    preds,
    num_classes,
):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(labels, preds):
        matrix[true_idx, pred_idx] += 1
    return matrix


def save_confusion_matrix(
    matrix,
    class_names,
    out_path,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if len(class_names) <= 30:
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_model(
    model,
    loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    class_names,
    output_dir,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    val_losses = []
    val_accs = []
    sample_dir = os.path.join(output_dir, "epoch_samples")
    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_loss = running / max(1, len(loader))
        train_losses.append(avg_loss)

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch} train loss: {avg_loss:.4f} | "
            f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f}"
        )
        save_epoch_sample(model, val_loader, device, class_names, sample_dir, epoch)

    return train_losses, val_losses, val_accs


def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            _, embeds = model(images)
            all_embeddings.append(embeds.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def get_feature_map(model, sample, device):
    model.eval()
    with torch.no_grad():
        features = model.conv1_features(sample.unsqueeze(0).to(device))
    features = features[0]
    channel_means = features.mean(dim=(1, 2))
    best_idx = int(torch.argmax(channel_means))
    fmap = features[best_idx].cpu().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    return fmap


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline CNN + K-means + viz")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Path catre Fruits-360 Training",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--activation", choices=["relu", "leaky_relu"], default=DEFAULT_ACTIVATION)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--kmeans-iters", type=int, default=DEFAULT_KMEANS_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


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

    set_seed(args.seed)
    device = get_device()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = FruitDataset(
        root=args.data_root,
        image_size=args.image_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    train_idx, val_idx = split_train_val(len(dataset), args.val_ratio, args.seed)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    pin_memory = device.type != "cpu"
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = SimpleCNN(
        num_classes=len(dataset.class_names),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        activation=args.activation,
        image_size=args.image_size,
    ).to(device)

    print(f"Samples: {len(dataset)}")
    print(f"Classes: {len(dataset.class_names)}")
    print(f"Using device: {device}")

    train_losses, val_losses, val_accs = train_model(
        model,
        loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        dataset.class_names,
        args.output_dir,
    )
    save_loss_curves(train_losses, val_losses, val_accs, args.output_dir)

    embed_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    embeddings = extract_embeddings(model, embed_loader, device)

    labels, centroids = kmeans(embeddings, args.k, args.seed, args.kmeans_iters)
    std_value = kmeans_std(embeddings, labels, centroids)
    points_3d = pca_reduce(embeddings, 3)

    save_embeddings_3d(points_3d, labels, os.path.join(args.output_dir, "embeddings_3d.png"))

    sample_img, _ = dataset[0]
    fmap = get_feature_map(model, sample_img, device)
    save_feature_map(fmap, os.path.join(args.output_dir, "conv1_feature_map.png"))

    criterion = nn.CrossEntropyLoss()
    _, _, val_labels, val_preds = evaluate_model(
        model,
        val_loader,
        device,
        criterion,
        collect_predictions=True,
    )
    if val_labels is not None and val_preds is not None and val_labels.numel() > 0:
        matrix = compute_confusion_matrix(
            val_labels.numpy(),
            val_preds.numpy(),
            len(dataset.class_names),
        )
        save_confusion_matrix(
            matrix,
            dataset.class_names,
            os.path.join(args.output_dir, "confusion_matrix.png"),
        )

    weights_path = os.path.join(args.output_dir, "baseline_model.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved baseline weights to {weights_path}")

    report = {
        "k": args.k,
        "kmeans_std": std_value,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "activation": args.activation,
        "epochs": args.epochs,
        "val_ratio": args.val_ratio,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "final_val_acc": val_accs[-1] if val_accs else None,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "weights": weights_path,
    }
    with open(os.path.join(args.output_dir, "baseline_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Baseline done")
    print(report)


if __name__ == "__main__":
    main()
