import argparse
import json
import os
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_ACTIVATION,
    DEFAULT_DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_K,
    DEFAULT_KMEANS_ITERS,
    DEFAULT_RL_EPISODES,
    DEFAULT_RL_STEPS,
    DEFAULT_TUNE_EPOCHS,
)
from data import FruitDataset, build_loader
from model import SimpleCNN
from clustering import kmeans, kmeans_std
from pca import pca_reduce
from visualize import save_embeddings_3d
from rl_agent import QLearningAgent
from rl_env import HyperparamEnv, actions
from utils import set_seed, get_device


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def train_model(
    model,
    loader,
    device,
    epochs,
    lr,
    weight_decay,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for images, labels in tqdm(
            loader,
            desc=f"Tune epoch {epoch + 1}/{epochs}",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_loss = running / max(1, len(loader))
        print(f"Tune epoch {epoch + 1} loss: {avg_loss:.4f}")


def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            _, embeds = model(images)
            all_embeddings.append(embeds.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def compute_kmeans_std(
    model,
    loader,
    device,
    k,
    kmeans_iters,
    seed,
):
    embeddings = extract_embeddings(model, loader, device)
    labels, centroids = kmeans(embeddings, k, seed, kmeans_iters)
    return kmeans_std(embeddings, labels, centroids)


def save_cluster_plot(
    model,
    loader,
    device,
    k,
    kmeans_iters,
    seed,
    out_path,
):
    embeddings = extract_embeddings(model, loader, device)
    labels, _ = kmeans(embeddings, k, seed, kmeans_iters)
    points_3d = pca_reduce(embeddings, 3)
    save_embeddings_3d(points_3d, labels, out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="RL pentru tuning hiperparametri CNN")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Path catre Fruits-360 Training")
    parser.add_argument("--baseline-weights", default="outputs/baseline_model.pth")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--activation", choices=["relu", "leaky_relu"], default=DEFAULT_ACTIVATION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--kmeans-iters", type=int, default=DEFAULT_KMEANS_ITERS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_RL_EPISODES)
    parser.add_argument("--steps", type=int, default=DEFAULT_RL_STEPS)
    parser.add_argument("--tune-epochs", type=int, default=DEFAULT_TUNE_EPOCHS)
    parser.add_argument("--lr-values", default="0.0003,0.001,0.003")
    parser.add_argument("--dropout-values", default="0.0,0.1,0.3,0.5")
    parser.add_argument("--k-values", default=str(DEFAULT_K))
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.2)
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

    if not os.path.isfile(args.baseline_weights):
        print(f"Baseline weights not found: {args.baseline_weights}")

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = FruitDataset(
        root=args.data_root,
        image_size=args.image_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    loader = build_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=True,
    )
    eval_loader = build_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=False,
    )

    lr_values = parse_float_list(args.lr_values)
    dropout_values = parse_float_list(args.dropout_values)
    k_values = parse_int_list(args.k_values)
    k_values = [k for k in k_values if k <= len(dataset)]
    if not k_values:
        raise ValueError("k-values must include at least one value <= number of samples")

    model = SimpleCNN(
        num_classes=len(dataset.class_names),
        embedding_dim=args.embedding_dim,
        dropout=DEFAULT_DROPOUT,
        activation=args.activation,
        image_size=args.image_size,
    ).to(device)
    model.load_state_dict(torch.load(args.baseline_weights, map_location=device))

    env = HyperparamEnv(lr_values, dropout_values, k_values)
    baseline_k = env.current_values()[2]
    baseline_std = compute_kmeans_std(
        model,
        eval_loader,
        device,
        baseline_k,
        args.kmeans_iters,
        args.seed,
    )
    print(f"Baseline kmeans std: {baseline_std:.6f}")

    agent = QLearningAgent(
        state_size=env.state_size(),
        action_size=env.action_size(),
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    best_std = baseline_std
    best_state = {
        "lr": env.current_values()[0],
        "dropout": env.current_values()[1],
        "k": env.current_values()[2],
        "std": best_std,
    }
    best_weights_path = os.path.join(args.output_dir, "rl_best_model.pth")

    step_history = []
    episode_summaries = []

    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}")
        state = env.reset()
        prev_std = baseline_std
        episode_best_std = prev_std
        episode_best_state = {
            "lr": env.current_values()[0],
            "dropout": env.current_values()[1],
            "k": env.current_values()[2],
            "std": prev_std,
        }
        episode_best_weights_path = os.path.join(
            args.output_dir,
            f"rl_episode_{ep + 1:03d}_best.pth",
        )
        episode_improvements = []

        for step_idx in tqdm(range(args.steps), desc="Steps"):
            prev_state = state
            action = agent.select_action(prev_state)
            state = env.step(action)
            lr_value, dropout_value, k_value = env.current_values()

            tuned_model = SimpleCNN(
                num_classes=len(dataset.class_names),
                embedding_dim=args.embedding_dim,
                dropout=dropout_value,
                activation=args.activation,
                image_size=args.image_size,
            ).to(device)
            tuned_model.load_state_dict(torch.load(args.baseline_weights, map_location=device))

            train_model(
                tuned_model,
                loader,
                device,
                args.tune_epochs,
                lr_value,
                args.weight_decay,
            )

            std_value = compute_kmeans_std(
                tuned_model,
                eval_loader,
                device,
                k_value,
                args.kmeans_iters,
                args.seed,
            )

            reward = prev_std - std_value
            action_name = actions[action] if action < len(actions) else str(action)
            step_info = {
                "episode": ep + 1,
                "step": step_idx + 1,
                "action": action_name,
                "lr": lr_value,
                "dropout": dropout_value,
                "k": k_value,
                "prev_std": prev_std,
                "std": std_value,
                "reward": reward,
            }
            step_history.append(step_info)
            agent.update(prev_state, action, reward, state)
            prev_std = std_value

            if reward > 0:
                episode_improvements.append(step_info)

            if std_value < episode_best_std:
                episode_best_std = std_value
                episode_best_state = {
                    "lr": lr_value,
                    "dropout": dropout_value,
                    "k": k_value,
                    "std": std_value,
                }
                torch.save(tuned_model.state_dict(), episode_best_weights_path)

            if std_value < best_std:
                best_std = std_value
                best_state = {
                    "lr": lr_value,
                    "dropout": dropout_value,
                    "k": k_value,
                    "std": std_value,
                }
                torch.save(tuned_model.state_dict(), best_weights_path)
                print(f"New best std: {best_std:.6f} saved to {best_weights_path}")

        episode_plot_weights = (
            episode_best_weights_path
            if os.path.exists(episode_best_weights_path)
            else args.baseline_weights
        )
        episode_model = SimpleCNN(
            num_classes=len(dataset.class_names),
            embedding_dim=args.embedding_dim,
            dropout=episode_best_state["dropout"],
            activation=args.activation,
            image_size=args.image_size,
        ).to(device)
        episode_model.load_state_dict(torch.load(episode_plot_weights, map_location=device))
        episode_cluster_path = os.path.join(
            args.output_dir,
            f"rl_episode_{ep + 1:03d}_clusters_3d.png",
        )
        save_cluster_plot(
            episode_model,
            eval_loader,
            device,
            episode_best_state["k"],
            args.kmeans_iters,
            args.seed,
            episode_cluster_path,
        )

        final_lr, final_dropout, final_k = env.current_values()
        episode_summaries.append(
            {
                "episode": ep + 1,
                "final_state": {"lr": final_lr, "dropout": final_dropout, "k": final_k},
                "best_state": episode_best_state,
                "best_weights": episode_best_weights_path
                if os.path.exists(episode_best_weights_path)
                else None,
                "cluster_plot": episode_cluster_path,
                "improvements": episode_improvements,
            }
        )

    history_path = os.path.join(args.output_dir, "rl_step_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(step_history, f, indent=2)

    episode_path = os.path.join(args.output_dir, "rl_episode_summaries.json")
    with open(episode_path, "w", encoding="utf-8") as f:
        json.dump(episode_summaries, f, indent=2)

    plot_weights_path = best_weights_path if os.path.exists(best_weights_path) else args.baseline_weights
    plot_dropout = best_state["dropout"] if os.path.exists(best_weights_path) else DEFAULT_DROPOUT
    plot_k = best_state["k"] if os.path.exists(best_weights_path) else baseline_k
    plot_model = SimpleCNN(
        num_classes=len(dataset.class_names),
        embedding_dim=args.embedding_dim,
        dropout=plot_dropout,
        activation=args.activation,
        image_size=args.image_size,
    ).to(device)
    plot_model.load_state_dict(torch.load(plot_weights_path, map_location=device))
    cluster_plot_path = os.path.join(args.output_dir, "rl_clusters_3d.png")
    save_cluster_plot(
        plot_model,
        eval_loader,
        device,
        plot_k,
        args.kmeans_iters,
        args.seed,
        cluster_plot_path,
    )

    report = {
        "baseline_std": baseline_std,
        "baseline_k": baseline_k,
        "best": best_state,
        "weights": best_weights_path,
        "lr_values": lr_values,
        "dropout_values": dropout_values,
        "k_values": k_values,
        "history": history_path,
        "episode_summaries": episode_path,
        "cluster_plot": cluster_plot_path,
    }
    with open(os.path.join(args.output_dir, "rl_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("RL tuning done")
    print(report)


if __name__ == "__main__":
    main()
