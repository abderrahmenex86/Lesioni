#!/usr/bin/env python3
import json
import os
import re

import matplotlib.pyplot as plt


def find_historical_histories(base_dir="vast"):
    """
    Scans the vast/ directory for historical histories v1-v5.
    Returns a dict mapping version names to their (history_json, train_script) paths.
    """
    paths_map = {}
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory '{base_dir}' not found.")
        return paths_map

    for i in range(1, 6):
        v_name = f"v{i}"
        v_dir = os.path.join(base_dir, v_name)
        if not os.path.exists(v_dir):
            continue

        # 1. Locate history file
        history_path = None
        direct_hist = os.path.join(v_dir, "training_history.json")
        if os.path.exists(direct_hist):
            history_path = direct_hist
        else:
            nested_hist_dir = os.path.join(v_dir, "history")
            if os.path.exists(nested_hist_dir):
                json_files = [f for f in os.listdir(nested_hist_dir) if f.endswith(".json")]
                if json_files:
                    history_path = os.path.join(nested_hist_dir, json_files[0])

        # 2. Locate train script
        train_path = None
        direct_train = os.path.join(v_dir, "train.py")
        if os.path.exists(direct_train):
            train_path = direct_train
        else:
            vast_sub = os.path.join(v_dir, "vast")
            if os.path.exists(vast_sub) and os.path.isdir(vast_sub):
                for f in os.listdir(vast_sub):
                    if f.endswith("train.py"):
                        train_path = os.path.join(vast_sub, f)
                        break

        if history_path:
            paths_map[v_name] = {"history": history_path, "train_script": train_path}

    return paths_map


def parse_hyperparams_from_script(filepath):
    """
    Statically reads the training script to parse hyperparameters.
    """
    meta = {"model": "U-Net", "batch_size": "?", "epochs": "?", "lrs": []}  # Default baseline fallback
    if not filepath or not os.path.exists(filepath):
        return meta

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse instantiated model class (e.g. model = UNetDiseaseSegmenter(...) or model = DiseaseSegmenter(...))
        model_match = re.search(r"model\s*=\s*([a-zA-Z0-9_]+)\(", content)
        if model_match:
            class_name = model_match.group(1)
            # Simplify names for legend readability
            if "UNet" in class_name:
                meta["model"] = "U-Net"
            elif "DeepLab" in class_name or "DiseaseSegmenter" in class_name:
                # In your historical runs, DiseaseSegmenter was often DeepLabV3 setup
                meta["model"] = "DeepLabV3" if "deeplab" in content.lower() else "U-Net"
            else:
                meta["model"] = class_name

        # Parse Batch Size
        bs_match = re.search(r"batch_size\s*=\s*(\d+)", content)
        if bs_match:
            meta["batch_size"] = bs_match.group(1)

        # Parse Epochs (safely evaluating multiplication configurations like 40 * 2)
        epochs_match = re.search(r"n_epochs\s*=\s*([\d\s*+]+)", content)
        if not epochs_match:
            epochs_match = re.search(r"num_epochs\s*=\s*([\d\s*+]+)", content)

        if epochs_match:
            expr = epochs_match.group(1).strip()
            try:
                meta["epochs"] = str(eval(expr, {"__builtins__": None}, {}))
            except Exception:
                meta["epochs"] = expr

        # Parse learning rates (looking for patterns like "lr": 1e-5 or lr=1e-4)
        lr_matches = re.findall(r'(?:"lr"|lr)\s*[:=]\s*([0-9e.-]+)', content)
        if lr_matches:
            unique_lrs = []
            for lr in lr_matches:
                if lr not in unique_lrs:
                    unique_lrs.append(lr)
            meta["lrs"] = unique_lrs

    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")

    return meta


def load_and_normalize_history(filepath):
    """
    Loads JSON history and normalizes double-scaled metrics dynamically.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    if "train" in data and "val" in data:
        train_metrics = data["train"]
        val_metrics = data["val"]
    else:
        train_metrics = {k.replace("train_", ""): v for k, v in data.items() if k.startswith("train_")}
        val_metrics = {k.replace("val_", ""): v for k, v in data.items() if k.startswith("val_")}

    def normalize_list(metric_list):
        if not metric_list:
            return []
        if any(v > 100.0 for v in metric_list):
            return [v / 100.0 for v in metric_list]
        return list(metric_list)

    return {
        "train_loss": train_metrics.get("loss", []),
        "val_loss": val_metrics.get("loss", []),
        "train_dice": normalize_list(train_metrics.get("dice", [])),
        "val_dice": normalize_list(val_metrics.get("dice", [])),
        "train_iou": train_metrics.get("iou", []),
        "val_iou": val_metrics.get("iou", []),
    }


def main():
    print("Beginning comparative parameters mapping...")
    os.makedirs("docs/figs", exist_ok=True)

    historical_map = find_historical_histories("vast")
    if not historical_map:
        print("No historical histories located. Place this script in the root folder containing the 'vast/' directory.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    peak_summary = {"versions": [], "dice": [], "iou": []}
    colors = {"v1": "royalblue", "v2": "darkorange", "v3": "forestgreen", "v4": "crimson", "v5": "purple"}

    for v_name in sorted(historical_map.keys()):
        paths = historical_map[v_name]
        try:
            # Statically parse training script for legend labeling
            meta = parse_hyperparams_from_script(paths["train_script"])

            # Format high-information label
            label_parts = []
            if meta["batch_size"] != "?":
                label_parts.append(f"BS:{meta['batch_size']}")
            if meta["lrs"]:
                lrs_str = "/".join(meta["lrs"])
                label_parts.append(f"LR:{lrs_str}")

            # Combine into: v4: U-Net (BS:8 | LR:1e-5/1e-3)
            meta_str = " | ".join(label_parts)
            legend_label = f"{v_name}: {meta['model']} ({meta_str})"

            metrics = load_and_normalize_history(paths["history"])
            epochs = list(range(1, len(metrics["val_loss"]) + 1))
            color = colors.get(v_name, "black")

            # Subplot 1: Validation Loss
            axs[0, 0].plot(epochs, metrics["val_loss"], label=legend_label, color=color, alpha=0.85, linewidth=1.75)

            # Subplot 2: Validation Dice Score
            if metrics["val_dice"]:
                axs[0, 1].plot(epochs, metrics["val_dice"], label=legend_label, color=color, alpha=0.85, linewidth=1.75)

            # Subplot 3: Validation IoU
            if metrics["val_iou"]:
                axs[1, 0].plot(epochs, metrics["val_iou"], label=legend_label, color=color, alpha=0.85, linewidth=1.75)

            # Compute benchmarks
            max_dice = max(metrics["val_dice"]) if metrics["val_dice"] else 0.0
            max_iou = max(metrics["val_iou"]) if metrics["val_iou"] else 0.0

            peak_summary["versions"].append(f"{v_name}\n({meta['model']})")
            peak_summary["dice"].append(max_dice)
            peak_summary["iou"].append(max_iou)

        except Exception as e:
            print(f"Failed parsing run {v_name}: {e}")

    # Subplot Configuration
    axs[0, 0].set_title("Cross-Entropy + Dice Objective Value (Val)", fontsize=12, fontweight="bold")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend(fontsize=9, loc="upper right")

    axs[0, 1].set_title("Boundary Overlap (Dice Score %)", fontsize=12, fontweight="bold")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Dice Score (%)")
    axs[0, 1].legend(fontsize=9, loc="lower right")

    axs[1, 0].set_title("Mean Intersection-over-Union (IoU %)", fontsize=12, fontweight="bold")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Mean IoU (%)")
    axs[1, 0].legend(fontsize=9, loc="lower right")

    # Subplot 4: Peaks comparative summary bar chart
    x_indices = range(len(peak_summary["versions"]))
    width = 0.35

    axs[1, 1].bar(
        [x - width / 2 for x in x_indices], peak_summary["dice"], width, label="Peak Dice %", color="teal", alpha=0.8
    )
    axs[1, 1].bar(
        [x + width / 2 for x in x_indices],
        peak_summary["iou"],
        width,
        label="Peak IoU %",
        color="darkorchid",
        alpha=0.8,
    )
    axs[1, 1].set_title("Model Peak Metrics Benchmark", fontsize=12, fontweight="bold")
    axs[1, 1].set_xticks(list(x_indices))
    axs[1, 1].set_xticklabels(peak_summary["versions"])
    axs[1, 1].set_ylabel("Percentage (%)")
    axs[1, 1].legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out_path = "docs/figs/training_metrics_grid.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plotting complete. comparative figures saved to: {out_path}")


if __name__ == "__main__":
    main()
