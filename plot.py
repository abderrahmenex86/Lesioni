import json
from pathlib import Path

import matplotlib.pyplot as plt


def get_latest_run_history(model_type, artifacts_directory):
    matching_directories = sorted(list(artifacts_directory.glob(f"*_{model_type}_*")))
    for directory in reversed(matching_directories):
        history_path = directory / "model_history.json"
        if history_path.exists():
            try:
                history_data = json.loads(history_path.read_text())
                if "val" in history_data and "dice" in history_data["val"]:
                    return directory, history_data
            except Exception:
                continue
    return None, None


def get_latest_optuna_summary(model_type, artifacts_directory):
    matching_directories = sorted(list(artifacts_directory.glob(f"*_{model_type}_*")))
    for directory in reversed(matching_directories):
        optuna_path = directory / "optuna_summary.json"
        if optuna_path.exists():
            try:
                optuna_data = json.loads(optuna_path.read_text())
                return directory, optuna_data
            except Exception:
                continue
    return None, None


def main():
    artifacts_directory = Path("artifacts")
    output_directory = Path("docs/figs")
    output_directory.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    model_types = ["unet", "deeplabv3", "linknet", "segformer"]
    model_histories = {}

    for model_type in model_types:
        run_dir, history = get_latest_run_history(model_type, artifacts_directory)
        if history:
            model_histories[model_type] = (run_dir, history)

    if model_histories:
        plt.figure(figsize=(10, 6))
        for model_type, (run_dir, history) in model_histories.items():
            validation_dice = history["val"]["dice"]
            epoch_indices = list(range(1, len(validation_dice) + 1))
            plt.plot(
                epoch_indices,
                validation_dice,
                "o-",
                label=f"{model_type.upper()} ({run_dir.name[:8]})",
                linewidth=2,
                markersize=5,
            )

        plt.title("Model Architecture Performance Comparison (Dice)", fontweight="bold", pad=15)
        plt.xlabel("Training Epoch")
        plt.ylabel("Validation Dice Score (%)")
        plt.legend(loc="lower right")
        plt.ylim(0, 100)
        plt.savefig(output_directory / "01_model_comparison.png", dpi=300)
        plt.close()
        print("[SUCCESS] Model comparison plot (Dice) generated.")

    best_model_type = None
    best_validation_dice_value = -1.0
    best_history = None
    best_directory_name = ""

    for model_type, (run_dir, history) in model_histories.items():
        max_dice = max(history["val"]["dice"])
        if max_dice > best_validation_dice_value:
            best_validation_dice_value = max_dice
            best_model_type = model_type
            best_history = history
            best_directory_name = run_dir.name

    if best_history:
        train_loss = best_history["train"]["loss"]
        val_loss = best_history["val"]["loss"]
        train_dice = best_history["train"]["dice"]
        val_dice = best_history["val"]["dice"]
        train_iou = best_history["train"]["iou"]
        val_iou = best_history["val"]["iou"]
        epoch_indices = list(range(1, len(train_loss) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Peak Performance Diagnostics: {best_model_type.upper()} ({best_directory_name})",
            fontweight="bold",
            fontsize=14,
        )

        axes[0, 0].plot(epoch_indices, train_loss, "o-", color="royalblue", label="Train")
        axes[0, 0].plot(epoch_indices, val_loss, "s--", color="darkorange", label="Val")
        axes[0, 0].set_title("Focal + Dice Loss Convergence")
        axes[0, 0].set_xlabel("Epoch Index")
        axes[0, 0].set_ylabel("Loss Value")
        axes[0, 0].legend()

        axes[0, 1].plot(epoch_indices, train_dice, "o-", color="forestgreen", label="Train")
        axes[0, 1].plot(epoch_indices, val_dice, "s--", color="crimson", label="Val")
        axes[0, 1].set_title("Dice Overlap Percentage (Excl. Background)")
        axes[0, 1].set_xlabel("Epoch Index")
        axes[0, 1].set_ylabel("Dice Score (%)")
        axes[0, 1].legend()

        axes[1, 0].plot(epoch_indices, train_iou, "o-", color="indigo", label="Train")
        axes[1, 0].plot(epoch_indices, val_iou, "s--", color="darkorchid", label="Val")
        axes[1, 0].set_title("Intersection over Union (IoU)")
        axes[1, 0].set_xlabel("Epoch Index")
        axes[1, 0].set_ylabel("IoU (%)")
        axes[1, 0].legend()

        generalization_gap = [validation - training for training, validation in zip(train_loss, val_loss)]
        axes[1, 1].bar(epoch_indices, generalization_gap, color="teal", alpha=0.7, label="Val - Train Loss")
        axes[1, 1].set_title("Generalization Delta Variance")
        axes[1, 1].set_xlabel("Epoch Index")
        axes[1, 1].set_ylabel("Loss Variance")
        axes[1, 1].legend()

        plt.savefig(output_directory / "02_best_run_diagnostics.png", dpi=300)
        plt.close()
        print(f"[SUCCESS] Peak diagnostics plot generated for: {best_model_type.upper()}")

    optuna_target_model = "segformer"
    run_dir, optuna_data = get_latest_optuna_summary(optuna_target_model, artifacts_directory)

    if not optuna_data:
        for model_type in model_types:
            run_dir, optuna_data = get_latest_optuna_summary(model_type, artifacts_directory)
            if optuna_data:
                optuna_target_model = model_type
                break

    if optuna_data and "all_trials" in optuna_data:
        trials = optuna_data["all_trials"]
        trial_indices = []
        trial_values = []
        running_maximums = []
        current_maximum = -1.0

        for trial in sorted(trials, key=lambda x: x["trial_number"]):
            val = trial.get("trial_value")
            if val is not None:
                trial_indices.append(trial["trial_number"])
                trial_values.append(val)
                if val > current_maximum:
                    current_maximum = val
                running_maximums.append(current_maximum)

        if trial_indices:
            plt.figure(figsize=(10, 6))
            plt.scatter(
                trial_indices,
                trial_values,
                color="royalblue",
                alpha=0.7,
                edgecolors="none",
                s=40,
                label="Individual Trial Value",
            )
            plt.plot(
                trial_indices,
                running_maximums,
                drawstyle="steps-post",
                color="crimson",
                linewidth=2.5,
                label="Optimal Trajectory",
            )

            plt.title(
                f"Meta-Heuristic Hyperparameter Convergence: {optuna_target_model.upper()}", fontweight="bold", pad=15
            )
            plt.xlabel("Optuna Trial Index")
            plt.ylabel("Validation Dice Score (%)")
            plt.legend(loc="lower right")
            plt.xticks(trial_indices)
            plt.savefig(output_directory / "03_optuna_convergence.png", dpi=300)
            plt.close()
            print(f"[SUCCESS] Optuna convergence plot generated for: {optuna_target_model.upper()}")

    if model_histories:
        plt.figure(figsize=(10, 6))
        for model_type, (run_dir, history) in model_histories.items():
            train_history_dice = history["train"]["dice"]
            validation_history_dice = history["val"]["dice"]
            generalization_gap_curve = [train - val for train, val in zip(train_history_dice, validation_history_dice)]
            epoch_indices = list(range(1, len(generalization_gap_curve) + 1))
            plt.plot(
                epoch_indices,
                generalization_gap_curve,
                ".-",
                label=f"{model_type.upper()} ({run_dir.name[:8]})",
                linewidth=1.8,
            )

        plt.title("Generalization Gap Comparison (Train Dice - Val Dice)", fontweight="bold", pad=15)
        plt.xlabel("Training Epoch")
        plt.ylabel("Generalization Gap (%)")
        plt.legend(loc="upper right")
        plt.savefig(output_directory / "04_generalization_gap_comparison.png", dpi=300)
        plt.close()
        print("[SUCCESS] Generalization gap comparison plot generated.")

    if optuna_data and "all_trials" in optuna_data:
        trials = optuna_data["all_trials"]
        learning_rates = []
        weight_decays = []
        dice_scores = []

        for trial in trials:
            trial_hyperparameters = trial.get("hyperparameters", {})
            trial_value = trial.get("trial_value")
            if (
                trial_value is not None
                and "learning_rate" in trial_hyperparameters
                and "weight_decay" in trial_hyperparameters
            ):
                learning_rates.append(trial_hyperparameters["learning_rate"])
                weight_decays.append(trial_hyperparameters["weight_decay"])
                dice_scores.append(trial_value)

        if learning_rates:
            plt.figure(figsize=(10, 6))
            scatter_plot_mappable = plt.scatter(
                learning_rates,
                weight_decays,
                c=dice_scores,
                cmap="viridis",
                s=120,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
            )
            color_bar_element = plt.colorbar(scatter_plot_mappable)
            color_bar_element.set_label("Validation Dice Score (%)", fontweight="bold")

            plt.xscale("log")
            plt.yscale("log")
            plt.title(
                f"Hyperparameter Optimization Search Landscape: {optuna_target_model.upper()}",
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Learning Rate (Log Scale)")
            plt.ylabel("Weight Decay (Log Scale)")
            plt.savefig(output_directory / "05_optuna_search_landscape.png", dpi=300)
            plt.close()
            print(f"[SUCCESS] Optuna search landscape plot generated for: {optuna_target_model.upper()}")

    if model_histories:
        plt.figure(figsize=(10, 6))
        for model_type, (run_dir, history) in model_histories.items():
            validation_iou = history["val"]["iou"]
            epoch_indices = list(range(1, len(validation_iou) + 1))
            plt.plot(
                epoch_indices,
                validation_iou,
                "o-",
                label=f"{model_type.upper()} ({run_dir.name[:8]})",
                linewidth=2,
                markersize=5,
            )

        plt.title("Model Architecture Performance Comparison (IoU)", fontweight="bold", pad=15)
        plt.xlabel("Training Epoch")
        plt.ylabel("Validation IoU Score (%)")
        plt.legend(loc="lower right")
        plt.ylim(0, 100)
        plt.savefig(output_directory / "06_multi_model_iou_comparison.png", dpi=300)
        plt.close()
        print("[SUCCESS] Model comparison plot (IoU) generated.")


if __name__ == "__main__":
    main()
