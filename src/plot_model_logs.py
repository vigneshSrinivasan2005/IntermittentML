from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_model_name(csv_path: Path) -> str:
	name = csv_path.stem
	if name.endswith("_epoch_metrics"):
		return name[: -len("_epoch_metrics")]
	return name


def plot_model_curves(csv_path: Path, output_dir: Path) -> Path:
	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"No rows in log file: {csv_path}")

	model_name = infer_model_name(csv_path)
	fig, ax = plt.subplots(figsize=(10, 5))

	ax.plot(df["epoch"], df["train_accuracy"], label="train_accuracy")
	ax.plot(df["epoch"], df["validate_accuracy"], label="validate_accuracy")
	ax.set_ylabel("Accuracy")
	ax.set_xlabel("Epoch")
	ax.set_title(f"{model_name} Train vs Validation Accuracy")
	ax.grid(True, alpha=0.3)
	ax.legend()

	fig.tight_layout()
	output_path = output_dir / f"{model_name}_accuracy_curve.png"
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return output_path


def plot_confusion_matrix(summary_path: Path, split: str, output_dir: Path) -> Path:
	with summary_path.open(encoding="utf-8") as f:
		summary = json.load(f)

	model_name = summary["model_name"]
	metrics = summary[f"final_{split}_metrics"]

	cm = np.array(
		[
			[metrics["tn"], metrics["fp"]],
			[metrics["fn"], metrics["tp"]],
		],
		dtype=np.int64,
	)

	fig, ax = plt.subplots(figsize=(5, 4))
	im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
	fig.colorbar(im, ax=ax)

	for i in range(2):
		for j in range(2):
			ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
			        color="white" if cm[i, j] > cm.max() / 2 else "black")

	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(["Predicted 0", "Predicted 1"])
	ax.set_yticklabels(["Actual 0", "Actual 1"])
	ax.set_title(f"{model_name} Confusion Matrix ({split})")

	fig.tight_layout()
	output_path = output_dir / f"{model_name}_confusion_matrix_{split}.png"
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return output_path


def plot_model_comparison(log_dir: Path, output_dir: Path) -> Path:
	summary_files = sorted(log_dir.glob("*_summary.json"))
	if not summary_files:
		return None

	models, precisions, recalls, f1s = [], [], [], []
	for summary_file in summary_files:
		with summary_file.open(encoding="utf-8") as f:
			summary = json.load(f)
		metrics = summary["final_evaluate_metrics"]
		models.append(summary["model_name"])
		precisions.append(metrics["precision"])
		recalls.append(metrics["recall"])
		f1s.append(metrics["f1"])

	x = np.arange(len(models))
	width = 0.25

	fig, ax = plt.subplots(figsize=(max(6, len(models) * 3), 5))
	ax.bar(x - width, precisions, width, label="Precision")
	ax.bar(x, recalls, width, label="Recall")
	ax.bar(x + width, f1s, width, label="F1")

	ax.set_xticks(x)
	ax.set_xticklabels(models)
	ax.set_ylim(0, 1.05)
	ax.set_ylabel("Score")
	ax.set_title("Model Comparison (Evaluation Set)")
	ax.legend()
	ax.grid(True, axis="y", alpha=0.3)

	for bar_group, values in zip([x - width, x, x + width], [precisions, recalls, f1s]):
		for bar_x, val in zip(bar_group, values):
			ax.text(bar_x, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

	fig.tight_layout()
	output_path = output_dir / "model_comparison.png"
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return output_path


def main():
	parser = argparse.ArgumentParser(description="Generate training curve plots from model log CSV files.")
	parser.add_argument(
		"--log-dir",
		type=Path,
		default=Path("outputs") / "model_logs",
		help="Directory containing *_epoch_metrics.csv files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory to write PNG plots. Defaults to <log-dir>/plots.",
	)
	args = parser.parse_args()

	log_dir = args.log_dir
	output_dir = args.output_dir if args.output_dir is not None else log_dir / "plots"

	if not log_dir.exists():
		print(f"Log directory not found: {log_dir}")
		print("Run training first to generate per-model epoch logs.")
		return

	csv_files = sorted(log_dir.glob("*_epoch_metrics.csv"))
	if not csv_files:
		print(f"No epoch metric files found in: {log_dir}")
		print("Expected files matching *_epoch_metrics.csv")
		return

	output_dir.mkdir(parents=True, exist_ok=True)
	generated_files = []

	for csv_file in csv_files:
		try:
			plot_path = plot_model_curves(csv_file, output_dir)
			generated_files.append(plot_path)
			print(f"Generated plot: {plot_path}")
		except Exception as exc:
			print(f"Skipped {csv_file}: {exc}")

	for summary_file in sorted(log_dir.glob("*_summary.json")):
		for split in ("validate", "evaluate"):
			try:
				plot_path = plot_confusion_matrix(summary_file, split, output_dir)
				generated_files.append(plot_path)
				print(f"Generated plot: {plot_path}")
			except Exception as exc:
				print(f"Skipped {summary_file} ({split}): {exc}")

	comparison_path = plot_model_comparison(log_dir, output_dir)
	if comparison_path:
		generated_files.append(comparison_path)
		print(f"Generated plot: {comparison_path}")

	if generated_files:
		print("Completed plot generation.")
	else:
		print("No plots were generated.")


if __name__ == "__main__":
	main()
