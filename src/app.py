from pathlib import Path
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class IntermittentSalesMLP(nn.Module):
	def __init__(self, input_dim: int, hidden_1: int = 64, hidden_2: int = 32) -> None:
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_1),
			nn.ReLU(),
			nn.Linear(hidden_1, hidden_2),
			nn.ReLU(),
			nn.Linear(hidden_2, 1),
		)

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		return self.network(features)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
	y_true = y_true.astype(np.int32)
	y_pred = y_pred.astype(np.int32)

	tp = int(((y_true == 1) & (y_pred == 1)).sum())
	tn = int(((y_true == 0) & (y_pred == 0)).sum())
	fp = int(((y_true == 0) & (y_pred == 1)).sum())
	fn = int(((y_true == 1) & (y_pred == 0)).sum())

	accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
	precision = tp / max(tp + fp, 1)
	recall = tp / max(tp + fn, 1)
	f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"tp": tp,
		"tn": tn,
		"fp": fp,
		"fn": fn,
	}


def load_and_encode_data(csv_path: Path, max_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
	df = pd.read_csv(
		csv_path,
		usecols=["Event Name", "Event Type", "isSale"],
		nrows=max_rows,
	)

	df["Event Name"] = df["Event Name"].fillna("None")
	df["Event Type"] = df["Event Type"].fillna("None")
	df["isSale"] = df["isSale"].astype(np.float32)

	encoded_features = pd.get_dummies(
		df[["Event Name", "Event Type"]],
		columns=["Event Name", "Event Type"],
		dtype=np.float32,
	)

	x_values = encoded_features.to_numpy(dtype=np.float32)
	y_values = df["isSale"].to_numpy(dtype=np.float32)

	return x_values, y_values


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a basic intermittent sales model")
	parser.add_argument(
		"--data",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "outputs" / "intermittent_data.csv",
		help="Path to intermittent_data.csv",
	)
	parser.add_argument("--max-rows", type=int, default=300000, help="Max rows to load for quick training")
	parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
	parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
	parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
	parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
	args = parser.parse_args()

	torch.manual_seed(42)
	np.random.seed(42)

	x_values, y_values = load_and_encode_data(args.data, args.max_rows)
	unique_labels = np.unique(y_values)
	if unique_labels.size < 2:
		raise ValueError("Target column 'isSale' has only one class in loaded rows. Increase --max-rows.")

	num_rows = x_values.shape[0]
	shuffled_indices = np.random.permutation(num_rows)
	test_count = int(num_rows * args.test_size)

	test_indices = shuffled_indices[:test_count]
	train_indices = shuffled_indices[test_count:]

	x_train = torch.tensor(x_values[train_indices], dtype=torch.float32)
	y_train = torch.tensor(y_values[train_indices], dtype=torch.float32).unsqueeze(1)
	x_test = torch.tensor(x_values[test_indices], dtype=torch.float32)
	y_test = torch.tensor(y_values[test_indices], dtype=torch.float32).unsqueeze(1)

	train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

	if torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	model = IntermittentSalesMLP(input_dim=x_train.shape[1]).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	criterion = nn.BCEWithLogitsLoss()

	print(f"Loaded rows: {num_rows}")
	print(f"Feature dimension after one-hot encoding: {x_train.shape[1]}")
	print(f"Using device: {device}")

	model.train()
	for epoch in range(1, args.epochs + 1):
		epoch_loss = 0.0
		for batch_features, batch_target in train_loader:
			batch_features = batch_features.to(device)
			batch_target = batch_target.to(device)

			optimizer.zero_grad()
			logits = model(batch_features)
			loss = criterion(logits, batch_target)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item() * batch_features.size(0)

		avg_loss = epoch_loss / max(len(train_loader.dataset), 1)
		print(f"Epoch {epoch:02d}/{args.epochs} - loss: {avg_loss:.5f}")

	model.eval()
	with torch.no_grad():
		logits = model(x_test.to(device)).cpu().squeeze(1).numpy()
		probabilities = 1.0 / (1.0 + np.exp(-logits))
		predictions = (probabilities >= 0.5).astype(np.int32)

	metrics = compute_metrics(y_test.squeeze(1).numpy(), predictions)
	print("\nEvaluation on test set")
	print(f"Accuracy : {metrics['accuracy']:.4f}")
	print(f"Precision: {metrics['precision']:.4f}")
	print(f"Recall   : {metrics['recall']:.4f}")
	print(f"F1 Score : {metrics['f1']:.4f}")
	print(f"Confusion Matrix -> TP: {metrics['tp']} TN: {metrics['tn']} FP: {metrics['fp']} FN: {metrics['fn']}")


if __name__ == "__main__":
	main()
