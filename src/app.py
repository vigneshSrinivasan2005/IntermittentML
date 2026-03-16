from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class BaseModel(nn.Module, ABC):
	@abstractmethod
	def forward(self, features):
		pass

	@abstractmethod
	def compute_loss(self, logits, targets):
		pass

	@abstractmethod
	def initialize_weights(self):
		pass

	@abstractmethod
	def predict_proba(self, features):
		pass

	@abstractmethod
	def predict(self, features, threshold=0.5):
		pass


class IntermittentSalesMLP(BaseModel):
	def __init__(self, input_dim, hidden_1=64, hidden_2=32):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(input_dim, hidden_1),
			nn.ReLU(),
			nn.Linear(hidden_1, hidden_2),
			nn.ReLU(),
			nn.Linear(hidden_2, 1),
		)
		self.loss_fn = nn.BCEWithLogitsLoss()
		self.initialize_weights()

	def forward(self, features):
		return self.network(features)

	def compute_loss(self, logits, targets):
		return self.loss_fn(logits, targets)

	def initialize_weights(self):
		for layer in self.network:
			if isinstance(layer, nn.Linear):
				nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu") # He initialization
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)

	def predict_proba(self, features):
		with torch.no_grad():
			logits = self(features).squeeze(1)
			return torch.sigmoid(logits)

	def predict(self, features, threshold=0.5):
		probabilities = self.predict_proba(features)
		return (probabilities >= threshold).to(dtype=torch.int32)


class WeightedIntermittentSalesMLP(IntermittentSalesMLP):
	def __init__(self, input_dim, hidden_1=64, hidden_2=32, pos_weight=2.0):
		super().__init__(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2)
		self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

	def compute_loss(self, logits, targets):
		return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


MODEL_REGISTRY = {
	"mlp": IntermittentSalesMLP,
	"weighted_mlp": WeightedIntermittentSalesMLP,
}


def compute_metrics(y_true, y_pred):
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


def load_and_encode_data(csv_path, max_rows=None):
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
	)# one hot encoding

	x_values = encoded_features.to_numpy(dtype=np.float32)
	y_values = df["isSale"].to_numpy(dtype=np.float32)

	return x_values, y_values


def build_train_test_tensors(
	x_values,
	y_values,
	test_size,
):
	num_rows = x_values.shape[0]
	shuffled_indices = np.random.permutation(num_rows)
	test_count = int(num_rows * test_size)

	test_indices = shuffled_indices[:test_count]
	train_indices = shuffled_indices[test_count:]

	x_train = torch.tensor(x_values[train_indices], dtype=torch.float32)
	y_train = torch.tensor(y_values[train_indices], dtype=torch.float32).unsqueeze(1)
	x_test = torch.tensor(x_values[test_indices], dtype=torch.float32)
	y_test = torch.tensor(y_values[test_indices], dtype=torch.float32).unsqueeze(1)

	return x_train, y_train, x_test, y_test


def get_device():
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def train_model(
	model,
	train_loader,
	optimizer,
	device,
	epochs,
):
	model.train()
	for epoch in range(1, epochs + 1):
		epoch_loss = 0.0
		for batch_features, batch_target in train_loader:
			batch_features = batch_features.to(device)
			batch_target = batch_target.to(device)

			optimizer.zero_grad()
			logits = model(batch_features)
			loss = model.compute_loss(logits, batch_target)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item() * batch_features.size(0)

		avg_loss = epoch_loss / max(len(train_loader.dataset), 1)
		print(f"Epoch {epoch:02d}/{epochs} - loss: {avg_loss:.5f}")


def evaluate_model(model, x_test, device):
	model.eval()
	predictions = model.predict(x_test.to(device))
	return predictions.cpu().numpy().astype(np.int32)


def run_model(
	model_name,
	model,
	x_train,
	y_train,
	x_test,
	y_test,
	device,
	epochs,
	batch_size,
	learning_rate,
):
	train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	print(f"\nRunning model: {model_name}")
	train_model(model, train_loader, optimizer, device, epochs)
	predictions = evaluate_model(model, x_test, device)

	metrics = compute_metrics(y_test.squeeze(1).numpy(), predictions)
	print("Evaluation on test set")
	print(f"Accuracy : {metrics['accuracy']:.4f}")
	print(f"Precision: {metrics['precision']:.4f}")
	print(f"Recall   : {metrics['recall']:.4f}")
	print(f"F1 Score : {metrics['f1']:.4f}")
	print(f"Confusion Matrix -> TP: {metrics['tp']} TN: {metrics['tn']} FP: {metrics['fp']} FN: {metrics['fn']}")


def main():
	data_path = Path(__file__).resolve().parents[1] / "outputs" / "intermittent_data.csv"
	max_rows = 50000
	epochs = 8
	batch_size = 2048
	learning_rate = 1e-3
	test_size = 0.2
	hidden_1 = 64
	hidden_2 = 32
	pos_weight = 2.0

	torch.manual_seed(42)
	np.random.seed(42)

	x_values, y_values = load_and_encode_data(data_path, max_rows=max_rows)
	unique_labels = np.unique(y_values)
	if unique_labels.size < 2:
		raise ValueError("Target column 'isSale' has only one class in loaded rows.")

	x_train, y_train, x_test, y_test = build_train_test_tensors(x_values, y_values, test_size)

	device = get_device()

	num_rows = x_values.shape[0]
	print(f"Loaded rows: {num_rows}")
	print(f"Feature dimension after one-hot encoding: {x_train.shape[1]}")
	print(f"Using device: {device}")

	mlp_model = IntermittentSalesMLP(input_dim=x_train.shape[1], hidden_1=hidden_1, hidden_2=hidden_2).to(device)
	run_model(
		model_name="mlp",
		model=mlp_model,
		x_train=x_train,
		y_train=y_train,
		x_test=x_test,
		y_test=y_test,
		device=device,
		epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
	)

	weighted_model = WeightedIntermittentSalesMLP(
		input_dim=x_train.shape[1],
		hidden_1=hidden_1,
		hidden_2=hidden_2,
		pos_weight=pos_weight,
	).to(device)
	run_model(
		model_name="weighted_mlp",
		model=weighted_model,
		x_train=x_train,
		y_train=y_train,
		x_test=x_test,
		y_test=y_test,
		device=device,
		epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
	)


if __name__ == "__main__":
	main()
