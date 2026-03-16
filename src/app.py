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
			if isinstance(layer, nn.Linear) and not layer is self.network[-1]:  # only for relu layers
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


class WAPEIntermittentSalesMLP(IntermittentSalesMLP):
	def __init__(self, input_dim, hidden_1=64, hidden_2=32):
		super().__init__(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2)

	def compute_loss(self, logits, targets):
		predictions = torch.sigmoid(logits)
		numerator = torch.sum(torch.abs(targets - predictions))
		denominator = torch.clamp(torch.sum(torch.abs(targets)), min=1e-6)
		return numerator / denominator


class DynamicWeightedIntermittentSalesMLP(IntermittentSalesMLP):
	def __init__(self, input_dim, hidden_1=64, hidden_2=32):
		super().__init__(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2)
		self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))

	def set_pos_weight_from_targets(self, y_train, device):
		total_ones = y_train.sum().item()
		total_zeros = len(y_train) - total_ones

		if total_ones <= 0:
			dynamic_weight = 1.0
		else:
			dynamic_weight = total_zeros / total_ones

		print(f"Calculated Dynamic pos_weight: {dynamic_weight:.4f}")
		self.pos_weight = torch.tensor([dynamic_weight], device=device, dtype=torch.float32)

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


def get_feature_columns():
	categorical_columns = [
		"Weekday",
		"Month",
		"Event Name",
		"Event Type",
		"dept Id",
		"store id",
	]
	numeric_columns = [
		"Price Change Percentage 7d",
		"Price Change Percentage 30d",
		"Time Since Last Sale",
		"3-Day Rolling Sales",
		"7-Day Rolling Sales",
		"28-Day Rolling Sales",
	]
	return categorical_columns, numeric_columns


def load_dataset_frame(csv_path, categorical_columns, numeric_columns, max_rows=None):
	df = pd.read_csv(
		csv_path,
		usecols=categorical_columns + numeric_columns + ["isSale"],
		nrows=max_rows,
	)

	for column in categorical_columns:
		df[column] = df[column].fillna("None").astype(str)
	for column in numeric_columns:
		df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(np.float32)
	df["isSale"] = df["isSale"].astype(np.float32)
	return df


def encode_features(df, categorical_columns, numeric_columns, encoded_columns=None):
	encoded_features = pd.get_dummies(
		df[categorical_columns],
		columns=categorical_columns,
		dtype=np.int8,
	)

	if encoded_columns is None:
		encoded_columns = encoded_features.columns.tolist()
	else:
		encoded_features = encoded_features.reindex(columns=encoded_columns, fill_value=0)

	numeric_features = df[numeric_columns]

	x_values = np.concatenate(
		[
			encoded_features.to_numpy(dtype=np.float32),
			numeric_features.to_numpy(dtype=np.float32),
		],
		axis=1,
	)
	y_values = df["isSale"].to_numpy(dtype=np.float32)

	return x_values, y_values, encoded_columns


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
	x_validate,
	y_validate,
	x_evaluate,
	y_evaluate,
	device,
	epochs,
	batch_size,
	learning_rate,
):
	if hasattr(model, "set_pos_weight_from_targets"):
		model.set_pos_weight_from_targets(y_train, device)

	train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	print(f"\nRunning model: {model_name}")
	train_model(model, train_loader, optimizer, device, epochs)
	validate_predictions = evaluate_model(model, x_validate, device)
	evaluate_predictions = evaluate_model(model, x_evaluate, device)

	validate_metrics = compute_metrics(y_validate.squeeze(1).numpy(), validate_predictions)
	evaluate_metrics = compute_metrics(y_evaluate.squeeze(1).numpy(), evaluate_predictions)

	print("Validation metrics")
	print(f"Accuracy : {validate_metrics['accuracy']:.4f}")
	print(f"Precision: {validate_metrics['precision']:.4f}")
	print(f"Recall   : {validate_metrics['recall']:.4f}")
	print(f"F1 Score : {validate_metrics['f1']:.4f}")
	print(f"Confusion Matrix -> TP: {validate_metrics['tp']} TN: {validate_metrics['tn']} FP: {validate_metrics['fp']} FN: {validate_metrics['fn']}")

	print("Evaluation metrics")
	print(f"Accuracy : {evaluate_metrics['accuracy']:.4f}")
	print(f"Precision: {evaluate_metrics['precision']:.4f}")
	print(f"Recall   : {evaluate_metrics['recall']:.4f}")
	print(f"F1 Score : {evaluate_metrics['f1']:.4f}")
	print(f"Confusion Matrix -> TP: {evaluate_metrics['tp']} TN: {evaluate_metrics['tn']} FP: {evaluate_metrics['fp']} FN: {evaluate_metrics['fn']}")


def main():
	output_dir = Path(__file__).resolve().parents[1] / "outputs"
	train_data_path = output_dir / "intermittent_train_data.csv"
	validate_data_path = output_dir / "intermittent_validate_data.csv"
	evaluate_data_path = output_dir / "intermittent_evaluate_data.csv"
	max_rows = 1000000
	epochs = 10
	batch_size = 2048
	learning_rate = 1e-3
	hidden_1 = 64
	hidden_2 = 32
	pos_weight = 2.0

	torch.manual_seed(42)
	np.random.seed(42)
	categorical_columns, numeric_columns = get_feature_columns()

	train_df = load_dataset_frame(train_data_path, categorical_columns, numeric_columns, max_rows=max_rows)
	validate_df = load_dataset_frame(validate_data_path, categorical_columns, numeric_columns, max_rows=max_rows)
	evaluate_df = load_dataset_frame(evaluate_data_path, categorical_columns, numeric_columns, max_rows=max_rows)

	x_train_values, y_train_values, encoded_columns = encode_features(train_df, categorical_columns, numeric_columns)
	x_validate_values, y_validate_values, _ = encode_features(validate_df, categorical_columns, numeric_columns, encoded_columns)
	x_evaluate_values, y_evaluate_values, _ = encode_features(evaluate_df, categorical_columns, numeric_columns, encoded_columns)

	unique_labels = np.unique(y_train_values)
	if unique_labels.size < 2:
		raise ValueError("Target column 'isSale' has only one class in train rows.")

	x_train = torch.tensor(x_train_values, dtype=torch.float32)
	y_train = torch.tensor(y_train_values, dtype=torch.float32).unsqueeze(1)
	x_validate = torch.tensor(x_validate_values, dtype=torch.float32)
	y_validate = torch.tensor(y_validate_values, dtype=torch.float32).unsqueeze(1)
	x_evaluate = torch.tensor(x_evaluate_values, dtype=torch.float32)
	y_evaluate = torch.tensor(y_evaluate_values, dtype=torch.float32).unsqueeze(1)

	device = get_device()

	print(f"Train rows loaded: {len(x_train)}")
	print(f"Validate rows loaded: {len(x_validate)}")
	print(f"Evaluate rows loaded: {len(x_evaluate)}")
	print(f"Feature dimension after one-hot encoding: {x_train.shape[1]}")
	print(f"Using device: {device}")

	#mlp_model = IntermittentSalesMLP(input_dim=x_train.shape[1], hidden_1=hidden_1, hidden_2=hidden_2).to(device)
	#run_model(
	#	model_name="mlp",
	#	model=mlp_model,
	#	x_train=x_train,
	#	y_train=y_train,
	#	x_test=x_test,
	#	y_test=y_test,
	#	device=device,
	#	epochs=epochs,
	#	batch_size=batch_size,
	#	learning_rate=learning_rate,
	#)

	# weighted_model = WeightedIntermittentSalesMLP(
	# 	input_dim=x_train.shape[1],
	# 	hidden_1=hidden_1,
	# 	hidden_2=hidden_2,
	# 	pos_weight=pos_weight,
	# ).to(device)
	# run_model(
	# 	model_name="weighted_mlp",
	# 	model=weighted_model,
	# 	x_train=x_train,
	# 	y_train=y_train,
	# 	x_validate=x_validate,
	# 	y_validate=y_validate,
	# 	x_evaluate=x_evaluate,
	# 	y_evaluate=y_evaluate,
	# 	device=device,
	# 	epochs=epochs,
	# 	batch_size=batch_size,
	# 	learning_rate=learning_rate,
	# )
	# wape_model = WAPEIntermittentSalesMLP(input_dim=x_train.shape[1], hidden_1=hidden_1, hidden_2=hidden_2).to(device)
	# run_model(
	# 	model_name="wape_mlp",
	# 	model=wape_model,
	# 	x_train=x_train,	
	# 	y_train=y_train,
	# 	x_validate=x_validate,
	# 	y_validate=y_validate,	
	# 	x_evaluate=x_evaluate,	
	# 	y_evaluate=y_evaluate,
	# 	device=device,
	# 	epochs=epochs,
	# 	batch_size=batch_size,
	# 	learning_rate=learning_rate,
	# )

	dynamic_weighted_model = DynamicWeightedIntermittentSalesMLP(
		input_dim=x_train.shape[1],
		hidden_1=hidden_1,
		hidden_2=hidden_2,
	).to(device)
	run_model(
		model_name="dynamic_weighted_mlp",
		model=dynamic_weighted_model,
		x_train=x_train,
		y_train=y_train,
		x_validate=x_validate,
		y_validate=y_validate,
		x_evaluate=x_evaluate,
		y_evaluate=y_evaluate,
		device=device,
		epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
	)

if __name__ == "__main__":
	main()
