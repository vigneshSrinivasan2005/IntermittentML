from pathlib import Path
from abc import ABC, abstractmethod
import json

import numpy as np
import pandas as pd
from fromScratchNN import (
	IntermittentSalesMLP,
	WeightedIntermittentSalesMLP,
	WAPEIntermittentSalesMLP,
	DynamicWeightedIntermittentSalesMLP,
)

class BaseModel(ABC):
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


def encode_features(df, categorical_columns, numeric_columns, encoded_columns=None, numeric_stats=None):
	encoded_features = pd.get_dummies(
		df[categorical_columns],
		columns=categorical_columns,
		dtype=np.int8,
	)

	if encoded_columns is None:
		encoded_columns = encoded_features.columns.tolist()
	else:
		encoded_features = encoded_features.reindex(columns=encoded_columns, fill_value=0)

	raw_numeric_features = df[numeric_columns].copy()
	if numeric_stats is None:
		numeric_mean = raw_numeric_features.mean()
		numeric_std = raw_numeric_features.std().replace(0, 1.0)
		numeric_stats = (numeric_mean, numeric_std)
	else:
		numeric_mean, numeric_std = numeric_stats

	normalized_numeric_features = (raw_numeric_features - numeric_mean) / numeric_std
	encoded_array = encoded_features.to_numpy(dtype=np.float32)
	raw_numeric_array = raw_numeric_features.to_numpy(dtype=np.float32)
	normalized_numeric_array = normalized_numeric_features.to_numpy(dtype=np.float32)

	x_values_by_mode = {
		"normalized": np.concatenate([encoded_array, normalized_numeric_array], axis=1),
		"non_normalized": np.concatenate([encoded_array, raw_numeric_array], axis=1),
		"both": np.concatenate([encoded_array, raw_numeric_array, normalized_numeric_array], axis=1),
	}
	y_values = df["isSale"].to_numpy(dtype=np.float32)

	return x_values_by_mode, y_values, encoded_columns, numeric_stats


def evaluate_split(model, x_data, y_data, threshold=0.5):
	logits = model(x_data)
	loss_val, _ = model.compute_loss(logits, y_data)
	probabilities = 1 / (1 + np.exp(-np.squeeze(logits, axis=1)))
	predictions = (probabilities >= threshold).astype(np.int32)
	metrics = compute_metrics(np.squeeze(y_data, axis=1), predictions)
	return metrics, loss_val

def get_batches(x, y, batch_size):
	indices = np.arange(len(x))
	np.random.shuffle(indices)
	for start_idx in range(0, len(x), batch_size):
		excerpt = indices[start_idx:start_idx + batch_size]
		yield x[excerpt], y[excerpt]

def train_model(
	model,
	optimizer,
	epochs,
	batch_size,
	x_train,
	y_train,
	x_validate,
	y_validate,
):
	history_rows = []
	train_metrics, train_eval_loss = evaluate_split(model, x_train, y_train)
	validate_metrics, validate_loss = evaluate_split(model, x_validate, y_validate)
	history_rows.append(
		{
			"epoch": 0,
			"train_loss": train_eval_loss,
			"train_eval_loss": train_eval_loss,
			"validate_loss": validate_loss,
			"train_accuracy": train_metrics["accuracy"],
			"validate_accuracy": validate_metrics["accuracy"],
			"train_precision": train_metrics["precision"],
			"validate_precision": validate_metrics["precision"],
			"train_recall": train_metrics["recall"],
			"validate_recall": validate_metrics["recall"],
			"train_f1": train_metrics["f1"],
			"validate_f1": validate_metrics["f1"],
		}
	)

	print(
		f"Epoch 00/{epochs} - loss: {train_eval_loss:.5f} "
		f"train_acc: {train_metrics['accuracy']:.4f} val_acc: {validate_metrics['accuracy']:.4f} "
		f"val_f1: {validate_metrics['f1']:.4f}"
	)

	for epoch in range(1, epochs + 1):
		epoch_loss = 0.0
		for batch_features, batch_target in get_batches(x_train, y_train, batch_size):
			optimizer.zero_grad()
			logits = model(batch_features)
			loss_val, loss_grad = model.compute_loss(logits, batch_target)
			model.network.backward(loss_grad.T)
			optimizer.step()
			epoch_loss += loss_val * len(batch_features)

		avg_loss = epoch_loss / max(len(x_train), 1)
		train_metrics, train_eval_loss = evaluate_split(model, x_train, y_train)
		validate_metrics, validate_loss = evaluate_split(model, x_validate, y_validate)

		history_rows.append(
			{
				"epoch": epoch,
				"train_loss": avg_loss,
				"train_eval_loss": train_eval_loss,
				"validate_loss": validate_loss,
				"train_accuracy": train_metrics["accuracy"],
				"validate_accuracy": validate_metrics["accuracy"],
				"train_precision": train_metrics["precision"],
				"validate_precision": validate_metrics["precision"],
				"train_recall": train_metrics["recall"],
				"validate_recall": validate_metrics["recall"],
				"train_f1": train_metrics["f1"],
				"validate_f1": validate_metrics["f1"],
			}
		)

		print(
			f"Epoch {epoch:02d}/{epochs} - loss: {avg_loss:.5f} "
			f"train_acc: {train_metrics['accuracy']:.4f} val_acc: {validate_metrics['accuracy']:.4f} "
			f"val_f1: {validate_metrics['f1']:.4f}"
		)

	return history_rows


def evaluate_model(model, x_test):
	predictions = model.predict(x_test, threshold=0.5)
	return predictions.astype(np.int32)


def save_model_logs(model_name, numeric_mode, epoch_history, validate_metrics, evaluate_metrics, output_dir):
	log_dir = output_dir / "model_logs"
	log_dir.mkdir(parents=True, exist_ok=True)

	epoch_metrics_path = log_dir / f"{model_name}_epoch_metrics.csv"
	pd.DataFrame(epoch_history).to_csv(epoch_metrics_path, index=False)

	epoch_df = pd.DataFrame(epoch_history)
	best_epoch_index = int(epoch_df["validate_f1"].idxmax())
	best_epoch = int(epoch_df.iloc[best_epoch_index]["epoch"])

	summary = {
		"model_name": model_name,
		"numeric_mode": numeric_mode,
		"best_validate_f1_epoch": best_epoch,
		"best_validate_f1": float(epoch_df["validate_f1"].max()),
		"final_epoch": int(epoch_df.iloc[-1]["epoch"]),
		"final_train_accuracy": float(epoch_df.iloc[-1]["train_accuracy"]),
		"final_validate_accuracy": float(epoch_df.iloc[-1]["validate_accuracy"]),
		"final_train_f1": float(epoch_df.iloc[-1]["train_f1"]),
		"final_validate_f1": float(epoch_df.iloc[-1]["validate_f1"]),
		"final_validate_metrics": validate_metrics,
		"final_evaluate_metrics": evaluate_metrics,
	}

	summary_path = log_dir / f"{model_name}_summary.json"
	with summary_path.open("w", encoding="utf-8") as summary_file:
		json.dump(summary, summary_file, indent=2)

	print(f"Saved model logs -> {epoch_metrics_path}")
	print(f"Saved model summary -> {summary_path}")


def run_model(
	model_name,
	model,
	numeric_mode,
	x_train,
	y_train,
	x_validate,
	y_validate,
	x_evaluate,
	y_evaluate,
	output_dir,
	epochs,
	batch_size,
	learning_rate,
):
	from fromScratchNN import AdamOptimizer
	if hasattr(model, "set_pos_weight_from_targets"):
		model.set_pos_weight_from_targets(y_train)

	optimizer = AdamOptimizer(model.parameters(), lr=learning_rate)

	print(f"\nRunning model: {model_name}")
	epoch_history = train_model(
		model,
		optimizer,
		epochs,
		batch_size,
		x_train,
		y_train,
		x_validate,
		y_validate,
	)
	validate_metrics, _ = evaluate_split(model, x_validate, y_validate)
	evaluate_metrics, _ = evaluate_split(model, x_evaluate, y_evaluate)

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

	save_model_logs(
		model_name=model_name,
		numeric_mode=numeric_mode,
		epoch_history=epoch_history,
		validate_metrics=validate_metrics,
		evaluate_metrics=evaluate_metrics,
		output_dir=output_dir,
	)


def main():
	output_dir = Path(__file__).resolve().parents[1] / "outputs"
	train_data_path = output_dir / "intermittent_train_data.csv"
	validate_data_path = output_dir / "intermittent_validate_data.csv"
	evaluate_data_path = output_dir / "intermittent_evaluate_data.csv"
	max_rows = 10000000
	epochs = 10
	batch_size = 2048
	learning_rate = 1e-3
	hidden_1 = 64
	hidden_2 = 32
	pos_weight = 2.0

	import time
	np.random.seed(int(time.time()))
	categorical_columns, numeric_columns = get_feature_columns()

	train_df = load_dataset_frame(train_data_path, categorical_columns, numeric_columns, max_rows=max_rows)
	validate_df = load_dataset_frame(validate_data_path, categorical_columns, numeric_columns, max_rows=max_rows)
	evaluate_df = load_dataset_frame(evaluate_data_path, categorical_columns, numeric_columns, max_rows=max_rows)

	x_train_by_mode, y_train_values, encoded_columns, numeric_stats = encode_features(
		train_df,
		categorical_columns,
		numeric_columns,
	)
	x_validate_by_mode, y_validate_values, _, _ = encode_features(
		validate_df,
		categorical_columns,
		numeric_columns,
		encoded_columns,
		numeric_stats,
	)
	x_evaluate_by_mode, y_evaluate_values, _, _ = encode_features(
		evaluate_df,
		categorical_columns,
		numeric_columns,
		encoded_columns,
		numeric_stats,
	)

	unique_labels = np.unique(y_train_values)
	if unique_labels.size < 2:
		raise ValueError("Target column 'isSale' has only one class in train rows.")

	y_train = y_train_values.astype(np.float32).reshape(-1, 1)
	y_validate = y_validate_values.astype(np.float32).reshape(-1, 1)
	y_evaluate = y_evaluate_values.astype(np.float32).reshape(-1, 1)

	print(f"Train rows loaded: {len(y_train)}")
	print(f"Validate rows loaded: {len(y_validate)}")
	print(f"Evaluate rows loaded: {len(y_evaluate)}")
	print(f"Available numeric feature modes: {', '.join(sorted(x_train_by_mode.keys()))}")
	mlp_numeric_mode = "normalized"
	weighted_numeric_mode = "normalized"
	wape_numeric_mode = "normalized"
	dynamic_weighted_numeric_mode = "normalized"

	x_train = x_train_by_mode[mlp_numeric_mode].astype(np.float32)
	x_validate = x_validate_by_mode[mlp_numeric_mode].astype(np.float32)
	x_evaluate = x_evaluate_by_mode[mlp_numeric_mode].astype(np.float32)
	print(f"Preparing model 'mlp' with numeric_mode='{mlp_numeric_mode}' (feature_dim={x_train.shape[1]})")

	mlp_model = IntermittentSalesMLP(input_dim=x_train.shape[1], hidden_1=hidden_1, hidden_2=hidden_2)
	run_model(
		model_name="mlp",
		model=mlp_model,
		numeric_mode=mlp_numeric_mode,
		x_train=x_train,
		y_train=y_train,
		x_validate=x_validate,
		y_validate=y_validate,
		x_evaluate=x_evaluate,
		y_evaluate=y_evaluate,
		output_dir=output_dir,
		epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
	)

	x_train = x_train_by_mode[weighted_numeric_mode].astype(np.float32)
	x_validate = x_validate_by_mode[weighted_numeric_mode].astype(np.float32)
	x_evaluate = x_evaluate_by_mode[weighted_numeric_mode].astype(np.float32)
	print(f"Preparing model 'weighted_mlp' with numeric_mode='{weighted_numeric_mode}' (feature_dim={x_train.shape[1]})")

	weighted_model = WeightedIntermittentSalesMLP(
		input_dim=x_train.shape[1],
		hidden_1=hidden_1,
		hidden_2=hidden_2,
		pos_weight=pos_weight,
	)
	run_model(
		model_name="weighted_mlp",
		model=weighted_model,
		numeric_mode=weighted_numeric_mode,
		x_train=x_train,
		y_train=y_train,
		x_validate=x_validate,
		y_validate=y_validate,
		x_evaluate=x_evaluate,
		y_evaluate=y_evaluate,
		output_dir=output_dir,
		epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
	)

	x_train = x_train_by_mode[wape_numeric_mode].astype(np.float32)
	x_validate = x_validate_by_mode[wape_numeric_mode].astype(np.float32)
	x_evaluate = x_evaluate_by_mode[wape_numeric_mode].astype(np.float32)
	print(f"Preparing model 'wape_mlp' with numeric_mode='{wape_numeric_mode}' (feature_dim={x_train.shape[1]})")

	wape_model = WAPEIntermittentSalesMLP(input_dim=x_train.shape[1], hidden_1=hidden_1, hidden_2=hidden_2)
	run_model(
		model_name="wape_mlp",
		model=wape_model,
		numeric_mode=wape_numeric_mode,
		x_train=x_train,
		y_train=y_train,
		x_validate=x_validate,
		y_validate=y_validate,
		x_evaluate=x_evaluate,
		y_evaluate=y_evaluate,
		output_dir=output_dir,
		epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
	)

	x_train = x_train_by_mode[dynamic_weighted_numeric_mode].astype(np.float32)
	x_validate = x_validate_by_mode[dynamic_weighted_numeric_mode].astype(np.float32)
	x_evaluate = x_evaluate_by_mode[dynamic_weighted_numeric_mode].astype(np.float32)
	print(f"Preparing model 'dynamic_weighted_mlp' with numeric_mode='{dynamic_weighted_numeric_mode}' (feature_dim={x_train.shape[1]})")

	dynamic_weighted_model = DynamicWeightedIntermittentSalesMLP(
		input_dim=x_train.shape[1],
		hidden_1=hidden_1,
		hidden_2=hidden_2,
	)
	run_model(
		model_name="dynamic_weighted_mlp",
		model=dynamic_weighted_model,
		numeric_mode=dynamic_weighted_numeric_mode,
		x_train=x_train,
		y_train=y_train,
		x_validate=x_validate,
		y_validate=y_validate,
		x_evaluate=x_evaluate,
		y_evaluate=y_evaluate,
		output_dir=output_dir,
		epochs=epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
	)

if __name__ == "__main__":
	main()
