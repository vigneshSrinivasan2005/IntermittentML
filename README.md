# IntermittentML
Run to install requirements

```
pip3 install -r src/requirements.txt
```

Download the dataset from: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Extract the CSV files and place them in the `data/` folder:
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`
- `calendar.csv`

Process the data into engineered features:
```
python3 pre-processing/intermittent_products.py
```
Output files will be created in `outputs/`:
- `intermittent_train_data.csv`
- `intermittent_validate_data.csv`
- `intermittent_evaluate_data.csv`


**Run with PyTorch**
```
python3 src/app.py
```

**Run without PyTorch**
```
python3 src/app_no_torch.py
```

Model training logs and metrics are saved to `outputs/model_logs/`:
- `{model_name}_epoch_metrics.csv` - Per-epoch metrics
- `{model_name}_summary.json` - Best epoch and final metrics

Generate plots:
```
python3 src/plot_model_logs.py
```
Plots will be saved to `outputs/model_logs/plots/`


To reproduce specific experimental results, edit the hyperparameters in the `main()` function of `src/app.py` or `src/app_no_torch.py`:

```python
max_rows = 1000000        # Number of rows to load (limit for testing)
epochs = 10               # Training epochs
batch_size = 2048         # Batch size for training
learning_rate = 1e-3      # learning rate
hidden_1 = 64             # First hidden layer dimension
hidden_2 = 32             # Second hidden layer dimension
pos_weight = 2.0          # Class weight for weighted models
```

**Default Configuration** (as in main script):
- Loads up to 1,000,000 rows from each split
- Trains for 10 epochs
- Uses normalized feature mode (standardized numeric features + one-hot encoded categoricals)
- Uses Adam optimizer with learning rate 1e-3
- Random seed: 42 (set in main() for reproducibility)

**To run with limited data for testing:**
- Change `max_rows = 10000` in main()
- This will train faster but may show different metrics due to different data distribution

```

## Interpreting Results

Each model's `summary.json` contains:
- `best_validate_f1_epoch`: Epoch with best validation F1 score
- `final_train_accuracy`, `final_validate_accuracy`: Classification accuracy
- `final_train_f1`, `final_validate_f1`: F1 scores (metric for imbalanced data)
- Final metrics on evaluation set

Epoch metrics CSV files track per-epoch progression:
- `train_loss`, `validate_loss`: Binary cross-entropy loss
- `train_accuracy`, `validate_accuracy`: Fraction of correct predictions
- `train_precision`, `validate_precision`: Of positive predictions, how many were correct
- `train_recall`, `validate_recall`: Of actual positives, how many were found
- `train_f1`, `validate_f1`: Harmonic mean of precision and recall

