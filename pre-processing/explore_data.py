import argparse
import csv
from pathlib import Path


def summarize_csv(path: Path, preview_rows: int) -> None:
	with path.open("r", newline="", encoding="utf-8") as csv_file:
		reader = csv.reader(csv_file)
		header = next(reader, [])

		row_count = 0
		preview = []
		for row in reader:
			row_count += 1
			if len(preview) < preview_rows:
				preview.append(row)

	print(f"\n=== {path.name} ===")
	print(f"Rows: {row_count:,}")
	print(f"Columns: {len(header):,}")
	print(f"File size: {path.stat().st_size / (1024 * 1024):.2f} MB")

	if header:
		shown_cols = ", ".join(header[:10])
		suffix = " ..." if len(header) > 10 else ""
		print(f"Header (first 10): {shown_cols}{suffix}")

	if preview:
		print("Sample rows:")
		for idx, row in enumerate(preview, start=1):
			shown_vals = ", ".join(row[:10])
			suffix = " ..." if len(row) > 10 else ""
			print(f"  {idx}. {shown_vals}{suffix}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simple CSV data explorer")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "data",
		help="Directory containing CSV files",
	)
	parser.add_argument(
		"--preview-rows",
		type=int,
		default=2,
		help="Number of sample rows to print per file",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	data_dir = args.data_dir

	if not data_dir.exists() or not data_dir.is_dir():
		raise FileNotFoundError(f"Data directory not found: {data_dir}")

	csv_files = sorted(data_dir.glob("*.csv"))
	if not csv_files:
		print(f"No CSV files found in {data_dir}")
		return

	print(f"Data directory: {data_dir}")
	print(f"CSV files found: {len(csv_files)}")

	for csv_path in csv_files:
		summarize_csv(csv_path, preview_rows=max(args.preview_rows, 0))


if __name__ == "__main__":
	main()
