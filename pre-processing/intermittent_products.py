# IDENTIFY INTERMITTENT PRODUCTS FROM THE DATASET
# USING CV^2 AND ADI condition if ADI > 1.32 AND CV^2 <= 0.49

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Identify intermittent products and export calendar-enriched sales rows"
	)
	project_root = Path(__file__).resolve().parents[1]
	parser.add_argument(
		"--sales-file",
		type=Path,
		default=project_root / "data" / "sales_train_validation.csv",
		help="Path to sales training CSV",
	)
	parser.add_argument(
		"--calendar-file",
		type=Path,
		default=project_root / "data" / "calendar.csv",
		help="Path to calendar CSV",
	)
	parser.add_argument(
		"--output-file",
		type=Path,
		default=project_root / "outputs" / "intermittent_data.csv",
		help="Path for output CSV",
	)
	parser.add_argument(
		"--adi-threshold",
		type=float,
		default=1.32,
		help="Intermittent threshold for ADI",
	)
	parser.add_argument(
		"--cv2-threshold",
		type=float,
		default=0.49,
		help="Intermittent threshold for CV^2",
	)
	return parser.parse_args()


def load_calendar_map(calendar_file: Path) -> dict[str, tuple[str, str, str]]:
	calendar_map: dict[str, tuple[str, str, str]] = {}
	with calendar_file.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			day_key = row.get("d", "")
			weekday = row.get("weekday", "")
			event_name_1 = row.get("event_name_1", "")
			event_name_2 = row.get("event_name_2", "")
			event_type_1 = row.get("event_type_1", "")
			event_type_2 = row.get("event_type_2", "")

			event_name = event_name_1 if event_name_1 else event_name_2
			event_type = event_type_1 if event_type_1 else event_type_2
			calendar_map[day_key] = (weekday, event_name, event_type)
	return calendar_map


def compute_adi_cv2(daily_sales: list[int]) -> tuple[float, float]:
	n_days = len(daily_sales)
	non_zero = [value for value in daily_sales if value > 0]
	n_non_zero = len(non_zero)

	if n_non_zero == 0:
		return float("inf"), float("inf")

	adi = n_days / n_non_zero

	mean_demand = sum(non_zero) / n_non_zero
	if n_non_zero < 2 or mean_demand == 0:
		cv2 = float("inf")
	else:
		variance = sum((value - mean_demand) ** 2 for value in non_zero) / (n_non_zero - 1)
		cv = math.sqrt(variance) / mean_demand
		cv2 = cv * cv

	return adi, cv2


def main() -> None:
	args = parse_args()

	if not args.sales_file.exists():
		raise FileNotFoundError(f"Sales file not found: {args.sales_file}")
	if not args.calendar_file.exists():
		raise FileNotFoundError(f"Calendar file not found: {args.calendar_file}")

	args.output_file.parent.mkdir(parents=True, exist_ok=True)
	calendar_map = load_calendar_map(args.calendar_file)

	processed_items = 0
	intermittent_items = 0
	output_rows = 0

	with args.sales_file.open("r", newline="", encoding="utf-8") as sales_handle, \
		args.output_file.open("w", newline="", encoding="utf-8") as out_handle:
		reader = csv.reader(sales_handle)
		header = next(reader, [])

		if len(header) < 7:
			raise ValueError("Sales file does not have expected columns")

		day_indices = [index for index, name in enumerate(header) if name.startswith("d_")]
		day_labels = [header[index] for index in day_indices]

		writer = csv.writer(out_handle)
		writer.writerow([
			"Weekday",
			"Event Name",
			"Event Type",
			"item Id",
			"dept Id",
			"store id",
			"isSale",
		])

		for row in reader:
			processed_items += 1
			item_id = row[1]
			dept_id = row[2]
			store_id = row[4]

			daily_sales: list[int] = []
			for index in day_indices:
				value = row[index].strip()
				daily_sales.append(int(value) if value else 0)

			adi, cv2 = compute_adi_cv2(daily_sales)
			is_intermittent = adi > args.adi_threshold and cv2 <= args.cv2_threshold

			if not is_intermittent:
				continue

			intermittent_items += 1

			for day_label, sale_value in zip(day_labels, daily_sales):
				weekday, event_name, event_type = calendar_map.get(day_label, ("", "", ""))
				is_sale = 1 if sale_value > 0 else 0
				writer.writerow([
					weekday,
					event_name,
					event_type,
					item_id,
					dept_id,
					store_id,
					is_sale,
				])
				output_rows += 1

			if processed_items % 1000 == 0:
				print(f"Processed items: {processed_items:,} | Intermittent items: {intermittent_items:,}")

	print("Done")
	print(f"Items processed: {processed_items:,}")
	print(f"Intermittent items: {intermittent_items:,}")
	print(f"Output rows written: {output_rows:,}")
	print(f"Output file: {args.output_file}")


if __name__ == "__main__":
	main()