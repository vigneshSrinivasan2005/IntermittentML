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
		"--prices-file",
		type=Path,
		default=project_root / "data" / "sell_prices.csv",
		help="Path to sell prices CSV",
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


def load_calendar_map(calendar_file: Path) -> dict[str, dict[str, str]]:
	calendar_map: dict[str, dict[str, str]] = {}
	with calendar_file.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			day_key = row.get("d", "")
			weekday = row.get("weekday", "")
			month = row.get("month", "")
			wm_yr_wk = row.get("wm_yr_wk", "")
			event_name_1 = row.get("event_name_1", "")
			event_name_2 = row.get("event_name_2", "")
			event_type_1 = row.get("event_type_1", "")
			event_type_2 = row.get("event_type_2", "")

			event_name = event_name_1 if event_name_1 else event_name_2
			event_type = event_type_1 if event_type_1 else event_type_2
			calendar_map[day_key] = {
				"weekday": weekday,
				"month": month,
				"wm_yr_wk": wm_yr_wk,
				"event_name": event_name,
				"event_type": event_type,
			}
	return calendar_map


def load_price_map(prices_file: Path) -> dict[tuple[str, str, str], float]:
	price_map: dict[tuple[str, str, str], float] = {}
	with prices_file.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			store_id = row.get("store_id", "")
			item_id = row.get("item_id", "")
			wm_yr_wk = row.get("wm_yr_wk", "")
			sell_price = row.get("sell_price", "")
			if not store_id or not item_id or not wm_yr_wk or not sell_price:
				continue
			price_map[(store_id, item_id, wm_yr_wk)] = float(sell_price)
	return price_map


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


def build_price_series(
	day_labels: list[str],
	calendar_map: dict[str, dict[str, str]],
	price_map: dict[tuple[str, str, str], float],
	store_id: str,
	item_id: str,
) -> list[float]:
	price_series: list[float] = []
	last_known_price = 0.0

	for day_label in day_labels:
		calendar_info = calendar_map.get(day_label, {})
		wm_yr_wk = calendar_info.get("wm_yr_wk", "")
		current_price = price_map.get((store_id, item_id, wm_yr_wk), last_known_price)
		price_series.append(current_price)
		if current_price > 0:
			last_known_price = current_price

	return price_series


def compute_price_change_percentages(price_series: list[float], lookback_days: int) -> list[float]:
	changes: list[float] = []
	for index, current_price in enumerate(price_series):
		if index < lookback_days:
			changes.append(0.0)
			continue

		previous_price = price_series[index - lookback_days]
		if previous_price <= 0 or current_price <= 0:
			changes.append(0.0)
			continue

		changes.append((current_price - previous_price) / previous_price)

	return changes


def compute_time_since_last_sale(daily_sales: list[int]) -> list[int]:
	time_since_last_sale: list[int] = []
	last_sale_index = -1

	for index, sale_value in enumerate(daily_sales):
		if last_sale_index == -1:
			time_since_last_sale.append(-1)
		else:
			time_since_last_sale.append(index - last_sale_index)

		if sale_value > 0:
			last_sale_index = index

	return time_since_last_sale


def main() -> None:
	args = parse_args()

	if not args.sales_file.exists():
		raise FileNotFoundError(f"Sales file not found: {args.sales_file}")
	if not args.calendar_file.exists():
		raise FileNotFoundError(f"Calendar file not found: {args.calendar_file}")
	if not args.prices_file.exists():
		raise FileNotFoundError(f"Prices file not found: {args.prices_file}")

	args.output_file.parent.mkdir(parents=True, exist_ok=True)
	calendar_map = load_calendar_map(args.calendar_file)
	price_map = load_price_map(args.prices_file)

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
			"Month",
			"Event Name",
			"Event Type",
			"item Id",
			"dept Id",
			"store id",
			"Price Change Percentage 7d",
			"Price Change Percentage 30d",
			"Time Since Last Sale",
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
			price_series = build_price_series(day_labels, calendar_map, price_map, store_id, item_id)
			price_change_7d = compute_price_change_percentages(price_series, 7)
			price_change_30d = compute_price_change_percentages(price_series, 30)
			time_since_last_sale = compute_time_since_last_sale(daily_sales)

			for index, (day_label, sale_value) in enumerate(zip(day_labels, daily_sales)):
				calendar_info = calendar_map.get(day_label, {})
				weekday = calendar_info.get("weekday", "")
				month = calendar_info.get("month", "")
				event_name = calendar_info.get("event_name", "")
				event_type = calendar_info.get("event_type", "")
				is_sale = 1 if sale_value > 0 else 0
				writer.writerow([
					weekday,
					month,
					event_name,
					event_type,
					item_id,
					dept_id,
					store_id,
					price_change_7d[index],
					price_change_30d[index],
					time_since_last_sale[index],
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