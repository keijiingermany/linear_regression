#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

DATA_FILE = "data.csv"
THETAS_FILE = "thetas.json"
OUTPUT_FILE = "plot.png"


def load_data(path: Path):
    xs, ys = [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if (
            not reader.fieldnames
            or "km" not in reader.fieldnames
            or "price" not in reader.fieldnames
        ):
            raise ValueError("CSV must have headers: km,price")
        for row in reader:
            xs.append(float(row["km"]))
            ys.append(float(row["price"]))
    return xs, ys


def load_thetas(path: Path):
    if not path.exists():
        raise FileNotFoundError("thetas.json not found. Run train.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["theta0"]), float(data["theta1"])


def estimate_price(x, theta0, theta1):
    return theta0 + theta1 * x


def main():
    try:
        xs, ys = load_data(Path(DATA_FILE))
        theta0, theta1 = load_thetas(Path(THETAS_FILE))

        # Calculate R² score
        preds = [estimate_price(x, theta0, theta1) for x in xs]
        m = len(xs)
        y_mean = sum(ys) / m
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)

        x_min, x_max = min(xs), max(xs)

        line_x = [x_min + (x_max - x_min) * i / 1000 for i in range(1001)]
        line_y = [estimate_price(x, theta0, theta1) for x in line_x]

        plt.figure()
        plt.scatter(xs, ys, label="Data")
        plt.plot(line_x, line_y, label=f"Regression line (R²={r2:.4f})")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price")
        plt.title("Car Price Prediction by Mileage")
        plt.legend()

        plt.show()

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
