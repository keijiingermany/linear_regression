import argparse
import csv
import json
import math
import sys
from pathlib import Path

DATA_FILE = "data.csv"
THETAS_FILE = "thetas.json"


def load_data(path: Path):
    xs, ys = [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["km"]))
            ys.append(float(row["price"]))
    return xs, ys


def load_thetas(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["theta0"]), float(data["theta1"])


def estimate_price(x, theta0, theta1):
    return theta0 + theta1 * x


def main():
    try:
        xs, ys = load_data(Path(DATA_FILE))
        theta0, theta1 = load_thetas(Path(THETAS_FILE))

        preds = [estimate_price(x, theta0, theta1) for x in xs]
        m = len(xs)

        mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / m
        rmse = math.sqrt(mse)
        mae = sum(abs(p - y) for p, y in zip(preds, ys)) / m

        y_mean = sum(ys) / m
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)

        print(f"MSE  : {mse}")
        print(f"RMSE : {rmse}")
        print(f"MAE  : {mae}")
        print(f"R2   : {r2}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
