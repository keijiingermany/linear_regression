import csv
import json
import sys
from pathlib import Path

DATA_FILE = "data.csv"
THETAS_FILE = "thetas.json"
LEARNING_RATE = 0.2
ITERATIONS = 2500


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

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
            try:
                x = float(row["km"])
                y = float(row["price"])
            except Exception:
                continue
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        raise ValueError("Dataset must contain at least 2 valid rows")
    return xs, ys


def minmax(values):
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        raise ValueError("Cannot normalize: all values are identical")
    return vmin, vmax


def normalize(values, vmin, vmax):
    scale = vmax - vmin
    return [(v - vmin) / scale for v in values]


def estimate_price(x, theta0, theta1):
    """ hypothesis: theta0 + theta1 * x """
    return theta0 + theta1 * x


def gradient_descent(xs, ys, lr, iterations):
    """ xs, ys are normalized (0..1) """
    theta0 = 0.0
    theta1 = 0.0
    m = len(xs)

    for _ in range(iterations):
        sum_err = 0.0
        sum_err_x = 0.0

        for x, y in zip(xs, ys):
            err = estimate_price(x, theta0, theta1) - y
            sum_err += err
            sum_err_x += err * x

        tmp_theta0 = lr * (sum_err / m)
        tmp_theta1 = lr * (sum_err_x / m)

        # simultaneous update
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1


def denormalize_thetas(theta0_n, theta1_n, x_min, x_max, y_min, y_max):
    """
    y_norm = theta0_n + theta1_n * x_norm
    x_norm = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * y_norm

    => y = theta0 + theta1 * x  (raw scale)
    """
    x_range = x_max - x_min
    y_range = y_max - y_min

    theta1 = (y_range * theta1_n) / x_range
    theta0 = (
        y_min + y_range * theta0_n -
        (y_range * theta1_n * x_min) / x_range
    )
    return theta0, theta1


def save_thetas(path: Path, theta0, theta1):
    payload = {"theta0": theta0, "theta1": theta1}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    try:
        xs_raw, ys_raw = load_data(Path(DATA_FILE))

        x_min, x_max = minmax(xs_raw)
        y_min, y_max = minmax(ys_raw)

        xs = normalize(xs_raw, x_min, x_max)
        ys = normalize(ys_raw, y_min, y_max)

        t0_n, t1_n = gradient_descent(
            xs, ys, LEARNING_RATE, ITERATIONS
        )
        t0, t1 = denormalize_thetas(
            t0_n, t1_n, x_min, x_max, y_min, y_max
        )

        save_thetas(Path(THETAS_FILE), t0, t1)

        print("Training completed")
        print(f"theta0 = {t0}")
        print(f"theta1 = {t1}")

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
