import json
import sys
from pathlib import Path

THETA_FILE = "thetas.json"


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def load_thetas(path: Path):
    if not path.exists():
        return 0.0, 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("theta0", 0.0)), float(data.get("theta1", 0.0))


def main():
    theta0, theta1 = load_thetas(Path(THETA_FILE))

    try:
        s = input("Enter mileage (km): ").strip()
        mileage = float(s)
        if mileage < 0:
            raise ValueError("mileage must be >= 0")
    except Exception:
        print("ERROR")
        sys.exit(1)

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price}")


if __name__ == "__main__":
    main()
