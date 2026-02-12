#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
    return theta0 + theta1 * x


def gradient_descent(xs, ys, lr, iterations):
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

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1


def denormalize_thetas(theta0_n, theta1_n, x_min, x_max, y_min, y_max):
    x_range = x_max - x_min
    y_range = y_max - y_min

    theta1 = (y_range * theta1_n) / x_range
    theta0 = (
        y_min + y_range * theta0_n -
        (y_range * theta1_n * x_min) / x_range
    )
    return theta0, theta1


def calculate_metrics(xs, ys, theta0, theta1):
    """Calculate all evaluation metrics"""
    preds = [estimate_price(x, theta0, theta1) for x in xs]
    m = len(xs)
    
    # MSE, RMSE, MAE
    mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / m
    rmse = math.sqrt(mse)
    mae = sum(abs(p - y) for p, y in zip(preds, ys)) / m
    
    # R²
    y_mean = sum(ys) / m
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


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

        # Grid search parameters
        learning_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        iterations_list = [50, 100, 500, 1000, 2000, 2500, 3000]

        best_score = -float('inf')
        best_lr = 0
        best_iter = 0
        best_theta0 = 0
        best_theta1 = 0
        best_metrics = {}

        results = []

        print("Testing different hyperparameters...")
        print("-" * 80)
        print(f"{'LR':<6} {'Iter':<6} {'R²':<10} {'MSE':<12} {'RMSE':<10} {'MAE':<10}")
        print("-" * 80)

        for lr in learning_rates:
            for iterations in iterations_list:
                t0_n, t1_n = gradient_descent(xs, ys, lr, iterations)
                t0, t1 = denormalize_thetas(t0_n, t1_n, x_min, x_max, y_min, y_max)
                
                metrics = calculate_metrics(xs_raw, ys_raw, t0, t1)
                results.append({
                    'lr': lr,
                    'iter': iterations,
                    'r2': metrics['r2'],
                    'mse': metrics['mse'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    't0': t0,
                    't1': t1
                })

                print(f"{lr:<6.2f} {iterations:<6d} {metrics['r2']:<10.6f} "
                      f"{metrics['mse']:<12.2f} {metrics['rmse']:<10.2f} {metrics['mae']:<10.2f}")

        # Normalize metrics for composite score
        r2_values = [r['r2'] for r in results]
        mse_values = [r['mse'] for r in results]
        rmse_values = [r['rmse'] for r in results]
        mae_values = [r['mae'] for r in results]
        
        r2_min, r2_max = min(r2_values), max(r2_values)
        mse_min, mse_max = min(mse_values), max(mse_values)
        rmse_min, rmse_max = min(rmse_values), max(rmse_values)
        mae_min, mae_max = min(mae_values), max(mae_values)
        
        for r in results:
            # Normalize R² (higher is better, already 0-1 scale)
            r2_norm = (r['r2'] - r2_min) / (r2_max - r2_min) if r2_max != r2_min else 1.0
            
            # Normalize and invert MSE, RMSE, MAE (lower is better)
            mse_norm = 1.0 - (r['mse'] - mse_min) / (mse_max - mse_min) if mse_max != mse_min else 1.0
            rmse_norm = 1.0 - (r['rmse'] - rmse_min) / (rmse_max - rmse_min) if rmse_max != rmse_min else 1.0
            mae_norm = 1.0 - (r['mae'] - mae_min) / (mae_max - mae_min) if mae_max != mae_min else 1.0
            
            # Composite score (R² weighted more heavily)
            r['composite_score'] = (0.5 * r2_norm + 0.2 * mse_norm + 0.15 * rmse_norm + 0.15 * mae_norm)
            
            if r['composite_score'] > best_score:
                best_score = r['composite_score']
                best_lr = r['lr']
                best_iter = r['iter']
                best_theta0 = r['t0']
                best_theta1 = r['t1']
                best_metrics = {
                    'r2': r['r2'],
                    'mse': r['mse'],
                    'rmse': r['rmse'],
                    'mae': r['mae']
                }

        print("-" * 80)
        print(f"\n✓ Best parameters found (composite score: {best_score:.4f}):")
        print(f"  Learning Rate: {best_lr}")
        print(f"  Iterations:    {best_iter}")
        print(f"\n  Metrics:")
        print(f"    R² Score:    {best_metrics['r2']:.6f}")
        print(f"    MSE:         {best_metrics['mse']:.2f}")
        print(f"    RMSE:        {best_metrics['rmse']:.2f}")
        print(f"    MAE:         {best_metrics['mae']:.2f}")
        print(f"\n  (Score weights: R²=50%, MSE=20%, RMSE=15%, MAE=15%)")

        # Save best thetas
        save_thetas(Path(THETAS_FILE), best_theta0, best_theta1)
        print(f"\n✓ Best thetas saved to {THETAS_FILE}")

        # Create visualization
        fig = plt.figure(figsize=(16, 5))
        
        # Composite Score Heatmap
        ax1 = plt.subplot(1, 3, 1)
        lr_unique = sorted(set(r['lr'] for r in results))
        iter_unique = sorted(set(r['iter'] for r in results))
        
        composite_grid = [[0.0 for _ in iter_unique] for _ in lr_unique]
        for r in results:
            i = lr_unique.index(r['lr'])
            j = iter_unique.index(r['iter'])
            composite_grid[i][j] = r['composite_score']

        im1 = ax1.imshow(composite_grid, cmap='RdYlGn', aspect='auto', origin='lower')
        ax1.set_xticks(range(len(iter_unique)))
        ax1.set_yticks(range(len(lr_unique)))
        ax1.set_xticklabels(iter_unique)
        ax1.set_yticklabels([f"{lr:.2f}" for lr in lr_unique])
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Composite Score Heatmap')
        plt.colorbar(im1, ax=ax1, label='Composite Score')

        # Mark best point
        best_i = lr_unique.index(best_lr)
        best_j = iter_unique.index(best_iter)
        ax1.plot(best_j, best_i, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2)

        # R² Score Heatmap
        ax2 = plt.subplot(1, 3, 2)
        r2_grid = [[0.0 for _ in iter_unique] for _ in lr_unique]
        for r in results:
            i = lr_unique.index(r['lr'])
            j = iter_unique.index(r['iter'])
            r2_grid[i][j] = r['r2']

        im2 = ax2.imshow(r2_grid, cmap='RdYlGn', aspect='auto', origin='lower')
        ax2.set_xticks(range(len(iter_unique)))
        ax2.set_yticks(range(len(lr_unique)))
        ax2.set_xticklabels(iter_unique)
        ax2.set_yticklabels([f"{lr:.2f}" for lr in lr_unique])
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('R² Score Heatmap')
        plt.colorbar(im2, ax=ax2, label='R² Score')

        # Line plot: Composite Score vs Iterations for each LR
        ax3 = plt.subplot(1, 3, 3)
        for lr in lr_unique:
            lr_results = [r for r in results if r['lr'] == lr]
            lr_results.sort(key=lambda x: x['iter'])
            iters = [r['iter'] for r in lr_results]
            scores = [r['composite_score'] for r in lr_results]
            ax3.plot(iters, scores, marker='o', label=f'LR={lr:.2f}')

        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Composite Score')
        ax3.set_title('Composite Score vs Iterations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
