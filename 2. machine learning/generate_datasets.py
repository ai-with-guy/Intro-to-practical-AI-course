"""Generate the synthetic datasets used by machine_learning.ipynb."""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
OUTPUT_DIR = Path(__file__).parent


def save_2d(name, X, y):
    data = pd.DataFrame(X, columns=["x1", "x2"])
    data["class"] = y.astype(int)
    data.to_csv(OUTPUT_DIR / name, index=False)


def make_linear(rng):
    x0 = rng.multivariate_normal([-1.4, -1.0], [[0.55, 0.15], [0.15, 0.55]], 250)
    x1 = rng.multivariate_normal([1.4, 1.0], [[0.55, 0.15], [0.15, 0.55]], 250)
    return np.vstack([x0, x1]), np.r_[np.zeros(250), np.ones(250)]


def make_xor(rng):
    centres = [(-1.4, -1.4), (1.4, 1.4), (-1.4, 1.4), (1.4, -1.4)]
    points = [rng.normal(centre, 0.42, size=(180, 2)) for centre in centres]
    return np.vstack(points), np.repeat([0, 0, 1, 1], 180)


def make_spiral(rng):
    points_per_class = 500
    theta = np.linspace(0.25, 3.6 * np.pi, points_per_class)
    radius = np.linspace(0.15, 5.0, points_per_class)
    arm0 = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    arm1 = np.c_[radius * np.cos(theta + np.pi), radius * np.sin(theta + np.pi)]
    arm0 += rng.normal(0, 0.30, arm0.shape)
    arm1 += rng.normal(0, 0.30, arm1.shape)
    return np.vstack([arm0, arm1]), np.r_[np.zeros(points_per_class), np.ones(points_per_class)]


def regression_surface(x1, x2):
    radius_squared = x1**2 + x2**2
    surface = np.where(radius_squared <= 16.0, 0.35, 0.0)

    outline = (radius_squared >= 12.7) & (radius_squared <= 15.7)
    eyes = ((x1 + 1.35) ** 2 / 0.30 + (x2 - 1.25) ** 2 / 0.55 <= 1.0) | (
        (x1 - 1.35) ** 2 / 0.30 + (x2 - 1.25) ** 2 / 0.55 <= 1.0
    )
    nose = (np.abs(x1) <= 0.28) & (x2 >= -0.25) & (x2 <= 0.75)
    smile_curve = x2 + 1.45 - 0.22 * x1**2
    smile = (np.abs(smile_curve) <= 0.16) & (np.abs(x1) <= 2.35)

    surface = np.where(outline, 1.6, surface)
    surface = np.where(eyes, 2.8, surface)
    surface = np.where(nose, 1.2, surface)
    return np.where(smile, 2.3, surface)


def make_regression(rng):
    X = rng.uniform(-5, 5, size=(2500, 2))
    target = regression_surface(X[:, 0], X[:, 1])
    data = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "target": target})
    data.to_csv(OUTPUT_DIR / "data4_regression.csv", index=False)

    axis = np.linspace(-5, 5, 300)
    grid_x1, grid_x2 = np.meshgrid(axis, axis)
    grid = pd.DataFrame({"x1": grid_x1.ravel(), "x2": grid_x2.ravel()})
    grid["target"] = regression_surface(grid["x1"], grid["x2"])
    grid.to_csv(OUTPUT_DIR / "data4_regression_surface.csv", index=False)


def sample_imbalanced(rng, negatives, positives):
    negative = rng.multivariate_normal([0.0, 0.0], [[1.5, 0.25], [0.25, 1.5]], negatives)
    positive = rng.multivariate_normal([1.5, 1.3], [[1.2, 0.2], [0.2, 1.2]], positives)
    return np.vstack([negative, positive]), np.r_[np.zeros(negatives), np.ones(positives)]


def main():
    rng = np.random.default_rng(SEED)

    save_2d("data1_linear.csv", *make_linear(rng))
    save_2d("data2_nonlinear.csv", *make_xor(rng))
    save_2d("data3_spiral.csv", *make_spiral(rng))
    make_regression(rng)

    # Different class ratios are intentional: deployment may not match collected training data.
    save_2d("data5_imbalanced_train.csv", *sample_imbalanced(rng, 950, 50))
    save_2d("data5_imbalanced_test.csv", *sample_imbalanced(rng, 300, 300))

    print("Generated datasets in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
