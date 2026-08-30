"""Estimate a susceptibility peak over graph size and edge probability."""

import numpy as np

from phaseflow.analysis import estimate_peak, run_percolation_sweep, summarize_sweep


sizes = [250, 500, 1000]
probabilities = np.linspace(0.5 / max(sizes), 1.8 / min(sizes), 12)
result = run_percolation_sweep(sizes, probabilities, trials=20, seed=11)

for row in summarize_sweep(result, "susceptibility"):
    print(
        f"n={int(row['size']):4d} p={row['control']:.6f} "
        f"susceptibility={row['mean']:.4f} ± {row['standard_error']:.4f}"
    )

print("largest-size peak:", estimate_peak(result))

