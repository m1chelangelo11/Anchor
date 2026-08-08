import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR

baseline_means = np.load(RESULTS_DIR / "baseline_drift_means.npy")
direct_means = np.load(RESULTS_DIR / "direct_diff_means.npy")

plt.semilogy(range(1, len(baseline_means) + 1), baseline_means, label="Mean Drift")
plt.semilogy(range(1, len(direct_means) + 1), direct_means, label="Mean Difference")
plt.xlabel('Prediction step k')
plt.ylabel('Drift or Difference value')
plt.title('Comparison of one-step and direct predictors')
plt.legend()
plt.grid()
plt.savefig(RESULTS_DIR / "model_comparison_plot.png", dpi=150)
plt.show()
