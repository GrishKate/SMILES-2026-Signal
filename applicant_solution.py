import json
import gdown

import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def rank1_from_band_matrix(band_matrix, _eigh=np.linalg.eigh):
      bx_n = band_matrix / (np.sqrt(np.mean(np.abs(band_matrix) ** 2, axis=0, keepdims=True)) + 1e-30)
      cov = bx_n.conj().T @ bx_n / bx_n.shape[0]
      _, vecs = _eigh(cov)
      vec = vecs[:, -1]
      shared = band_matrix @ vec
      denom = np.vdot(shared, shared) + 1e-30
      return np.column_stack(
          [
              (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
              for ch in range(band_matrix.shape[1])
          ]
      )

def your_canceller(tx_n, rx):
    """Release placeholder: applicants should replace this with their own method."""
    e = 0
    for i in range(2):
      pred_fc = helpers["fit_tx_prediction"](rx - e)
      resid = rx - pred_fc
      e = rank1_from_band_matrix(resid)
    result = rx - pred_fc - e
    return result


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)