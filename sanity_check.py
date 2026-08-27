"""One-off visual check: does the processed GWL series look physically real?"""

import matplotlib.pyplot as plt
import pandas as pd

from src import config

well_id = config.WELL_IDS[0][0]
df = pd.read_csv(config.DATA_PROCESSED_DIR / f"{well_id}_features.csv", parse_dates=["date"])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(df["date"], df["gwl"])
axes[0].set_ylabel("Depth to water (ft)")
axes[0].set_title(f"Monthly GWL — {well_id}")
axes[0].invert_yaxis()  # deeper water table = lower on the page, more intuitive

axes[1].bar(df["date"], df["rainfall_mm"], width=20)
axes[1].set_ylabel("Rainfall (mm)")

plt.tight_layout()
out_path = config.PROJECT_ROOT / "sanity_check_plot.png"
plt.savefig(out_path)
print(f"Saved plot to {out_path}")