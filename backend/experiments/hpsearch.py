"""Random hyperparameter search under scaffold CV, logged to JSONL.

Single-seed 5-fold during search (cheap ranking signal); winners are re-run on
3 seeds afterwards, because a single seed's spread (+/-0.005) is the same order
as the differences being chased.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from bench import build, run, Y, BASE

BLOCKS = sys.argv[1].split(",")
N_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LOG = Path(__file__).parent / f"hp_{sys.argv[3] if len(sys.argv) > 3 else 'run'}.jsonl"

# Hand-picked probes first: they test the axes most likely to be mis-set in the
# hand-tuned baseline (tree count, depth, column sampling) before random search.
PROBES = [
    {},                                                              # baseline
    dict(n_estimators=1500),
    dict(n_estimators=3000, learning_rate=0.015),
    dict(n_estimators=1500, max_depth=8),
    dict(n_estimators=1500, max_depth=4),
    dict(n_estimators=1500, colsample_bytree=0.2),
    dict(n_estimators=1500, colsample_bytree=0.8),
    dict(n_estimators=1500, min_child_weight=1),
]

rng = np.random.default_rng(11)
def sample():
    return dict(
        n_estimators=int(rng.choice([600, 1000, 1500, 2500, 4000])),
        learning_rate=float(rng.choice([0.01, 0.015, 0.02, 0.03, 0.05])),
        max_depth=int(rng.integers(4, 11)),
        min_child_weight=float(rng.choice([1, 2, 3, 5, 8, 12])),
        subsample=float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        colsample_bytree=float(rng.choice([0.15, 0.25, 0.4, 0.6, 0.8])),
        colsample_bynode=float(rng.choice([0.5, 0.8, 1.0])),
        reg_lambda=float(rng.choice([0.5, 1, 2, 5, 10, 25])),
        reg_alpha=float(rng.choice([0, 0.1, 0.5, 2, 5])),
        gamma=float(rng.choice([0, 0, 0.05, 0.2])),
    )

configs = PROBES + [sample() for _ in range(N_ITER)]
best = -9e9
with LOG.open("w") as fh:
    for i, cfg in enumerate(configs):
        t = time.time()
        agg, d = run(BLOCKS, params=cfg, seeds=(0,), verbose=False)
        r2 = agg["r2"][0]
        best = max(best, r2)
        fh.write(json.dumps({"i": i, "r2": r2, "rmse": agg["rmse"][0],
                             "spearman": agg["spearman"][0], "d": d, "cfg": cfg}) + "\n")
        fh.flush()
        tag = " *BEST*" if r2 >= best else ""
        print(f"[{i:3d}] R2={r2:.4f} rmse={agg['rmse'][0]:.4f} ({time.time()-t:.0f}s){tag} {cfg}",
              flush=True)
print("done")
