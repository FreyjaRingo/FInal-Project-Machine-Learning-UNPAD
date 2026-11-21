"""Run only the missing thresholds-only runs for Table 4 (XGBoost and MLP, binary relevance).
Run from the repository root on the GPU laptop:
    python runner/run_pending_thr.py
It loads the functions of the Structured notebook unchanged, runs
    {xgb, mlp} x br x {none, weight, mlsmote} x thr x 5 folds  (30 runs)
and appends the rows to results/experiments.csv and the scores to results/preds/,
exactly as the notebook would. Finished runs are skipped, so it can be restarted.
"""
import json, sys, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Indonesian_Multilabel_Hate_Speech_Detection_Structured.ipynb"
import os
os.chdir(ROOT)
sys.modules.setdefault("IPython.display", types.SimpleNamespace(display=print))
nb = json.load(open(NB, encoding="utf-8"))
g = {"__name__": "nb"}
for i in [4, 7, 11] + list(range(14, 200)):
    cell = nb["cells"][i]
    if cell["cell_type"] != "code":
        continue
    if i == 11:
        g["ROOT"], g["RES"] = ROOT, ROOT / "results"
    exec(compile("".join(cell["source"]), f"cell{i}", "exec"), g)

if g["DEVICE"] != "cuda":
    sys.exit("CUDA tidak terdeteksi. Jalankan di laptop dengan GPU supaya hasilnya sebanding dengan run XGBoost/MLP lain.")

data, folds = g["load_inputs"](ROOT)
out = ROOT / "results" / "experiments.csv"
done = g["completed_runs"](out)
grid = [(a, "br", c, "thr", "word", f) for a in ["xgb", "mlp"] for c in ["none", "weight", "mlsmote"] for f in range(5)]
todo = [x for x in grid if (g["config_id"](x[0], x[1], x[2], x[3]), x[5]) not in done]
print(f"{len(todo)} dari {len(grid)} run perlu dijalankan di {g['DEVICE_LABEL']}")
for algo, st, cond, tun, feat, fold in todo:
    t = time.time()
    row = g["run_experiment"](ROOT, algo, st, cond, tun, feat, fold, data, folds)
    g["append_rows"](out, [row])
    print(f"{row['config']:<28} f{fold}  macro-F1={row['macro_f1']:.4f}  minority={row['minority_macro_f1']:.4f}  {time.time()-t:.0f}s", flush=True)

import pandas as pd
r = pd.read_csv(out)
r = r[r.config.isin([g["config_id"](a, "br", c, "thr") for a in ["xgb", "mlp"] for c in ["none", "weight", "mlsmote"]])]
print("\nRingkasan untuk Tabel 4 (kolom Thresholds only):")
print(r.groupby("config")[["macro_f1", "minority_macro_f1"]].mean().round(3).to_string())
