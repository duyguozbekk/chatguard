"""Evaluate every model on the SAME held-out test split (OLID test, 860 tweets).

Reports macro-F1 (main metric, classes are imbalanced), accuracy, per-class F1, model size
(MB on disk) and CPU latency per single message (normalization + model), averaged over
N_RUNS >= 100 runs after warm-up. Writes results/results.json, results/results.md,
results/results.png and results/failures.json."""
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import load_splits
from src.predictors import load_all

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_RUNS, N_WARMUP = 200, 10
BUDGET_MS = 50.0


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif Path("/proc/cpuinfo").exists():
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return f"{cpu}, CPU only, {torch.get_num_threads()} threads, {platform.system()} {platform.release()}, torch {torch.__version__}"


def latency_ms(pred, msgs):
    for m in msgs[:N_WARMUP]:
        pred.predict_proba([m])
    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        pred.predict_proba([msgs[i % len(msgs)]])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    RES.mkdir(exist_ok=True)
    raw = load_splits(normalized=False)
    x_test, y_test = raw["test"]
    hw = hardware()
    print("[eval] hardware:", hw)
    preds_all, rows = {}, []
    for name, p in load_all().items():
        proba = p.predict_proba(x_test)
        pred = (proba >= 0.5).astype(int)
        preds_all[name] = (proba, pred)
        mean_ms, p95_ms = latency_ms(p, x_test)
        f1c = f1_score(y_test, pred, average=None)
        row = {"model": name, "precision": p.precision,
               "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
               "accuracy": round(float(accuracy_score(y_test, pred)), 4),
               "f1_clean": round(float(f1c[0]), 4), "f1_offensive": round(float(f1c[1]), 4),
               "size_mb": round(p.size_mb(), 2), "latency_ms_mean": round(mean_ms, 2),
               "latency_ms_p95": round(p95_ms, 2), "meets_budget": mean_ms <= BUDGET_MS,
               "confusion_matrix": confusion_matrix(y_test, pred).tolist()}
        rows.append(row)
        print(f"[eval] {name:30s} macroF1={row['macro_f1']:.4f}  size={row['size_mb']:.1f}MB  "
              f"latency={mean_ms:.2f}ms (p95 {p95_ms:.2f})")

    fp32, int8 = preds_all["BERT fine-tuned (FP32)"], preds_all["BERT quantized (INT8)"]
    agreement = float((fp32[1] == int8[1]).mean())
    summary = {"hardware": hw, "test_set": "OLID / tweet_eval-offensive test split",
               "n_test": len(y_test), "n_latency_runs": N_RUNS, "latency_budget_ms": BUDGET_MS,
               "fp32_int8_agreement": round(agreement, 4), "models": rows}
    json.dump(summary, open(RES / "results.json", "w"), indent=2)

    # markdown table (copied into README / slides)
    md = [f"Hardware: {hw}  \nTest set: {len(y_test)} tweets (same for all models). "
          f"Latency = mean over {N_RUNS} single-message runs.\n",
          "| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | p95 (ms) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['precision']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
                  f"{r['size_mb']:.1f} | {r['latency_ms_mean']:.2f} | {r['latency_ms_p95']:.2f} |")
    md.append(f"\nFP32 vs INT8 BERT prediction agreement on test: {agreement:.1%}")
    (RES / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # failure examples of the deployed (INT8) model
    proba, pred = int8
    fails = [{"text": x_test[i], "gold": ["clean", "offensive"][y_test[i]],
              "pred": ["clean", "offensive"][pred[i]], "p_offensive": round(float(proba[i]), 3)}
             for i in np.argsort(-np.abs(proba - np.array(y_test))) if pred[i] != y_test[i]][:15]
    json.dump(fails, open(RES / "failures.json", "w"), indent=2, ensure_ascii=False)

    # chart: macro-F1 and latency per model
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["model"].replace(" (", "\n(").replace(": ", ":\n") for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(names, [r["macro_f1"] for r in rows], color="#4C72B0")
    ax[0].set_title("Macro-F1 on test (higher is better)"); ax[0].set_ylim(0, 1)
    ax[1].bar(names, [r["latency_ms_mean"] for r in rows], color="#DD8452")
    ax[1].axhline(BUDGET_MS, ls="--", color="red", label=f"budget {BUDGET_MS:.0f} ms")
    ax[1].set_title("CPU latency per message, ms (lower is better)"); ax[1].legend()
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
        for c in a.containers:
            a.bar_label(c, fmt="%.3g", fontsize=8)
    plt.tight_layout(); plt.savefig(RES / "results.png", dpi=160)
    print("[eval] wrote results/results.json, results.md, results.png, failures.json")


if __name__ == "__main__":
    main()
