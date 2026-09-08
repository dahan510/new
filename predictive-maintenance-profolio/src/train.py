from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F  # noqa: E402
from clean import run_pipeline  # noqa: E402

HORIZON = 30
COST_FN = 8_000_000
COST_FP = 300_000
SEED = 42


def load_history() -> pd.DataFrame:
    files = sorted((ROOT / "data" / "history").glob("*.csv"))
    if not files:
        raise SystemExit(
            "data/history/*.csv 가 없습니다. collector.py를 먼저 돌리세요."
        )
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main() -> int:
    raw = load_history()
    print(
        f"원본 {len(raw):,}행 ({len(list((ROOT / 'data' / 'history').glob('*.csv')))}개 파일)"
    )

    clean, log, rep = run_pipeline(raw, verbose=True)
    clean = clean[~clean["is_gap"].astype(bool)].copy()

    d = F.make_horizon_label(clean, horizon=HORIZON)
    d = F.build(d, windows=(10, 30, 60), shift_one=True)
    feat = [c for c in F.feature_columns(d) if c not in ("y", "machine_failure")]
    d = d.dropna(subset=feat + ["y"]).sort_values("ts").reset_index(drop=True)
    if len(d) < 500 or d["y"].nunique() < 2:
        raise SystemExit(
            f"학습에 쓸 데이터가 부족합니다({len(d)}행). 수집을 더 하세요."
        )

    X = d[feat].values
    y = d["y"].values.astype(int)
    cut = int(len(d) * 0.75)  # ★ 시간순 split
    print(
        f"\ntrain {cut:,} / test {len(d) - cut:,} | "
        f"train 양성률 {y[:cut].mean() * 100:.2f}% / test {y[cut:].mean() * 100:.2f}%"
    )

    mdl = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2
    )
    mdl.fit(X[:cut], y[:cut])
    p = mdl.predict_proba(X[cut:])[:, 1]
    yt = y[cut:]

    # 비용 최소 임계값
    ths = np.linspace(0.01, 0.99, 197)
    costs = []
    for t in ths:
        tn, fp, fn, tp = confusion_matrix(
            yt, (p >= t).astype(int), labels=[0, 1]
        ).ravel()
        costs.append(fn * COST_FN + fp * COST_FP)
    i = int(np.argmin(costs))
    th = float(ths[i])

    metrics = {
        "n_rows": int(len(d)),
        "n_features": len(feat),
        "horizon_min": HORIZON,
        "train_end": str(d["ts"].iloc[cut - 1]),
        "test_start": str(d["ts"].iloc[cut]),
        "positive_rate_test": round(float(yt.mean()), 4),
        "roc_auc": round(float(roc_auc_score(yt, p)), 4),
        "pr_auc": round(float(average_precision_score(yt, p)), 4),
        "threshold": round(th, 3),
        "precision": round(
            float(precision_score(yt, (p >= th).astype(int), zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(yt, (p >= th).astype(int), zero_division=0)), 4
        ),
        "f1": round(float(f1_score(yt, (p >= th).astype(int), zero_division=0)), 4),
        "cost_at_threshold": int(costs[i]),
        "cost_at_0.5": int(costs[int(np.argmin(np.abs(ths - 0.5)))]),
    }
    print("\n[성능]")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")

    imp = pd.Series(mdl.feature_importances_, index=feat).sort_values(ascending=False)
    print("\n[변수 중요도 상위 10]")
    print(imp.head(10).round(4).to_string())

    outdir = ROOT / "models"
    outdir.mkdir(exist_ok=True)
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    imp.head(30).to_csv(outdir / "feature_importance.csv", header=["importance"])
    pd.DataFrame(
        {
            "ts": d["ts"].iloc[cut:].values,
            "machine_id": d["machine_id"].iloc[cut:].values,
            "y": yt,
            "prob": p,
        }
    ).to_csv(outdir / "test_predictions.csv", index=False)
    print(f"\n저장: models/metrics.json, feature_importance.csv, test_predictions.csv")
    print("★ 모델 파일(.pkl)은 저장하지 않습니다 — sklearn 버전이 다르면 못 읽습니다.")
    print(
        "  대시보드는 예측 결과 CSV를 읽습니다. 재현은 이 스크립트를 다시 돌리면 됩니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
