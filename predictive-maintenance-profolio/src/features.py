from __future__ import annotations

import numpy as np
import pandas as pd

BASE = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
]


def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    """도메인 지식으로 만드는 파생변수. 모델보다 이게 성능을 더 올립니다."""
    df = df.copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]  # 방열 여력
    df["power_w"] = df["torque_nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60
    df["wear_torque"] = df["tool_wear_min"] * df["torque_nm"]  # 과부하 지표(OSF)
    df["vib_per_rpm"] = df["vibration_mms"] / df["rot_speed_rpm"].replace(0, np.nan)
    return df


def add_rolling(
    df: pd.DataFrame, cols=None, windows=(10, 30, 60), shift_one: bool = True
) -> pd.DataFrame:
    """설비별 과거 window분 통계.

    shift_one=True면 현재 시점 값을 제외합니다(t-1까지만 사용).
    실시간 예측에서 '현재 값이 이미 도착했는지'가 불확실할 때 안전한 선택입니다.
    """
    cols = cols or BASE + ["temp_diff_k", "power_w", "wear_torque"]
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values(["machine_id", "ts"]).copy()
    g = df.groupby("machine_id")

    new = {}
    for c in cols:
        s = g[c]
        for w in windows:
            base = s.shift(1) if shift_one else df[c]
            r = base.groupby(df["machine_id"]).rolling(w, min_periods=max(2, w // 3))
            new[f"{c}_m{w}"] = r.mean().reset_index(level=0, drop=True)
            new[f"{c}_s{w}"] = r.std().reset_index(level=0, drop=True)
        # 변화량 (기울기 대용)
        new[f"{c}_d1"] = s.diff()
        new[f"{c}_d10"] = s.diff(10)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def make_horizon_label(
    df: pd.DataFrame, horizon: int = 30, src: str = "machine_failure", out: str = "y"
) -> pd.DataFrame:
    """★ 예지보전의 진짜 목표변수: "앞으로 horizon분 안에 고장이 나는가"

    이걸 안 하고 '지금 고장인가'를 맞히면 그건 예지(prediction)가 아니라
    감지(detection)입니다. 이미 고장난 뒤에 알려주는 모델은 쓸모가 없습니다.

    구현: 미래 구간의 최댓값을 가져오되, 현재 시점은 제외(shift(-1))합니다.
    """
    df = df.sort_values(["machine_id", "ts"]).copy()

    def fwd(s: pd.Series) -> pd.Series:
        # 뒤집어서 rolling → 다시 뒤집으면 '미래 창'이 됩니다
        return s[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)

    df[out] = df.groupby("machine_id")[src].transform(fwd)
    return df


def build(df: pd.DataFrame, windows=(10, 30, 60), shift_one=True) -> pd.DataFrame:
    return add_rolling(add_physics(df), windows=windows, shift_one=shift_one)


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        "ts",
        "machine_id",
        "type",
        "machine_failure",
        "collected_at",
        "id",
        "is_gap",
        "spike_any",
        "spike_count",
    }
    return [
        c
        for c in df.columns
        if c not in drop
        and not c.startswith("spike_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
