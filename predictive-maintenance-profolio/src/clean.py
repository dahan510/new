from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
]

# 물리적으로 가능한 범위 (설비 스펙 + 상식)
PHYS_RANGE = {
    "air_temp_k": (270.0, 340.0),  # -3 ~ 67 도
    "process_temp_k": (280.0, 360.0),
    "rot_speed_rpm": (500.0, 4000.0),
    "torque_nm": (1.0, 100.0),
    "tool_wear_min": (0.0, 400.0),
    "vibration_mms": (0.05, 30.0),  # ISO 10816: 11 mm/s 초과면 위험
    "current_a": (0.1, 40.0),
    "humidity_pct": (0.0, 100.0),
}


# 정제 파이프라인을 거치면서 데이터가 몇 개씩 줄어드는지 기록
class StepLog:
    def __init__(self):
        self.rows = []

    def __call__(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        prev = self.rows[-1][1] if self.rows else len(df)
        self.rows.append((name, len(df), len(df) - prev))
        return df

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["단계", "행수", "증감"])


# 정제
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(
        df["ts"], errors="coerce"
    )  # "ts" 컬럼을 날짜/시간 형식으로 변환
    for c in SENSOR_COLS + ["machine_failure"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts", "machine_id"])


def snap_timestamp(df: pd.DataFrame, freq: str = "min") -> pd.DataFrame:
    """초 단위로 흔들린 타임스탬프를 분 격자에 붙입니다."""
    df = df.copy()
    df["ts"] = df["ts"].dt.round(freq)
    return df


def drop_dups(df: pd.DataFrame) -> pd.DataFrame:
    """(machine_id, ts) 중복 제거. 값이 다른 중복은 '나중 것'을 신뢰합니다."""
    return (
        df.sort_values("collected_at")
        .drop_duplicates(subset=["machine_id", "ts"], keep="last")
        .sort_values(["machine_id", "ts"])
        .reset_index(drop=True)
    )


# 단위 통일
def detect_and_fix_temp_unit(df: pd.DataFrame, cols=("air_temp_k", "process_temp_k")):
    """켈빈이어야 하는 컬럼에 섭씨가 섞였는지 판정합니다.

    판정 근거는 '물리적 불가능'입니다.
    공장 실내 온도가 200 K(-73도)일 수는 없습니다. 그러니 200 미만은 섭씨입니다.
    임계값을 데이터가 아니라 도메인에서 가져오는 게 핵심입니다.
    """
    df = df.copy()
    report = {}
    for c in cols:
        if c not in df.columns:
            continue
        mask = df[c].notna() & (df[c] < 200)
        report[c] = int(mask.sum())
        df.loc[mask, c] = df.loc[mask, c] + 273.15
    return df, report


def detect_vibration_unit(
    df: pd.DataFrame, col="vibration_mms", factor=9.81, ratio=4.0
):
    df = df.copy()
    med = df.groupby("machine_id")[col].transform("median")
    mask = df[col].notna() & (df[col] > med * ratio)
    df.loc[mask, col] = df.loc[mask, col] / factor
    return df, int(mask.sum())


# 스파이크 탐지
def hampel_flag(s: pd.Series, window: int = 11, n_sigma: float = 5.0) -> pd.Series:

    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad
    sigma = sigma.replace(0, np.nan)
    return ((s - med).abs() > n_sigma * sigma).fillna(False)


def flag_spikes(df: pd.DataFrame, cols=None, window=11, n_sigma=5.0):
    cols = cols or SENSOR_COLS
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[f"spike_{c}"] = df.groupby("machine_id")[c].transform(
            lambda s: hampel_flag(s, window, n_sigma)
        )
    spike_cols = [f"spike_{c}" for c in cols if f"spike_{c}" in df.columns]
    df["spike_any"] = df[spike_cols].any(axis=1)
    df["spike_count"] = df[spike_cols].sum(axis=1)
    return df


# 결측 보간
def interpolate_short_gaps(df: pd.DataFrame, cols=None, max_gap: int = 5):
    """max_gap분 이하의 짧은 구간만 시간 보간합니다.

    ★ 긴 끊김을 보간하면 '없던 데이터를 만들어내는' 것이 됩니다.
    30분 통신 두절 구간을 직선으로 채우면 모델은 그 30분을 '아주 안정적인 구간'으로
    배웁니다. 실제로는 아무 정보가 없는데도 말입니다.
    """
    cols = cols or SENSOR_COLS
    df = df.sort_values(["machine_id", "ts"]).copy()
    filled = {}
    for c in cols:
        if c not in df.columns:
            continue
        before = df[c].isna().sum()
        df[c] = df.groupby("machine_id")[c].transform(
            lambda s: s.interpolate(
                method="linear", limit=max_gap, limit_direction="both"
            )
        )
        filled[c] = int(before - df[c].isna().sum())
    return df, filled


# 드리프트 보정
def estimate_drift(df: pd.DataFrame, col="process_temp_k", ref="air_temp_k"):
    """설비별로 (col - ref)의 일별 중앙값이 시간에 따라 밀리는지 봅니다.

    같은 라인의 다른 설비를 기준선으로 씁니다.
    '설비 전체가 같이 오르면 공정 변화, 한 대만 오르면 센서 문제'라는 논리입니다.
    """
    d = df.dropna(subset=[col, ref]).copy()
    d["diff"] = d[col] - d[ref]
    d["day"] = (d["ts"] - d["ts"].min()).dt.total_seconds() / 86400.0
    daily = (
        d.groupby(["machine_id", d["day"].astype(int)])["diff"]
        .median()
        .rename("v")
        .reset_index()
        .rename(columns={"day": "d"})
    )
    fleet = daily.groupby("d")["v"].median().rename("fleet")
    daily = daily.join(fleet, on="d")
    daily["resid"] = daily["v"] - daily["fleet"]

    out = {}
    for m, g in daily.groupby("machine_id"):
        if len(g) < 3:
            out[m] = 0.0
            continue
        slope = np.polyfit(g["d"], g["resid"], 1)[0]
        out[m] = float(slope)
    return out, daily


def correct_drift(
    df: pd.DataFrame, slopes: dict, col="process_temp_k", min_slope: float = 0.05
):
    """추정된 기울기가 임계 이상인 설비만 보정합니다."""
    df = df.copy()
    t0 = df["ts"].min()
    days = (df["ts"] - t0).dt.total_seconds() / 86400.0
    applied = {}
    for m, s in slopes.items():
        if abs(s) < min_slope:
            applied[m] = 0.0
            continue
        mask = df["machine_id"] == m
        df.loc[mask, col] = df.loc[mask, col] - s * days[mask]
        applied[m] = s
    return df, applied
