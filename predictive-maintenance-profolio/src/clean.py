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
    ★ 임계값을 데이터가 아니라 도메인에서 가져오는 게 핵심입니다.
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
    """진동값에 m/s²가 섞였는지 추정합니다.

    ★ 주의: 이건 온도만큼 확실하지 않습니다.
    27 mm/s는 물리적으로 불가능한 값이 아닙니다(고장난 설비면 나올 수 있음).
    그래서 '설비별 중앙값의 ratio배 이상'이라는 통계적 기준을 씁니다.
    메타데이터(태그 단위표)가 있으면 그걸 쓰는 게 항상 낫습니다.
    """
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
