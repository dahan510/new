from __future__ import annotations

import numpy as np
import pandas as pd

# 설비 스펙
MACHINES = {
    # machine_id : (품질등급, 마모한계 계수, 공구교체주기(분))
    "CNC-01": {"type": "L", "osf_limit": 11000, "tool_life": 210},
    "CNC-02": {"type": "M", "osf_limit": 12000, "tool_life": 225},
    "CNC-03": {"type": "H", "osf_limit": 13000, "tool_life": 240},
}

# 관측 오염 강도 (기본값 = "현장급")
POLLUTION = {
    "dropout_rate": 0.015,  # 통신 끊김으로 통째로 사라지는 구간 발생 확률
    "dropout_len": (3, 40),  # 끊김 길이(분)
    "nan_rate": 0.008,  # 개별 센서값만 NaN
    "spike_rate": 0.004,  # 센서 튐(전기 노이즈)
    "dup_rate": 0.006,  # 같은 레코드 중복 전송
    "ts_jitter_rate": 0.05,  # 타임스탬프 흔들림
    "unit_mix_rate": 0.10,  # 단위 혼재(K 대신 섭씨로 오는 구간)
    "drift_per_day": 0.35,  # 온도 센서 드리프트 (K/day)
}


# 물리 기반 참값 생성
def _simulate_one(
    machine_id: str, n_minutes: int, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    spec = MACHINES[machine_id]
    ts = pd.date_range(start, periods=n_minutes, freq="min")

    # --- 공정 부하: 근무 시간대에 높고 야간에 낮음 (일주기) ---
    hour = ts.hour + ts.minute / 60.0
    duty = 0.55 + 0.45 * np.sin((hour - 6) / 24 * 2 * np.pi)  # 0.1 ~ 1.0
    duty = np.clip(duty + rng.normal(0, 0.05, n_minutes), 0.05, 1.0)

    # --- 공기 온도: 계절/일교차 + 랜덤워크 ---
    air = 298.0 + 2.0 * np.sin((hour - 14) / 24 * 2 * np.pi)
    air = air + np.cumsum(rng.normal(0, 0.02, n_minutes))  # 완만한 표류
    air = air + rng.normal(0, 0.15, n_minutes)

    # --- 공구 마모: 누적되다가 교체하면 0으로 ---
    tool_life = spec["tool_life"]
    wear_rate = 1.0 + 0.6 * duty  # 부하 클수록 빨리 닳음
    wear = np.zeros(n_minutes)
    acc = rng.uniform(0, 60)  # 시작 시점 마모도는 랜덤
    limit = tool_life * rng.uniform(0.90, 1.15)
    for i in range(n_minutes):
        acc += wear_rate[i]
        if acc > limit:  # 계획 교체 (정비반 재량으로 조금씩 다름)
            acc = 0.0
            limit = tool_life * rng.uniform(0.90, 1.15)
        wear[i] = acc

    # --- 회전수: 부하에 반비례(무거운 절삭일수록 저속) ---
    rpm = 2860 - 1500 * duty + rng.normal(0, 45, n_minutes)
    rpm = np.clip(rpm, 1150, 2900)

    # --- 토크: 부하에 비례, 마모되면 저항 증가 ---
    torque = 10 + 40 * duty + 0.02 * wear + rng.normal(0, 2.0, n_minutes)
    torque = np.clip(torque, 3.0, 80.0)

    # --- 냉각(HVAC) 이상: 가끔 공장 공조가 죽어 실내가 더워짐 ---
    hvac_fail = np.zeros(n_minutes, dtype=bool)
    for _ in range(max(1, n_minutes // 2000)):
        s = rng.integers(0, max(1, n_minutes - 120))
        hvac_fail[s : s + rng.integers(40, 120)] = True
    air = air + 5.5 * hvac_fail  # 실내 온도 상승

    # --- 공정 온도: 공기온도 + 절삭열. 쿨런트가 process 쪽은 어느 정도 잡아줌 ---
    power_w = torque * rpm * 2 * np.pi / 60.0  # [W]
    proc = air + 8.5 + power_w / 1400.0 + 0.004 * wear
    proc = proc - 6.0 * hvac_fail  # 온도차(방열 여력)가 줄어듦
    proc = proc + rng.normal(0, 0.12, n_minutes)

    # --- 진동: 마모·회전수에 비례. 마모 후반에 급격히 커짐 ---
    vib = (
        0.8
        + 0.0009 * rpm
        + 0.9 * (wear / tool_life) ** 3
        + rng.normal(0, 0.06, n_minutes)
    )
    vib = np.clip(vib, 0.1, None)

    # --- 전류: 전력/전압(380V, 역률 0.85, 3상) ---
    current = power_w / (380 * 1.732 * 0.85) + rng.normal(0, 0.15, n_minutes)
    current = np.clip(current, 0.2, None)

    # --- 습도: 온도와 약한 음의 관계 ---
    humid = 55 - 1.8 * (air - 298) + rng.normal(0, 2.5, n_minutes)
    humid = np.clip(humid, 15, 95)

    df = pd.DataFrame(
        {
            "ts": ts,
            "machine_id": machine_id,
            "type": spec["type"],
            "air_temp_k": air,
            "process_temp_k": proc,
            "rot_speed_rpm": rpm,
            "torque_nm": torque,
            "tool_wear_min": wear,
            "vibration_mms": vib,
            "current_a": current,
            "humidity_pct": humid,
        }
    )

    # 고장 라벨 (AI4I 2020 정의)
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n_minutes) < 0.004)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power_w < 3500) | (power_w > 9000)
    osf = (wear * torque) > spec["osf_limit"]
    rnf = rng.random(n_minutes) < 0.0002  # 원인 불명 랜덤 고장

    df["twf"] = twf.astype(int)
    df["hdf"] = hdf.astype(int)
    df["pwf"] = pwf.astype(int)
    df["osf"] = osf.astype(int)
    df["rnf"] = rnf.astype(int)
    df["machine_failure"] = (twf | hdf | pwf | osf | rnf).astype(int)
    df["power_w"] = power_w
    return df


def simulate_truth(
    n_minutes: int = 1440, start: str | pd.Timestamp = "2024-01-01", seed: int = 42
) -> pd.DataFrame:
    """오염 없는 참값을 생성합니다."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    parts = [_simulate_one(m, n_minutes, start, rng) for m in MACHINES]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ts", "machine_id"]).reset_index(drop=True)


truth = simulate_truth(n_minutes=1440 * 14, start="2024-01-01", seed=42)
print("설비 수 :", truth["machine_id"].nunique())
print("기간    :", truth["ts"].min(), "~", truth["ts"].max())
print("행 수   :", f"{len(truth):,}")

# 오염 주입
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

print(truth[SENSOR_COLS].describe().loc[["mean", "std", "min", "50%", "max"]].round(2))


# 6가지 오염
def pollute(
    truth: pd.DataFrame,  # 정상적인 센서 데이터
    seed: int = 7,  # 랜덤 데이터를 만들 때 사용할 기준값
    cfg: dict | None = None,  # 오염 설정값을 추가로 받을 수 있음
    return_masks: bool = False,  # 오염된 위치를 표시한 mask도 반환할지 결정
):
    c = dict(POLLUTION)  # 기본 오염 설정을 복사해서 c에 저장
    if cfg:
        c.update(cfg)
    rng = np.random.default_rng(seed)
    df = truth.copy()
    masks = pd.DataFrame(index=df.index)

    # 관측 데이터에는 참값 라벨 중 machine_failure만 남깁니다.
    # 즉, 실제 센서가 관측할 수 없는 '정답 정보'를 없애는 것
    df = df.drop(columns=["twf", "hdf", "pwf", "osf", "rnf", "power_w"])

    # 전체 데이터에서 가장 이른 시간(첫 번째 시간)을 기준점으로 설정
    t0 = df["ts"].min()
    days = (df["ts"] - t0).dt.total_seconds() / 86400.0

    # --- (a) 센서 드리프트: CNC-02 온도 센서만 서서히 밀림 ---
    m2 = df["machine_id"] == "CNC-02"
    df.loc[m2, "process_temp_k"] += c["drift_per_day"] * days[m2]

    # --- (b) 단위 혼재: 특정 구간에서 온도가 섭씨로 들어옴 ---
    n = len(df)
    unit_block = np.zeros(n, dtype=bool)
    n_blocks = max(1, int(n * c["unit_mix_rate"] / 200))
    for _ in range(n_blocks):
        s = rng.integers(0, n - 200)
        unit_block[s : s + 200] = True
    df.loc[unit_block, "air_temp_k"] -= 273.15
    df.loc[unit_block, "process_temp_k"] -= 273.15
    masks["unit_temp"] = unit_block
    # 진동 단위도 일부는 m/s^2 로 (×9.81)
    vib_block = rng.random(n) < 0.04
    df.loc[vib_block, "vibration_mms"] *= 9.81
    masks["unit_vib"] = vib_block

    # --- (c) 센서 튐: 값이 순간적으로 10~50배 또는 0 ---
    for col in SENSOR_COLS:
        hit = rng.random(n) < c["spike_rate"]
        mode = rng.random(n)
        df.loc[hit & (mode < 0.5), col] = df.loc[hit & (mode < 0.5), col] * rng.uniform(
            8, 40
        )
        df.loc[hit & (mode >= 0.5), col] = 0.0
        masks[f"spike_{col}"] = hit

    # --- (d) 개별 결측 ---
    for col in SENSOR_COLS:
        hit = rng.random(n) < c["nan_rate"]
        df.loc[hit, col] = np.nan
        masks[f"nan_{col}"] = hit

    # --- (e) 통신 끊김: 행 자체가 사라짐 ---
    drop_mask = np.zeros(n, dtype=bool)
    n_drop = int(n * c["dropout_rate"] / 10)
    for _ in range(max(1, n_drop)):
        s = rng.integers(0, n)
        ln = rng.integers(*c["dropout_len"]) * 3  # 설비 3대 × 분
        drop_mask[s : s + ln] = True
    masks["dropped"] = drop_mask
    keep = ~drop_mask
    df = df[keep].copy()
    kept_masks = masks[keep].copy()

    # --- (f) 중복 전송 ---
    n2 = len(df)
    dup_idx = rng.random(n2) < c["dup_rate"]
    dups = df[dup_idx].copy()
    kept_masks["is_dup"] = False
    dup_masks = kept_masks[dup_idx].copy()
    dup_masks["is_dup"] = True
    df = pd.concat([df, dups], ignore_index=True)
    kept_masks = pd.concat([kept_masks, dup_masks], ignore_index=True)

    # --- (g) 타임스탬프 흔들림 + 순서 뒤섞임 ---
    n3 = len(df)
    jitter = np.where(
        rng.random(n3) < c["ts_jitter_rate"], rng.integers(-90, 90, n3), 0
    )
    df["ts"] = df["ts"] + pd.to_timedelta(jitter, unit="s")
    kept_masks["ts_jittered"] = jitter != 0
    order = rng.permutation(n3)
    df = df.iloc[order].reset_index(drop=True)
    kept_masks = kept_masks.iloc[order].reset_index(drop=True)

    # --- (h) 실제 수집기가 붙이는 메타 컬럼 ---
    df["collected_at"] = pd.Timestamp("2024-01-01")
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")  # 문자열로 들어옴(현실)
    if return_masks:
        return df, kept_masks
    return df


obs, masks = pollute(truth, seed=7, return_masks=True)
print("참값 행수 :", f"{len(truth):,}")
print("관측 행수 :", f"{len(obs):,}", f"({len(obs) - len(truth):+,})")
# print(obs.head(3).to_string())
print(obs["air_temp_k"].describe().round(2).to_string())
print("200 K 미만 비율: %.2f%%" % ((obs["air_temp_k"] < 200).mean() * 100))
inj = pd.DataFrame({"건수": masks.sum(), "비율(%)": (masks.mean() * 100).round(3)})
print(inj.to_string())


# 실시간 수집용
def sample_window(
    n_minutes: int = 60,  # 몇 분 동안의 데이터를 가져올지
    end: pd.Timestamp | None = None,  # 데이터의 마지막 시간
    seed: int | None = None,  # 랜덤값을 만들 때 사용할 기준값
) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().floor("min") if end is None else pd.Timestamp(end)
    start = end - pd.Timedelta(minutes=n_minutes)
    # 시드를 날짜에서 뽑으면 같은 날 다시 돌려도 같은 값이 나옵니다(재현성)
    if seed is None:
        seed = int(start.strftime("%Y%m%d%H"))
    truth = simulate_truth(n_minutes=n_minutes, start=start, seed=seed)
    obs = pollute(truth, seed=seed + 1)
    obs["collected_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return obs


if __name__ == "__main__":
    t = simulate_truth(n_minutes=1440, start="2024-01-01", seed=42)
    o = pollute(t, seed=7)
    print("truth   :", t.shape)
    print("observed:", o.shape)
    print(o.head(3).to_string())
