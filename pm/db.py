from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sensors.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id      TEXT    NOT NULL,
    ts              TEXT    NOT NULL,          -- ISO8601 문자열 (UTC 기준)
    type            TEXT,
    air_temp_k      REAL,
    process_temp_k  REAL,
    rot_speed_rpm   REAL,
    torque_nm       REAL,
    tool_wear_min   REAL,
    vibration_mms   REAL,
    current_a       REAL,
    humidity_pct    REAL,
    machine_failure INTEGER,
    collected_at    TEXT,
    UNIQUE (machine_id, ts)                    -- ★ 중복 방어선
);

CREATE INDEX IF NOT EXISTS ix_sensor_ts      ON sensor_raw (ts);
CREATE INDEX IF NOT EXISTS ix_sensor_machine ON sensor_raw (machine_id, ts);

CREATE TABLE IF NOT EXISTS collect_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT,
    window_start  TEXT,
    window_end    TEXT,
    rows_received INTEGER,
    rows_inserted INTEGER,
    rows_skipped  INTEGER,
    note          TEXT
);
"""

COLUMNS = [
    "machine_id",
    "ts",
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
    "machine_failure",
    "collected_at",
]


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
    """(insert된 행 수, 중복으로 건너뛴 행 수)를 돌려줍니다."""
    df = df.reindex(columns=COLUMNS)
    before = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    sql = (
        f"INSERT OR IGNORE INTO sensor_raw ({','.join(COLUMNS)}) "
        f"VALUES ({','.join('?' * len(COLUMNS))})"
    )
    con.executemany(
        sql, df.where(pd.notna(df), None).itertuples(index=False, name=None)
    )
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    inserted = after - before
    return inserted, len(df) - inserted


def log_run(con, window_start, window_end, received, inserted, skipped, note=""):
    con.execute(
        "INSERT INTO collect_log (run_at, window_start, window_end,"
        " rows_received, rows_inserted, rows_skipped, note)"
        " VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        (str(window_start), str(window_end), received, inserted, skipped, note),
    )
    con.commit()


def read_all(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM sensor_raw ORDER BY ts, machine_id", con)
