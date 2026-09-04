from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db as dbmod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"


def main() -> int:
    files = sorted(HIST.glob("*.csv"))
    if not files:
        print(f"[WARN] {HIST} 에 CSV가 없습니다. 먼저 collector.py를 돌리세요.")
        return 1

    dbp = dbmod.DB_PATH
    if dbp.exists():
        dbp.unlink()
        print("기존 DB 삭제 후 재생성합니다.")

    con = dbmod.connect(dbp)
    total_in = total_ins = 0
    for f in files:
        df = pd.read_csv(f)
        ins, skip = dbmod.upsert(con, df)
        total_in += len(df)
        total_ins += ins
        print(f"  {f.name:<20} 읽음 {len(df):>6,} / 신규 {ins:>6,} / 중복 {skip:>5,}")
    n = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    con.close()
    print(f"\nCSV {len(files)}개 / 읽은 행 {total_in:,} / DB {n:,}행")
    print(f"차이 {total_in - n:,}행은 (machine_id, ts) 중복입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
