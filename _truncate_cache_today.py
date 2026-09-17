# -*- coding: utf-8 -*-
"""清掉盘中缓存: 删除 data/cache 里 date==TODAY(2026-09-08) 的行(盘中半截bar),
重写缓存名为实际范围(末根=前一交易日), 让 update_daily_data 判定落后一天而去重拉今天的完整收盘bar。
停牌/未含今日行的文件不动。用法: ./.venv/Scripts/python _truncate_cache_today.py
"""
import os, sys, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime
from pathlib import Path
import pandas as pd

# 清理目标日 = 含盘中半截bar的当天; 默认今天, 可用 TODAY 环境变量覆盖。
TODAY = os.environ.get("TODAY", datetime.now().strftime("%Y-%m-%d"))
CACHE = Path("data/cache")
PAT = re.compile(r"^(\d{6})_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_qfq\.parquet$")

n_today = n_drop = n_bad = 0
for f in sorted(CACHE.glob("*.parquet")):
    m = PAT.match(f.name)
    if not m:
        continue
    try:
        df = pd.read_parquet(f)
    except Exception as e:
        print(f"  ! 读取失败 {f.name}: {e}", flush=True)
        n_bad += 1
        continue
    if "date" not in df.columns:
        continue
    df["date"] = pd.to_datetime(df["date"])
    mx = df["date"].max()
    if mx.strftime("%Y-%m-%d") != TODAY:
        continue                                   # 未含今日行(停牌等), 不动
    n_today += 1
    sub = df[df["date"] < pd.Timestamp(TODAY)]
    n_drop += int(len(df) - len(sub))
    if sub.empty:
        print(f"  ! {f.name} 只剩今日行, 跳过(异常)", flush=True)
        n_bad += 1
        continue
    s = sub["date"].min().strftime("%Y-%m-%d")
    e = sub["date"].max().strftime("%Y-%m-%d")
    new = CACHE / f"{m.group(1)}_{s}_{e}_qfq.parquet"
    sub.reset_index(drop=True).to_parquet(new, index=False)
    f.unlink()

print(f"[清理] 含{TODAY}行 {n_today} 只, 删除盘中行 {n_drop}, 异常 {n_bad}")
