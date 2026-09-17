# -*- coding: utf-8 -*-
"""严格三买(递补严格一买) 近N日信号接口。

口径: 严格(canonical) 线段级, build_structure + scan_points (与 K=2 回测同源)。
窗口: 近 N 个交易日收盘买点 (exec ∈ [n-N, n-1]); exec==n 记为"明日执行"。
排序: 收跌优先 → 688/300 > 00 > 60 → 近者优先。
买入过滤: 涨停(300/688 为 20% 板, 其余 10% 板) / 跌超 5% 不买。
递补: 若近 N 日无三买, 用严格一买递补。

用法:
  from chan_signal_api import recent_chan_signals
  df, kind = recent_chan_signals(days=5)          # 读缓存, 无缓存则扫
  df, kind = recent_chan_signals(days=5, refresh=True)   # 强制重扫全市场(~5-10min)

  from chan_signal_api import recent_chan_signals_both
  df3, df1 = recent_chan_signals_both(days=5)     # 三买/一买各自独立, 不递补
  df 列: code,name,kind,buy_date,close,entry_ret,today_ret,bucket,board,is_down,
         zg_ext,depth,gap,ratio
         entry_ret=买点触发那根的涨跌(仅用于排序和"收跌/收涨"列),
         today_ret=今日涨跌(最新一根 bar 相对前一交易日, 页面展示用)
"""
import sys
import os
import time
import pickle
from pathlib import Path
from datetime import datetime

# exe/ 目录下运行时, 把项目根目录加入 sys.path 以便导入 bt_chan_backtest / myquant
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from bt_chan_backtest import build_cache_map, load_stock, build_structure
from myquant.strategy.chan_common import DEFAULT_PARAMS, scan_points
from realtime_snapshot import fetch_realtime_bars, append_today_bar

CACHE = Path("reports/chan_recent_signals.pkl")

# 缓存结构版本。改 _to_df 的列(增删/改名)时必须 +1, 否则当天的旧 pkl 会被当成
# 有效缓存读出来, 少一列直接报错。
CACHE_SCHEMA = 2

NAMES = {}
try:
    _nf = pd.read_csv("data/all_stocks.csv", dtype={"代码": str})
    NAMES = dict(zip(_nf["代码"], _nf["名称"]))
except Exception:
    pass


def board_prio(code):
    if code.startswith("688") or code.startswith("300"):
        return 0
    if code.startswith("00"):
        return 1
    if code.startswith("60"):
        return 2
    return 3


def board_label(code):
    return {0: "688/300", 1: "00主板", 2: "60主板"}.get(board_prio(code), "其它")


def limit_pct(code):
    return 0.20 if (code.startswith("300") or code.startswith("688")) else 0.10


def prev_down(segs, j):
    k = j - 1
    while k >= 0 and segs[k].d >= 0:
        k -= 1
    return k


def scan_all(days):
    """全市场扫描近 N 日严格 buy3/buy1, 返回原始记录列表。"""
    params = dict(DEFAULT_PARAMS)
    cm = build_cache_map()
    codes = sorted(cm)
    live = fetch_realtime_bars(codes)   # 当日盘中实时 bar
    n_live = 0
    recs = []
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            df = load_stock(cm[code])
        except Exception:
            continue
        if len(df) < 110:
            continue
        lv = live.get(code)
        if lv:
            df = append_today_bar(df, lv)
            n_live += 1
        n = len(df)
        dates = df["date"]
        close = df["close"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        # 今日涨跌 = 最新一根 bar（已含上面追加的当日 bar）相对前一交易日的涨跌幅。
        # 与 entry_ret 不同: entry_ret 是买点触发那根的涨跌, 「明日可买」时恒为 0。
        day_ret = float(close[n - 1] / close[n - 2] - 1) if n >= 2 and close[n - 2] > 0 else 0.0
        segs, _ = build_structure(df, "canonical")
        if len(segs) < 8:
            continue
        buys, _ = scan_points(segs, params)
        lo_ex = n - days
        for e in buys:
            ex = e["exec"]
            if ex is None or ex < lo_ex or ex > n:
                continue
            if ex >= 1 and close[ex - 1] > 0:
                er = float(close[ex] / close[ex - 1] - 1) if ex < n else 0.0
            else:
                er = 0.0
            lp = limit_pct(code)
            if er >= lp - 0.001 or er < -0.05:
                continue          # 涨停 / 跌超5% 不买
            if ex == n:
                bucket, bd, ds = "明日可买", "下一交易日", -1
                px = float(close[n - 1])
            elif ex == n - 1:
                bucket, bd, ds = "今日", str(dates[ex].date()), int(dates[ex].strftime("%Y%m%d"))
                px = float(close[ex])
            else:
                bucket, bd, ds = "近5日", str(dates[ex].date()), int(dates[ex].strftime("%Y%m%d"))
                px = float(close[ex])
            rec = dict(code=code, name=str(NAMES.get(code, "")).replace(" ", ""),
                       kind=e["kind"], buy_date=bd, date_sort=ds, close=round(px, 2),
                       entry_ret=round(er * 100, 2), today_ret=round(day_ret * 100, 2),
                       bucket=bucket,
                       board=board_label(code), board_prio=board_prio(code),
                       is_down=int(er < 0))
            if e["kind"] == "buy3":
                s = segs[e["seg"]]
                zg = float(e["zg"])
                bh = float(e["bh"]) if e.get("bh") is not None else None
                rec["zg_ext"] = round((px / zg - 1) * 100, 1)
                rec["depth"] = round((bh - s.lo) / bh * 100, 1) if bh else None
                lo0, lo1 = min(s.sbar, s.ebar), max(s.sbar, s.ebar)
                low_bar = lo0 + int(np.argmin(low[lo0:lo1 + 1]))
                rec["gap"] = int(s.conf - low_bar) if s.conf is not None else None
            elif e["kind"] == "buy1":
                s = segs[e["seg"]]
                pd_ = prev_down(segs, e["seg"])
                pds = segs[pd_] if pd_ >= 0 else None
                rec["ratio"] = round(float(s.area / pds.area), 2) \
                    if pds is not None and pds.area > 0 else None
            recs.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  [扫描] {i+1}/{len(codes)} 命中 {len(recs)}  {time.time()-t0:.0f}s",
                  flush=True)
    if n_live:
        print(f"  [实时] 已追加 {n_live} 只当日盘中 bar", flush=True)
    return recs


def sort_recs(recs):
    """收跌优先(is_down=1 排前) → 板块(0..3) → 近者优先(今日>近5日>明日可买, 日期降序)。"""
    def key(r):
        return (-r["is_down"], r["board_prio"], -r["date_sort"])
    return sorted(recs, key=key)


def check_freshness():
    """检查日线缓存最后一根 bar 是否最新。返回 (last_date_str, 提示)。"""
    import glob
    today = datetime.now().date()
    fs = glob.glob("data/cache/*_qfq.parquet") or glob.glob("data/cache/*.parquet")
    last = None
    for f in fs[:8]:
        try:
            d = pd.read_parquet(f, columns=["date"])["date"].max().date()
            last = d if last is None else max(last, d)
        except Exception:
            pass
    if last is None:
        return None, "⚠️ 未找到日线缓存"
    diff = (today - last).days
    if diff <= 0:
        return str(last), "✓ 数据已更新至今日"
    elif diff <= 3:   # 周末/假期/盘中
        return str(last), f"✓ 日线截至 {last}；扫描时已自动追加今日盘中实时 bar"
    else:
        return str(last), f"⚠️ 数据过期：最后 {last}，已差 {diff} 天，请先 update_daily_data.py"


def _to_df(recs):
    cols = ["code", "name", "kind", "buy_date", "close", "entry_ret", "today_ret",
            "bucket", "board", "is_down", "zg_ext", "depth", "gap", "ratio"]
    df = pd.DataFrame(recs)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["kind_label"] = df["kind"].map({"buy3": "三买", "buy1": "一买"})
    df["is_down_label"] = df["is_down"].map({0: "收涨", 1: "收跌"})
    return df[["code", "name", "kind_label", "buy_date", "close", "entry_ret",
               "today_ret", "is_down_label", "bucket", "board",
               "zg_ext", "depth", "gap", "ratio"]]


def _scan_and_cache(days, today):
    """扫描并写缓存, 返回 (df3, df1) —— 严格三买与严格一买各自独立, 不递补。"""
    recs = scan_all(days)
    buy3 = [r for r in recs if r["kind"] == "buy3"]
    buy1 = [r for r in recs if r["kind"] == "buy1"]
    df3 = _to_df(sort_recs(buy3))
    df1 = _to_df(sort_recs(buy1))
    CACHE.parent.mkdir(parents=True, exist_ok=True)   # reports/ 不进版本库, 首次运行可能不存在
    with open(CACHE, "wb") as f:
        pickle.dump({"schema": CACHE_SCHEMA, "days": days, "date": today,
                     "df3": df3, "df1": df1}, f)
    return df3, df1


def _load_both_cache(days, today):
    if CACHE.exists():
        try:
            with open(CACHE, "rb") as f:
                c = pickle.load(f)
            if (c.get("schema") == CACHE_SCHEMA and c.get("days") == days
                    and c.get("date") == today and "df3" in c
                    and "today_ret" in getattr(c["df3"], "columns", [])):
                return c["df3"], c["df1"]
        except Exception:
            pass
    return None


def recent_chan_signals_both(days=5, refresh=False):
    """近 N 日严格三买 + 严格一买, 各自独立返回 (df3, df1), 不递补。

    df3/df1 均已按 收跌→板块→近者 排序。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    cached = None if refresh else _load_both_cache(days, today)
    if cached is not None:
        return cached
    print(f"全市场扫描 严格三买/一买 (近 {days} 日) ...", flush=True)
    return _scan_and_cache(days, today)


def recent_chan_signals(days=5, refresh=False):
    """近 N 日严格三买(递补一买)。返回 (df, primary_kind)。

    primary_kind = 'buy3' 或 'buy1'(递补)。df 已按 收跌→板块→近者 排序。
    """
    df3, df1 = recent_chan_signals_both(days, refresh)
    if not df3.empty:
        return df3, "buy3"
    return df1, "buy1"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    days = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    refresh = "--refresh" in sys.argv
    df, kind = recent_chan_signals(days=days, refresh=refresh)
    print(f"\n主口径: {'严格三买' if kind == 'buy3' else '严格一买(递补)'}  共 {len(df)} 笔")
    print(df.to_string(index=False))
