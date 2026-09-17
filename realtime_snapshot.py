# -*- coding: utf-8 -*-
"""当日实时盘口快照 → 构造当日盘中 bar, 供日线扫描追加"今天"这根未完成的 K 线。

AlphaFeed quotes.get 返回当日实时: open/high/low/last_price/volume/amount/trade_date。
日线缓存是前复权(qfq), 其最新一根就是不复权的现价(已验证 prev_close 对齐),
故实时 bar 可直接追加到 qfq 序列末尾, 无需再复权。

用法:
  from realtime_snapshot import fetch_realtime_bars
  bars = fetch_realtime_bars(codes)   # {code: dict(date, open, high, low, close, volume, amount)}
"""
import sys
import yaml
import pandas as pd


def _to_af_symbol(code):
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("9", "8", "4")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _alphafeed_api():
    from alphafeed import AlphaFeed
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    key = (cfg.get("data", {}) or {}).get("alphafeed_api_key")
    return AlphaFeed(api_key=key, timeout=30)


def fetch_realtime_bars(codes, batch=300):
    """批量拉当日实时快照 → {code: dict(date, open, high, low, close, volume, amount)}。

    非交易日 / 数据源无当日数据时返回空 dict; 由调用方按 date > 缓存末日 决定是否追加。
    """
    out = {}
    codes = sorted(set(codes))
    try:
        api = _alphafeed_api()
    except Exception as e:
        print(f"[实时] AlphaFeed 初始化失败: {e}")
        return out

    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        syms = [_to_af_symbol(c) for c in chunk]
        try:
            df = api.quotes.get(symbols=syms, to_dataframe=True)
        except Exception as e:
            print(f"[实时] 批次 {i // batch + 1} 拉取失败: {type(e).__name__}: {e}")
            continue
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            code = str(r["symbol"]).split(".")[0]
            try:
                td = pd.to_datetime(r["trade_date"])
            except Exception:
                continue
            if pd.isna(td):
                continue
            out[code] = dict(date=td,
                             open=float(r["open"]),
                             high=float(r["high"]),
                             low=float(r["low"]),
                             close=float(r["last_price"]),
                             volume=float(r["volume"]),
                             amount=float(r["amount"]))
    return out


def append_today_bar(df, live):
    """把当日实时 bar 追加到 df 末尾 (若 live.date > df 末日)。返回 df。"""
    if not live:
        return df
    last_date = pd.Timestamp(df["date"].iloc[-1])
    if live["date"] <= last_date:
        return df
    row = pd.DataFrame([{c: live[c] for c in
                         ("date", "open", "high", "low", "close", "volume", "amount")}])
    return pd.concat([df, row], ignore_index=True)
