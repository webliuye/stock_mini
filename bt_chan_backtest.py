"""缠论买卖点 · 三类买点 vs 两口径 · 回测对比

把《教你炒股票108课·张星星注解版》的一/二/三类买点做成可回测信号。
口径 (买点所在结构的构造方法):
  务实  = 分型/swing zigzag 段 (图上近似, chan_pragmatic)
  严格  = 包含处理→分型→笔→特征序列线段 (62课后原文体系, chan_canonical)
买族: 一买(背驰新低) / 二买(一买后抬高底回调) / 三买(中枢突破后回踩不回)。
共 2×3 = 6 个变体, 卖出族/风控完全相同 (SELL_DEFAULT), 公平对比买点差异。

级别语义: 数据只有日线单周期, 无法做原文"1分钟起逐级递归"; 两口径都在日线图上做
"图上结构", 把图上段/线段当作构造中枢的次级别走势 (课57/58 最小分析级别图做法),
产出的是图上级别的买卖点, 非严格跨级别日线级别。

用法:
  ./.venv/Scripts/python bt_chan_backtest.py            # 默认抽样 500
  ./.venv/Scripts/python bt_chan_backtest.py 50         # 小样快速验证
  ./.venv/Scripts/python bt_chan_backtest.py all
  ./.venv/Scripts/python bt_chan_backtest.py 50 pragmatic second   # 子集
  ./.venv/Scripts/python bt_chan_backtest.py --events 000001       # 单股事件抽查
"""
import random, re, sys
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
from loguru import logger

from myquant.backtest.engine import BacktestEngine
from myquant.strategy.base import BaseStrategy
from myquant.strategy.chan_pragmatic import ChanPragmaticStrategy
from myquant.strategy.chan_canonical import ChanCanonicalStrategy, canonical_segs
from myquant.strategy.chan_canonical_pen import ChanCanonicalPenStrategy, canonical_pen
from myquant.strategy.chan_common import (
    DEFAULT_PARAMS, macd_series, build_swings, legs_from_swings,
    scan_points, assemble, fractal_points,
)

logger.remove()
logger.add(sys.stderr, level="ERROR")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CACHE_DIR = Path("data/cache")
PAT = re.compile(r"^(\d{6})_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_qfq\.parquet$")
N_ARG = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "500"
START = "2023-01-01"
END = "2026-08-28"
SEED = 42

# ── 卖出族/风控 默认 (6 变体共享, 头部可改) ──
SELL_DEFAULT = {
    "start_date": START,
    "use_sym_sell": True,        # 三卖 (中枢向下离开后反抽不回)
    "use_top_divergence": True,  # 一卖 (顶背驰)
    "use_top_break5": True,      # 顶分型后收盘跌破5日线 (课79/80)
    "break5_min_hold": 2,        # 买入后至少持有 N 天
    "stop_loss": 0.08,
    "trail_pct": 0.0,            # 0 = 关
    "hold_max": 60,
    "amount_min": 0.0,           # 0 = 关 (不加流动性门, 保持买卖点纯粹)
}

# ── 6 变体矩阵: (策略类, 口径名, 买族, 族名) ──
VARIANTS = [
    (ChanPragmaticStrategy, "务实", "first",  "务实/一买"),
    (ChanPragmaticStrategy, "务实", "second", "务实/二买"),
    (ChanPragmaticStrategy, "务实", "third",  "务实/三买"),
    (ChanCanonicalStrategy, "严格", "first",  "严格/一买"),
    (ChanCanonicalStrategy, "严格", "second", "严格/二买"),
    (ChanCanonicalStrategy, "严格", "third",  "严格/三买"),
    (ChanCanonicalPenStrategy, "严格笔", "first",  "严格笔/一买"),
    (ChanCanonicalPenStrategy, "严格笔", "second", "严格笔/二买"),
    (ChanCanonicalPenStrategy, "严格笔", "third",  "严格笔/三买"),
]

STYLE_NAME = {"first": "一买", "second": "二买", "third": "三买"}
STYLE_CN = {"first": "一买", "second": "二买", "third": "三买"}

# 退出模式研究: 三种"拿主升"的离场 (结构卖点/顶背驰三卖始终开启, 只改破5日线处理)
EXIT_MODES = [
    ("破5现行", dict(use_top_break5=True, break5_mode="fractal")),
    ("破5有效", dict(use_top_break5=True, break5_mode="valid", break5_days=3)),
    ("纯结构",  dict(use_top_break5=False)),
]


class _PreSig(BaseStrategy):
    """喂预计算 signal 帧给引擎 (study 模式复用同一结构算多种退出)。"""
    def __init__(self, out, name):
        super().__init__(name=name, params={})
        self._out = out

    def generate_signals(self, data):
        return self._out


def build_structure(df, impl):
    """算一次形态结构 → (segs, tops_pairs)。口径与 generate_signals 内一致。"""
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    _, _, hist = macd_series(close, 12, 26, 9)
    if impl == "pragmatic":
        segs = legs_from_swings(build_swings(high, low, k=2, min_gap=3), high, low, hist)
        tops, _ = fractal_points(high, low, k=1)
        tops_pairs = [(int(b), int(c)) for b, c in tops]
    else:
        segs, *_, turns, _ = canonical_segs(high, low, hist)
        tops_pairs = [(t["obar"], t["conf"]) for t in turns if t["typ"] > 0]
    return segs, tops_pairs


def build_out(df, structure, impl, style, params):
    """由已缓存结构 → 组装 signal/position 帧 (含 start_date 屏蔽, 同策略)。"""
    segs, tops_pairs = structure
    buys, sells = scan_points(segs, params)
    kind = {"first": "buy1", "second": "buy2", "third": "buy3"}[style]
    buys = [e for e in buys if e["kind"] == kind]
    out = assemble(df, buys, sells, tops_pairs, params)
    sd = params.get("start_date")
    if sd:
        pre = out["date"] < pd.Timestamp(sd)
        out.loc[pre, ["signal", "position"]] = (0.0, 0)
    return out


def hold_main():
    """--hold 研究: 3 退出模式 × 6 买族, 结构只算一次, 引擎级完整回测。

    买点与结构共用主表; 退出 = {破5现行 / 破5有效(连续3日收MA5下方) / 纯结构卖点}。
    风控: 止损8%, 无限持仓上限 (放开 hold_max), 无流动性门。
    """
    cache_clean = build_cache_map()
    codes = sorted(cache_clean.keys())
    if N_ARG == "all":
        selected = codes
    else:
        selected = random.Random(SEED).sample(codes, min(int(N_ARG), len(codes)))

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    engine = BacktestEngine(config)
    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)

    print(f"[信息] 加载数据 ...", end="", flush=True)
    dfs = {}
    for idx, code in enumerate(selected):
        try:
            df = load_stock(cache_clean[code])
            if len(df) < 110:
                continue
            dfs[code] = df
        except Exception:
            pass
        if (idx + 1) % 100 == 0:
            print(f"\r[信息] 加载数据 {idx + 1}/{len(selected)} ...", end="", flush=True)
    print(f"\r[信息] 加载完成, 有效 {len(dfs)} 只                ")

    # 基础参数: 与 SELL_DEFAULT 一致但 hold_max=0(不封顶) — 研究"拿住主升"
    base = dict(DEFAULT_PARAMS)
    base.update({"start_date": START, "stop_loss": 0.08, "trail_pct": 0.0,
                 "hold_max": 0, "amount_min": 0.0, "break5_min_hold": 2,
                 "break5_mode": "fractal", "break5_days": 3})

    agg = {}
    orders = []
    print(f"[信息] 运行 {len(EXIT_MODES)} 退出 × 6 买族 × {len(dfs)} 只 (共享结构计算) ...")
    done = 0
    for code, df in dfs.items():
        structs = {"pragmatic": build_structure(df, "pragmatic"),
                   "canonical": build_structure(df, "canonical")}
        for impl, tag in (("pragmatic", "务实"), ("canonical", "严格")):
            structure = structs[impl]
            for style in ("first", "second", "third"):
                fam = f"{tag}/{STYLE_CN[style]}"
                for em, em_cfg in EXIT_MODES:
                    params = dict(base)
                    params.update(em_cfg)
                    out = build_out(df, structure, impl, style, params)
                    name = f"chan_{impl}_{style}|{em}"
                    engine._reset()
                    r = engine.run(_PreSig(out, name), df, symbol=code, verbose=False)
                    win = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
                    bh = (win["close"].iloc[-1] / win["close"].iloc[0] - 1) * 100 if len(win) >= 2 else 0.0
                    expo = 0.0
                    eq = r.equity_curve
                    if eq is not None and not eq.empty and "position" in eq.columns:
                        e = eq[(eq["date"] >= start_ts) & (eq["date"] <= end_ts)]
                        if len(e) > 0:
                            expo = float((e["position"] > 0).mean() * 100)
                    sells = [t for t in r.trades if t.direction == "SELL"]
                    holds, ei = [], None
                    for ii, rr in out.iterrows():
                        if rr["signal"] == 1:
                            ei = ii
                        elif rr["signal"] == -1 and ei is not None:
                            holds.append(ii - ei)
                            ei = None
                    rec = {"ret": r.total_return_pct * 100, "bh": bh,
                           "mdd": r.max_drawdown_pct * 100,
                           "expo": expo, "nbuy": r.total_trades,
                           "sell_pnl": [t.pnl_pct for t in sells],
                           "hold": holds, "open": r.total_trades - len(sells)}
                    a = agg.setdefault((fam, em), [])
                    a.append(rec)
        done += 1
        if done % 100 == 0:
            print(f"  …{done}/{len(dfs)} 只结构算完", flush=True)

    # 汇总
    out_rows = []
    emit = lambda s="": (out_rows.append(s), print(s))
    emit("\n" + "=" * 116)
    emit(f"  缠论买点 × 退出模式 研究 — 能否'拿住主升'")
    emit(f"  窗口 {START} ~ {END} | 样本 {len(dfs)} 只 | 种子 {SEED}")
    emit(f"  买族: 6 族(务实/严格 × 一/二/三买) | 退出: 破5现行 / 破5有效(连续3日收MA5下方≈有效跌破) / 纯结构卖点")
    emit(f"  结构卖出(顶背驰一卖/三卖)恒开 | 止损8% | 持仓不封顶 | 无流动性门")
    emit(f"  '留仓' = 到窗口末仍未平仓的买入次数 (open 仓位计入收益但无平仓盈亏)")
    emit("=" * 116)
    hdr = ("  {:<9}{:<10}{:>5}{:>6}{:>7}{:>8}{:>8}{:>8}{:>7}{:>7}{:>6}"
           ).format("买族", "退出", "有交", "买入", "留仓", "收益", "持仓%", "期望/笔",
                    "胜率", "回撤", "均持日")
    emit(hdr)
    emit("  " + "-" * 110)
    for fam in (f"{t}/{STYLE_CN[s]}" for t in ("务实", "严格") for s in ("first", "second", "third")):
        for em, _ in EXIT_MODES:
            a = agg.get((fam, em))
            if not a or not any(x["nbuy"] > 0 for x in a):
                emit(f"  {fam:<9}{em:<10}  —")
                continue
            trad = [x for x in a if x["nbuy"] > 0]
            sells = [p for x in trad for p in x["sell_pnl"]]
            holds = [h for x in trad for h in x["hold"]]
            nb = sum(x["nbuy"] for x in trad)
            op = sum(x["open"] for x in trad)
            wr = float(np.mean([p > 0 for p in sells]) * 100) if sells else float("nan")
            pt = float(np.mean(sells) * 100) if sells else float("nan")
            hd = float(np.mean(holds)) if holds else 0.0
            line = ("  {:<9}{:<10}{:>5}{:>6}{:>7}{:>+7.2f}%{:>7.1f}%{:>+7.2f}%"
                    "{:>6.1f}%{:>6.1f}%{:>5.1f}"
                    ).format(fam, em, len(trad), nb, op,
                             np.mean([x["ret"] for x in trad]),
                             np.mean([x["expo"] for x in trad]),
                             pt, wr,
                             np.mean([x["mdd"] for x in trad]),
                             hd)
            emit(line)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rpt = Path("reports") / f"bt_chan_hold_{ts}.txt"
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text("\n".join(out_rows), encoding="utf-8")
    emit(f"\n报告已保存: {rpt}")


def build_cache_map():
    cache_map = {}
    for p in CACHE_DIR.glob("*.parquet"):
        m = PAT.match(p.name)
        if m:
            code, end = m.group(1), m.group(3)
            if code not in cache_map or end > cache_map[code][1]:
                cache_map[code] = (str(p), end)
    return {c: fp for c, (fp, _) in cache_map.items()}


def load_stock(fp):
    df = pd.read_parquet(fp)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    return df


def dump_events(code, cache_clean):
    """单股事件抽查: 交易日志 (买/卖日期、价位、理由), 用于无未来函数人工核对。"""
    if code not in cache_clean:
        print(f"[错误] 无 {code} 的缓存数据")
        return
    df = load_stock(cache_clean[code])
    for Cls, tag, style, _ in VARIANTS:
        strat = Cls(params={"buy_style": style, **SELL_DEFAULT})
        out = strat.generate_signals(df)
        sig = out[out["signal"] != 0]
        print(f"\n===== {tag} / {STYLE_NAME[style]}  {code}  ({len(out)} 根K) =====")
        if len(sig) == 0:
            print("  无交易信号")
            continue
        for _, r in sig.iterrows():
            act = "买" if r["signal"] == 1 else "卖"
            reason = r["buy_reason"] if r["signal"] == 1 else r["sell_reason"]
            print(f"  {r['date'].strftime('%Y-%m-%d')} {act}  close={r['close']:.2f}  {reason}")


def main():
    # --events 分支
    if "--events" in sys.argv:
        i = sys.argv.index("--events")
        code = sys.argv[i + 1].zfill(6)
        cache_clean = build_cache_map()
        dump_events(code, cache_clean)
        return
    # --hold 分支 (退出模式研究)
    if "--hold" in sys.argv:
        hold_main()
        return

    cache_clean = build_cache_map()
    codes = sorted(cache_clean.keys())

    # 口径/买族 子集过滤 (供快速调参)
    subset = [a for a in sys.argv[1:] if a in ("pragmatic", "canonical", "pen", "first", "second", "third")]
    want_impl = [a for a in ("pragmatic", "canonical", "pen") if a in subset]
    want_style = [a for a in ("first", "second", "third") if a in subset]
    _IMPL_TAG = {"pragmatic": "务实", "canonical": "严格", "pen": "严格笔"}
    variants = [v for v in VARIANTS
                if (not want_impl or v[1] in [_IMPL_TAG.get(a) for a in want_impl])
                and (not want_style or v[2] in want_style)]
    if not variants:
        variants = VARIANTS

    if N_ARG == "all":
        selected = codes
    else:
        rng = random.Random(SEED)
        selected = rng.sample(codes, min(int(N_ARG), len(codes)))

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    engine = BacktestEngine(config)
    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)

    print(f"[信息] 加载数据 ...", end="", flush=True)
    dfs = {}
    for idx, code in enumerate(selected):
        try:
            df = load_stock(cache_clean[code])
            if len(df) < 110:
                continue
            dfs[code] = df
        except Exception:
            pass
        if (idx + 1) % 100 == 0:
            print(f"\r[信息] 加载数据 {idx + 1}/{len(selected)} ...", end="", flush=True)
    print(f"\r[信息] 加载完成, 有效 {len(dfs)} 只                ")

    def run_variant(Cls, style, code, df):
        strat = Cls(params={"buy_style": style, **SELL_DEFAULT})
        engine._reset()
        r = engine.run(strat, df, symbol=code, verbose=False)
        win = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        bh = (win["close"].iloc[-1] / win["close"].iloc[0] - 1) * 100 if len(win) >= 2 else 0.0
        # 市场暴露 = 引擎真实持仓占比 (窗口内)
        exposure = 0.0
        eq = r.equity_curve
        if eq is not None and not eq.empty and "position" in eq.columns:
            e = eq[(eq["date"] >= start_ts) & (eq["date"] <= end_ts)]
            if len(e) > 0:
                exposure = float((e["position"] > 0).mean() * 100)
        sells = [t.pnl_pct for t in r.trades if t.direction == "SELL"]
        return {
            "ret": r.total_return_pct * 100, "bh": bh,
            "wr": r.win_rate * 100, "pf": r.profit_factor,
            "mdd": r.max_drawdown_pct * 100, "trades": r.total_trades,
            "expo": exposure, "sell_pnl": sells,
        }, r

    agg = {}
    print(f"[信息] 运行 {len(variants)} 个变体 × {len(dfs)} 只 ...")
    for Cls, tag, style, label in variants:
        v_res, all_sells = [], []
        for code, df in dfs.items():
            rec, r = run_variant(Cls, style, code, df)
            if rec["trades"] > 0:
                v_res.append(rec)
                all_sells.extend(rec["sell_pnl"])
        agg[label] = (v_res, all_sells)
        print(f"  完成 {label} (有交易 {len(v_res)} 只, 平仓 {len(all_sells)} 笔)")

    def summarize(v_res, all_sells):
        if not v_res:
            return None
        pf_vals = [x["pf"] for x in v_res if x["pf"] < 999]
        per_trade = float(np.mean(all_sells) * 100) if all_sells else float("nan")
        return {
            "n": len(v_res), "ntrades": len(all_sells),
            "ret": np.mean([x["ret"] for x in v_res]),
            "bh": np.mean([x["bh"] for x in v_res]),
            "wr": np.mean([x["wr"] for x in v_res]),
            "pf": np.mean(pf_vals) if pf_vals else np.nan,
            "mdd": np.mean([x["mdd"] for x in v_res]),
            "ntr": np.mean([x["trades"] for x in v_res]),
            "expo": np.mean([x["expo"] for x in v_res]),
            "pt": per_trade,
        }

    out = []
    emit = lambda s="": (out.append(s), print(s))
    emit("\n" + "=" * 108)
    emit(f"  缠论三类买点 · 务实/严格两口径 回测对比")
    emit(f"  窗口 {START} ~ {END} | 样本 {len(dfs)} 只 | 种子 {SEED}")
    emit(f"  级别语义: 日线图上'图上结构'(图上级别买卖点), 非严格跨级别日线级别")
    emit(f"  卖出默认(全部变体共用): 顶背驰一卖/三卖/顶分型破5日/止损8%/持60日上限; 无流动性门")
    emit("=" * 108)
    hdr = ("  {:<12}{:>6}{:>7}{:>9}{:>8}{:>8}{:>7}{:>7}{:>7}{:>7}{:>7}{:>7}"
           ).format("变体", "有交易", "总笔", "收益", "B&H", "超额", "胜率", "盈亏比",
                    "回撤", "持仓", "期望/笔", "均交易")
    emit(hdr)
    emit("  " + "-" * 104)
    for _, _, _, label in variants:
        s = summarize(*agg.get(label, ([], [])))
        if not s:
            emit(f"  {label:<12}{'-':>6}")
            continue
        excess = s["ret"] - s["bh"]
        line = ("  {:<12}{:>6}{:>7}{:>+8.2f}%{:>+7.2f}%{ex:>+7.2f}%"
                "{:>6.1f}%{:>6.2f}{:>6.1f}%{:>6.1f}%{:>+7.2f}%{:>6.1f}"
                ).format(label, s["n"], s["ntrades"], s["ret"], s["bh"], s["wr"],
                         s["pf"], s["mdd"], s["expo"], s["pt"], s["ntr"], ex=excess)
        emit(line)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rpt = Path("reports") / f"bt_chan_backtest_{ts}.txt"
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text("\n".join(out), encoding="utf-8")
    emit(f"\n报告已保存: {rpt}")


if __name__ == "__main__":
    main()
