"""缠论买卖信号 · 严格分型/笔 + 笔级中枢通道 (ChanCanonicalPenStrategy)

本模块是 chan_canonical 的一对一副本, 唯一差异在"中枢/买卖点锚定的级别":
  原版(chan_canonical): 笔端点(turns) → 特征序列线段(segs) → 线段级中枢/买卖点。
  本版(笔级通道)     : 笔端点(turns) → 直接把相邻笔当次级别走势的"笔腿"
                        → 笔级中枢/买卖点 —— 与主流实现口径对齐
                        (chan-lun-core 用 find_zs(bis) 笔级中枢,
                         czsc 无线段层, 分型/中枢/买点全在笔级)。

为什么要这条通道 (见 _diff_chan_ref 对账与 [[chan-ref-crosscheck-findings]]):
  我们线段级中枢常为 0 个(单段巨大可长达17个月), 中心极稀 → 上涨被大线段吞掉,
  旧中枢不重锚, 于是"严格三买"多是够得着远古 ZG 的结构回声;
  主流把中枢建在笔级, 中心随上涨频繁上移, 三买锚在当前 ZG 附近。

笔端点本版与原版同源(merge→fractal→turns, 与参照 ~99% 对齐), 因此本版隔离出的
差异只有一个: 中枢从线段级 → 笔级。买卖点、卖出族、风控、防未来口径完全不变。

防未来: 笔腿 seg 的 conf = 终点 turn 的分型确认 bar(相邻两腿才定死一笔),
  买卖事件 exec = conf+1, 与 chan_common 总原则一致, 引擎当日收盘成交不泄漏。
"""
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseStrategy
from .chan_canonical import merge_bars, merged_fractals, build_turns
from .chan_common import (
    DEFAULT_PARAMS,
    macd_series,
    make_seg,
    scan_points,
    assemble,
)


def pen_legs_from_turns(turns, high, low, hist):
    """相邻笔端点(turns) → 笔腿 List[Seg]。

    防未来与严格版同构: 端点 turn i+1 会一直被同向更极端分型替换, 直到下一反向
    turn i+2 出现才定死; 故第 a 腿 (a→a+1) 的 conf 取 turns[a+2]["conf"] ——
    与 canonical_segs 用 turns[e+2].conf 定线段尾部完全同一纪律。缺后继 turn 的
    尾部 1~2 腿不可定死 → 丢弃 (可执行日只能来自已闭合腿)。"""
    legs = []
    for a in range(len(turns) - 2):
        t0, t1 = turns[a], turns[a + 1]
        legs.append(make_seg(a, t0["obar"], t1["obar"], t0["price"], t1["price"],
                             turns[a + 2]["conf"], high, low, hist))
    return legs


def canonical_pen(high, low, hist):
    """完整形态管线 → 笔腿 List[Seg] (合并K/分型/笔端点与原版一致)。"""
    merged = merge_bars(high, low)
    fractals = merged_fractals(merged, high, low, len(high))
    turns = build_turns(fractals)
    legs = pen_legs_from_turns(turns, high, low, hist)
    return legs, merged, fractals, turns


class ChanCanonicalPenStrategy(BaseStrategy):
    """严格分型/笔 + 笔级中枢 (chan_canonical 复制改级别通道)。"""
    def __init__(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        super().__init__(name="chan_canonical_pen", params=p)
        self.buy_style = p["buy_style"]
        self.macd = tuple(p.get("macd", (12, 26, 9)))
        self.start_date = p.get("start_date")
        if self.start_date:
            self.start_date = pd.Timestamp(self.start_date)
        logger.info(f"缠论严格笔级通道: 买族={self.buy_style}, "
                    f"信号起点={self.start_date or '全期'}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        n = len(df)
        if n < 60:
            df["signal"] = 0.0
            df["position"] = 0
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()

        _, _, hist = macd_series(close, *self.macd)
        legs, merged, fractals, turns = canonical_pen(high, low, hist)

        buys, sells = scan_points(legs, self.params)
        kind = {"first": "buy1", "second": "buy2", "third": "buy3"}[self.buy_style]
        buys = [e for e in buys if e["kind"] == kind]

        # break5 顶分型监视: 顶 turn (原始bar, conf), 与原版同
        tops_pairs = [(t["obar"], t["conf"]) for t in turns if t["typ"] > 0]

        out = assemble(df, buys, sells, tops_pairs, self.params)
        if self.start_date is not None:
            pre = out["date"] < self.start_date
            out.loc[pre, ["signal", "position"]] = (0.0, 0)

        n_buy = int((out["signal"] == 1).sum())
        n_sell = int((out["signal"] == -1).sum())
        logger.info(f"缠论严格笔级[{self.buy_style}] 合并K{len(merged)} 分型{len(fractals)} "
                    f"端点{len(turns)} 笔腿{len(legs)} 买事件{len(buys)} 卖事件{len(sells)} "
                    f"→ 买入{n_buy} 卖出{n_sell}")
        return out
