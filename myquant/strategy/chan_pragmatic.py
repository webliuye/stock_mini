"""缠论买卖信号 · 务实骨架口径 (ChanPragmaticStrategy)

形态构造: k 邻域分型 → 贪心 zigzag swing → leg (每段 = 两个相邻 swing 极值)。
分型/中枢/买卖点均为"图上近似", 不涉及 62 课后严格笔-线段定义 (见 chan_canonical)。

三族买点 (buy_style 参数选择):
  first  一买: 下行段创新低且 MACD 柱面积 < 前一下行段 (背驰抄底)
  second 二买: 一买后首个回调段不创新低 (抬高底, 回踩买入)
  third  三买: 中枢向上离开后回踩段低点不破 ZG
卖出: 顶背驰一卖 / 三卖 (对称) / 顶分型破5日线 / 止损 / 移动止损 / 持有上限。

防未来函数: 买卖事件统一带 exec=结构确认次日; 引擎当日收盘成交, 因此不泄漏。
级别语义: 日线上做图上级别结构 (见 chan_common 模块说明)。
"""
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseStrategy
from .chan_common import (
    DEFAULT_PARAMS,
    macd_series,
    build_swings,
    legs_from_swings,
    scan_points,
    assemble,
    fractal_points,
)


class ChanPragmaticStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        super().__init__(name="chan_pragmatic", params=p)
        self.buy_style = p["buy_style"]
        self.fractal_k = int(p.get("fractal_k", 2))
        self.min_gap = int(p.get("prag_min_gap", 3))
        self.macd = tuple(p.get("macd", (12, 26, 9)))
        self.start_date = p.get("start_date")
        if self.start_date:
            self.start_date = pd.Timestamp(self.start_date)
        logger.info(f"缠论务实骨架: 买族={self.buy_style}, fractal_k={self.fractal_k}, "
                    f"min_gap={self.min_gap}, 信号起点={self.start_date or '全期'}")

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
        swings = build_swings(high, low, k=self.fractal_k, min_gap=self.min_gap)
        segs = legs_from_swings(swings, high, low, hist)

        buys, sells = scan_points(segs, self.params)
        # 按 buy_style 只留一族买点 (scan 已维护一买喂二买的状态)
        kind = {"first": "buy1", "second": "buy2", "third": "buy3"}[self.buy_style]
        buys = [e for e in buys if e["kind"] == kind]

        # break5 顶分型监视用 (k=1 三根K顶分型, 贴近课79/80 用法)
        tops, _ = fractal_points(high, low, k=1)
        tops_pairs = [(int(b), int(c)) for b, c in tops]

        out = assemble(df, buys, sells, tops_pairs, self.params)
        if self.start_date is not None:
            pre = out["date"] < self.start_date
            out.loc[pre, ["signal", "position"]] = (0.0, 0)

        n_buy = int((out["signal"] == 1).sum())
        n_sell = int((out["signal"] == -1).sum())
        logger.info(f"缠论务实[{self.buy_style}] 段{len(segs)} 买事件{len(buys)} 卖事件{len(sells)} "
                    f"→ 买入{n_buy} 卖出{n_sell}")
        return out
