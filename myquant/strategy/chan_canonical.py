"""缠论买卖信号 · 严格分型-笔-线段口径 (ChanCanonicalStrategy)

严格 62 课后体系管线:
  1) K线包含处理 (课62/65): 相邻包含合并, 向上取高高/向下取低低, 顺序单向一次过
  2) 分型 (课62): 合并K上严格定义——顶分型第二K 高点与低点均为相邻三根最高; 底对称
  3) 笔 (课77近似): 分型候选异型交替、同型取更极值、异型间距≥MIN_PEN_GAP(合并K)
  4) 线段 (课67/71/78 主算法): 以反方向笔为特征序列, 处理后的元素上找分型断线段;
     第一/二元素缺口=第二种情况需更高元素确认——本版统一用"第三元素顶/底分型确认"
     (即端点 e 由更晚的 turn e+2 的分型确认, 天然无未来), 端点=分型极值。
    缺口特例(78)不单独分支, 以较晚确认换取稳健。
  5) 线段→中枢(≥3线段重叠, 复用 chan_common.scan_points) → 一/二/三买 + 对称卖点。

三族买点、卖出族、风控、防未来与信号口径均与务实版一致 (见 chan_common 说明)。
买卖点直接在线段序列上按 课57 "最小分析级别图" 语义识别。
"""
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseStrategy
from .chan_common import (
    DEFAULT_PARAMS,
    macd_series,
    make_seg,
    scan_points,
    assemble,
)

EPSc = 1e-9
MIN_PEN_GAP = 3   # 异型分型候选之间最小合并K间距 (分型窗口不共用且留有独立K)


def merge_bars(high, low):
    """顺序单向包含处理。返回 merged bars: 每项 [g, d, orig_lo, orig_hi]。
    方向判定: 与已落定最后一根合并K相比 (课65 g_n≥g_{n-1} 为向上处理)。"""
    merged = []
    for i in range(len(high)):
        g, d = float(high[i]), float(low[i])
        if not merged:
            merged.append([g, d, i, i])
            continue
        last = merged[-1]
        contained = (g <= last[0] + EPSc and d >= last[1] - EPSc) \
            or (g >= last[0] - EPSc and d <= last[1] + EPSc)
        if not contained:
            merged.append([g, d, i, i])
            continue
        up = True
        if len(merged) >= 2:
            up = merged[-1][0] >= merged[-2][0]
        if up:
            merged[-1] = [max(g, last[0]), max(d, last[1]), last[2], i]
        else:
            merged[-1] = [min(g, last[0]), min(d, last[1]), last[2], i]
    return merged


def merged_fractals(merged, high, low, n):
    """合并K上严格分型 (原文定义: 顶分型第二K 高/低点均相邻最高)。
    返回 list of dict {mj, typ, price, obar, conf}:
      mj=合并K中心, typ=+1顶/-1底, price=分型价格,
      obar=分型极值所在原始bar (极值合并K内部 argmax/argmin),
      conf=分型确认原始bar (合并K j+1 的末尾, 只有 j+1 存在才作数)。"""
    m = len(merged)
    out = []
    for j in range(1, m - 1):
        g0, d0 = merged[j - 1][0], merged[j - 1][1]
        g1, d1 = merged[j][0], merged[j][1]
        g2, d2 = merged[j + 1][0], merged[j + 1][1]
        lo, hi = merged[j][2], merged[j][3]
        if g1 > g0 + EPSc and g1 > g2 + EPSc and d1 >= d0 - EPSc and d1 >= d2 - EPSc:
            ob = lo + int(np.argmax(high[lo:hi + 1]))
            out.append(dict(mj=j, typ=+1, price=g1, obar=ob,
                            conf=merged[j + 1][3]))
        elif d1 < d0 - EPSc and d1 < d2 - EPSc and g1 <= g0 + EPSc and g1 <= g2 + EPSc:
            ob = lo + int(np.argmin(low[lo:hi + 1]))
            out.append(dict(mj=j, typ=-1, price=d1, obar=ob,
                            conf=merged[j + 1][3]))
    return out


def build_turns(fractals):
    """分型候选 → 交替笔端点 (贪心: 同型取更极值, 异型且合并K间距≥MIN_PEN_GAP 成笔)。"""
    turns = []
    for f in fractals:
        if not turns:
            turns.append(f)
            continue
        last = turns[-1]
        if last["typ"] == f["typ"]:
            if (f["typ"] > 0 and f["price"] > last["price"]) \
                    or (f["typ"] < 0 and f["price"] < last["price"]):
                turns[-1] = f
        else:
            if f["mj"] - last["mj"] >= MIN_PEN_GAP:
                turns.append(f)
    return turns


def segment_split(turns):
    """特征序列法给线段划分。turns 是交替笔端点序列; 线段交替起于顶/底。

    返回 list of (s, e): 闭合线段 = turns[s..e]。end 判定: 向上线段取"反方向(向下)
    笔"的特征序列元素(其起点为顶), 三元素成顶分型即断于中间元素的顶; 向下线段对称。
    第三种元素的极值 turn (e+2) 需已存在 → 天然确认且无未来 (断点e的确认用 e+2 的 conf)。
    """
    m = len(turns)
    res = []
    s = 0
    while s < m - 1:
        top_start = turns[s]["typ"] > 0      # 顶起 → 向下线段
        e = None
        q = 0
        while True:
            t1 = s + 3 + 2 * q               # 潜在端点 (中间元素极值)
            t2 = s + 5 + 2 * q               # 第三元素极值 (确认用)
            if t2 > m - 1:
                break
            if top_start:
                # 向下线段: 特征=向上笔(起于底), 底分型: t1 底最低
                if turns[t1]["price"] < turns[t1 - 2]["price"] - EPSc \
                        and turns[t1]["price"] < turns[t2]["price"] - EPSc:
                    e = t1
                    break
            else:
                # 向上线段: 特征=向下笔(起于顶), 顶分型: t1 顶最高
                if turns[t1]["price"] > turns[t1 - 2]["price"] + EPSc \
                        and turns[t1]["price"] > turns[t2]["price"] + EPSc:
                    e = t1
                    break
            q += 1
        if e is None:
            break                             # 尾部开线段 (未确认) 丢弃
        res.append((s, e))
        s = e
    return res


def canonical_segs(high, low, hist):
    """完整形态管线 → 闭合线段 List[Seg]。段 conf=确认 turn(e+2).conf。"""
    merged = merge_bars(high, low)
    fractals = merged_fractals(merged, high, low, len(high))
    turns = build_turns(fractals)
    pairs = segment_split(turns)
    segs = []
    for idx, (s, e) in enumerate(pairs):
        conf = turns[e + 2]["conf"] if e + 2 < len(turns) else None
        if conf is None:
            continue
        segs.append(make_seg(idx, turns[s]["obar"], turns[e]["obar"],
                             turns[s]["price"], turns[e]["price"],
                             conf, high, low, hist))
    return segs, merged, fractals, turns, pairs


# ---------------------------------------------------------------------------
# 71课 mode1/2 候选线段确认 (课67/71 特征序列两标准; 参照 chan-lun-core find_xds)
# 先量化再选: 本组函数仅作 A/B, 默认 canonical_segs/segment_split 不动。
#   特征元素 = 与线段反向的笔 (含完整区间), 先做包含处理, 在合并后元素上判
#   顶/底分型(全区间: 中间元素 high&low 均最高/最低)。分型第1/2元素:
#     无缺口(有重叠) → mode1, 段立即断于分型中间元素起点(极值 turn);
#     有缺口       → mode2, 记 pending 端点, 等"第二特征序列"(同向笔)出现
#                     反向分型才闭合于 pending。
# 防未来: 闭合段的 conf = 判定所用最新笔腿终点 turn 的 fractal-conf (与现口径
#   "turn 值定死"纪律一致); 末尾未闭合段丢弃。
# ---------------------------------------------------------------------------

def merge_feats(feats):
    """特征序列包含处理 (搬 chan/xd.py _merge_feats)。元素 dict {lo,hi,i0,i1}:
    笔区间[lo,hi], (i0,i1)=该元素保留笔的起止 turn (i0 即该笔起点=极值 turn,
    供断点用)。仅"一个元素完全包含另一个"才合并; 向上处理取 max(lo)/max(hi) 保
    高点更高侧笔, 向下取 min/min 保低点更低侧笔。"""
    out = [feats[0]]
    for e in feats[1:]:
        last = out[-1]
        last_cov = (last["lo"] <= e["lo"] + EPSc and last["hi"] >= e["hi"] - EPSc)
        e_cov = (e["lo"] <= last["lo"] + EPSc and e["hi"] >= last["hi"] - EPSc)
        if not (last_cov or e_cov):
            out.append(e)
            continue
        up = (last["hi"] >= out[-2]["hi"]) if len(out) >= 2 \
            else (e["hi"] >= last["hi"])
        if up:
            kee = e if e["hi"] >= last["hi"] - EPSc else last
            out[-1] = dict(lo=max(last["lo"], e["lo"]), hi=max(last["hi"], e["hi"]),
                           i0=kee["i0"], i1=kee["i1"])
        else:
            kee = e if e["lo"] <= last["lo"] + EPSc else last
            out[-1] = dict(lo=min(last["lo"], e["lo"]), hi=min(last["hi"], e["hi"]),
                           i0=kee["i0"], i1=kee["i1"])
    return out


def _feat_fx(feats, kind):
    """末尾三(合并后)特征元素是否构成分型; 返回首元素下标或 None。
    顶分型: 中间元素 high/low 均三者最高; 底对称。"""
    if len(feats) < 3:
        return None
    a, b, c = feats[-3], feats[-2], feats[-1]
    if kind == "top":
        if b["hi"] > a["hi"] + EPSc and b["hi"] > c["hi"] + EPSc \
                and b["lo"] > a["lo"] + EPSc and b["lo"] > c["lo"] + EPSc:
            return len(feats) - 3
    else:
        if b["lo"] < a["lo"] - EPSc and b["lo"] < c["lo"] - EPSc \
                and b["hi"] < a["hi"] - EPSc and b["hi"] < c["hi"] - EPSc:
            return len(feats) - 3
    return None


def segment_split71(turns):
    """特征序列 mode1/2 线段划分 (状态机移植 chan-lun-core find_xds, 作用在我们
    的 turns 上)。turns 为交替笔端点。返回 list of dict
      {s, e, mode, ct}: s..e=段起止 turn, mode=1/2, ct=判定所用最新笔腿终点 turn
    (conf=turns[ct].conf)。末尾未闭合段丢弃。"""
    m = len(turns)
    if m < 4:
        return []
    # 相邻 turns → 笔腿元数据 (与线段反向者作特征序列)
    pens = []
    for i in range(m - 1):
        t0, t1 = turns[i], turns[i + 1]
        lo = min(t0["price"], t1["price"]); hi = max(t0["price"], t1["price"])
        pens.append(dict(lo=lo, hi=hi, i0=i, i1=i + 1,
                         d="down" if t0["typ"] > 0 else "up"))
    out = []
    direction = pens[0]["d"]
    start = 0
    feats = []       # 第一特征序列(合并后)
    pending = None   # mode2 待确认端点 turn
    second = []      # mode2 第二特征序列(合并后)

    def flip(d):
        return "down" if d == "up" else "up"

    for p in pens:
        if p["d"] == direction:
            if pending is not None:
                # mode2: 第二特征序列收集同向笔, 等反向分型确认
                second.append(p); second = merge_feats(second)
                kind2 = "bottom" if direction == "up" else "top"
                if _feat_fx(second, kind2) is not None:
                    out.append(dict(s=start, e=pending, mode=2, ct=p["i1"]))
                    direction = flip(direction)
                    start = pending
                    pending = None; second = []; feats = []
            continue
        # 反向笔 → 第一特征序列元素
        feats.append(p); feats = merge_feats(feats)
        if pending is not None:
            continue
        kind = "top" if direction == "up" else "bottom"
        fxi = _feat_fx(feats, kind)
        if fxi is None:
            continue
        a, b, c = feats[fxi], feats[fxi + 1], feats[fxi + 2]
        if direction == "up":
            gap = a["hi"] < b["lo"] - EPSc          # 顶分型第1/2元素有缺口
        else:
            gap = a["lo"] > b["hi"] + EPSc          # 底分型第1/2元素有缺口
        point = b["i0"]                              # 中间特征元素起点 = 段端点极值 turn
        if not gap:
            out.append(dict(s=start, e=point, mode=1, ct=p["i1"]))
            direction = flip(direction)
            start = point
            feats = []
        else:
            pending = point
            second = []
    return out


def canonical_segs71(high, low, hist):
    """71 mode1/2 候选管线 (同 canonical_segs, 仅换 segment_split71)。
    返回 (segs, merged, fractals, turns, seg_modes); seg_modes 与 segs 对齐。"""
    merged = merge_bars(high, low)
    fractals = merged_fractals(merged, high, low, len(high))
    turns = build_turns(fractals)
    splits = segment_split71(turns)
    segs, modes = [], []
    for sp in splits:
        s, e, ct = sp["s"], sp["e"], sp["ct"]
        if e >= len(turns) or ct >= len(turns):
            continue
        conf = turns[ct]["conf"]
        if conf is None:
            continue
        segs.append(make_seg(len(segs), turns[s]["obar"], turns[e]["obar"],
                             turns[s]["price"], turns[e]["price"],
                             conf, high, low, hist))
        modes.append(sp["mode"])
    return segs, merged, fractals, turns, modes


class ChanCanonicalStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        p = dict(DEFAULT_PARAMS)
        if params:
            p.update(params)
        super().__init__(name="chan_canonical", params=p)
        self.buy_style = p["buy_style"]
        self.macd = tuple(p.get("macd", (12, 26, 9)))
        self.start_date = p.get("start_date")
        if self.start_date:
            self.start_date = pd.Timestamp(self.start_date)
        logger.info(f"缠论严格笔线段: 买族={self.buy_style}, "
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
        segs, merged, fractals, turns, pairs = canonical_segs(high, low, hist)

        buys, sells = scan_points(segs, self.params)
        kind = {"first": "buy1", "second": "buy2", "third": "buy3"}[self.buy_style]
        buys = [e for e in buys if e["kind"] == kind]

        # break5 顶分型监视: 线段端点里的顶 turn (原始bar, conf)
        tops_pairs = [(t["obar"], t["conf"]) for t in turns if t["typ"] > 0]

        out = assemble(df, buys, sells, tops_pairs, self.params)
        if self.start_date is not None:
            pre = out["date"] < self.start_date
            out.loc[pre, ["signal", "position"]] = (0.0, 0)

        n_buy = int((out["signal"] == 1).sum())
        n_sell = int((out["signal"] == -1).sum())
        logger.info(f"缠论严格[{self.buy_style}] 合并K{len(merged)} 分型{len(fractals)} "
                    f"端点{len(turns)} 线段{len(segs)} 买事件{len(buys)} 卖事件{len(sells)} "
                    f"→ 买入{n_buy} 卖出{n_sell}")
        return out
