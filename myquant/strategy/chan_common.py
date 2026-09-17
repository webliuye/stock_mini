"""缠论形态公共工具 (务实 swing 口径 与 严格笔-线段口径 共用)

级别说明: 数据只有日线单周期, 无法按原文做"1分钟起逐级递归"。本模块在日线图上做
"图上结构"——把图上能划出的段/线段当作构造中枢的次级别走势 (对应课57/58 在最小
分析级别图上的示范), 由此产生的是图上级别的三类买卖点, 非严格跨级别日线级别。

两个口径只在"段的构造"不同 (务实=分型/zigzag swing, 严格=特征序列线段),
买卖点识别规则共用本模块 scan_points; 由策略层喂入各自构造出的段列表。

防未来函数总原则: 所有买卖事件都带"可执行日 exec = 结构确认日 + 1";
结构确认日只依赖 exec 当日及以前的数据。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

EPS = 1e-9

# 两口径策略共享的默认参数
DEFAULT_PARAMS = {
    "buy_style": "third",        # "first"|"second"|"third"
    "macd": (12, 26, 9),
    "warm": 40,                  # 事件判定的 bar 热身 (MACD/面积可靠)
    # 卖出/风控
    "use_sym_sell": True,        # 三卖(向下离开中枢后反抽不回) 结构卖点
    "use_top_divergence": True,  # 一卖/顶背驰
    "use_top_break5": True,      # 顶分型后跌破5日线离场 (课79 分型辅助操作)
    "break5_mode": "fractal",    # "fractal"=收盘<MA5即走 | "valid"=连续break5_days日收MA5下方(有效跌破)
    "break5_days": 3,            # "valid" 模式需要的连续天数
    "break5_min_hold": 2,        # 买入后至少持有 N 天才允许 break5 离场
    "stop_loss": 0.08,           # 收盘 <= 买入价*(1-stop_loss) 离场
    "trail_pct": 0.0,            # 移动止损 (收盘从持仓峰值回落 X), 0=关
    "hold_max": 60,              # 最长持有交易日, 0=不限
    "amount_min": 0.0,           # 买入日成交额下限(亿), 0=关
    # 入场闸 (2026-09-08 一买 lift+市值 双闸结论, 默认全关=现有行为不变)
    #   lift = 买入收盘/信号低点(买点price) - 1。entry_lift_max>0 才启用该带。
    "entry_lift_min": 0.0,       # 买价相对信号低点的最低涨幅(%), 0 = 不设下限
    "entry_lift_max": 0.0,       # 买价相对信号低点的最高涨幅(%), 0 = 关
    "entry_mcap_max": 0.0,       # 买时市值上限(亿); 需 df 带 total_shares 列或 params["shares"], 0=关
    # 早退闸 (2026-09-08 由 noamp 改设): 买入后 early_low_after 个交易日内,
    #   盘中 low 跌破"买入前 early_low_before 个交易日的最低价"→ 当日收盘卖出。
    #   口径=用户指定: 新低参考=买入前3日低点; 盘中触及即当日收盘卖。默认关。
    "early_low_after": 0,        # 买入后监视窗口(交易日), 0=关 (会与 noamp 相互独立)
    "early_low_before": 3,       # 新低参考: 买入前 N 个交易日的区间最低 (不含买入日)
    "start_date": "2023-01-01",  # 之前仅作结构热身, 不产生信号
    # 中枢重锚 (2026-09-02 修复): 离开中枢后未回抽、反而再出同向新段(创新高/新低)
    #   → 旧中枢已甩在下方/上方, 立即作废, 由后续段重新累积。
    #   修: 强趋势里中枢不抬升、任意深回落都"够得着"远古 ZG 而滥发三买/三卖。
    "reanchor_zs": True,         # False = 旧行为(保留单个中枢直到三买/三卖)
}


class Seg:
    """一段 (务实=leg / 严格=线段)。字段:
    i:    序号; sbar/ebar: 起止(原始bar索引, sbar<=ebar); sp/ep: 起止价格;
    hi/lo: 段区间原始高低; d: +1上行/-1下行; conf: 段确认bar(可执行=conf+1);
    area: 段内按方向 MACD 柱面积 (背驰力度)."""
    __slots__ = ("i", "sbar", "ebar", "sp", "ep", "hi", "lo", "d", "conf", "area")

    def __init__(self, i, sbar, ebar, sp, ep, hi, lo, d, conf, area):
        self.i, self.sbar, self.ebar = i, sbar, ebar
        self.sp, self.ep, self.hi, self.lo = sp, ep, hi, lo
        self.d, self.conf, self.area = d, conf, area


def macd_series(close, fast=12, slow=26, signal=9):
    """标准 MACD → (diff, dea, hist); hist=2*(diff-dea) 同现有策略。"""
    c = pd.Series(np.asarray(close, dtype=float))
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    diff = ef - es
    dea = diff.ewm(span=signal, adjust=False).mean()
    hist = 2.0 * (diff - dea)
    return diff.to_numpy(), dea.to_numpy(), hist.to_numpy()


def seg_strength(hist, i0, i1, direction):
    """段 [i0,i1) 的力度: 上行取0轴上方红柱面积和, 下行取绿柱面积和。"""
    if i1 <= i0:
        return 0.0
    h = hist[i0:i1]
    if direction > 0:
        return float(np.sum(np.clip(h, 0, None)))
    return float(np.sum(np.clip(-h, 0, None)))


def fractal_points(high, low, k):
    """k-邻域严格分型。仅保留左右各 k 根都在数据内的分型(确认日=i+k)。
    返回 (tops, bots), 各为 (n,2) int 数组, 行=[bar, confirm_bar];
    top=high[i] 为邻域严格最大, bot=low[i] 为邻域严格最小。"""
    n = len(high)
    tops, bots = [], []
    if n < 2 * k + 1:
        return np.zeros((0, 2), int), np.zeros((0, 2), int)
    for i in range(k, n - k):
        win_h = high[i - k:i + k + 1]
        if (win_h < high[i]).sum() == len(win_h) - 1:   # 自身为严格唯一最大
            tops.append((i, i + k))
        win_l = low[i - k:i + k + 1]
        if (win_l > low[i]).sum() == len(win_l) - 1:    # 自身为严格唯一最小
            bots.append((i, i + k))
    return np.array(tops, int).reshape(-1, 2), np.array(bots, int).reshape(-1, 2)


def build_swings(high, low, k=2, min_gap=3):
    """分型候选 → 贪心交替 zigzag swing。返回 list of [bar, confirm_bar, price, is_top(+1/-1)]。
    min_gap: 相邻异向 swing 点之间最小 bar 距离 (剔除过小摆动)。"""
    tops, bots = fractal_points(high, low, k)
    events = [(b, c, +1, float(high[b])) for b, c in tops]
    events += [(b, c, -1, float(low[b])) for b, c in bots]
    events.sort(key=lambda e: (e[0], e[2]))
    swings = []
    for b, c, tp, p in events:
        if not swings:
            swings.append([b, c, p, tp])
            continue
        last = swings[-1]
        if last[3] == tp:
            if (tp > 0 and p > last[2]) or (tp < 0 and p < last[2]):
                swings[-1] = [b, c, p, tp]
        else:
            if b - last[0] >= min_gap:
                swings.append([b, c, p, tp])
    return swings


def make_seg(idx, b0, b1, p0, p1, conf, high, low, hist):
    """由原始 bar 起止建 Seg; 计算区间 hi/lo、按方向 MACD 面积。"""
    a, z = min(b0, b1), max(b0, b1)
    a, z = max(0, a), min(len(high) - 1, z)
    d = 1 if p1 >= p0 else -1
    hi = float(np.max(high[a:z + 1]))
    lo = float(np.min(low[a:z + 1]))
    area = seg_strength(hist, a, z + 1, d)
    return Seg(idx, a, z, p0, p1, hi, lo, d, conf, area)


def legs_from_swings(swings, high, low, hist):
    """相邻 swing 点组成 leg 序列 (交替方向), 返回 List[Seg]。"""
    segs = []
    for a in range(len(swings) - 1):
        s0, s1 = swings[a], swings[a + 1]
        segs.append(make_seg(a, s0[0], s1[0], s0[2], s1[2], s1[1], high, low, hist))
    return segs


def top_confirm_day_map(high, tops):
    """tops: list of (top_bar, confirm_bar)。返回 {confirm_day: (top_bar, high_price)},
    同一天多顶取更高者。供 break5 卖出局部监视用。"""
    n = len(high)
    m = {}
    for tb, cb in tops:
        if cb is not None and 0 <= cb < n:
            cur = m.get(cb)
            if cur is None or high[tb] >= cur[1]:
                m[cb] = (tb, float(high[tb]))
    return m


def scan_points(segs: List[Seg], params: Dict) -> tuple:
    """在段列表上扫描买卖点 (只依赖已确认段, 无未来)。

    输出两列表, 元素为 dict: {kind, exec, price, seg, ref}。kind ∈
    buy1/buy2/buy3/sell1/sell3。buy 供入场, sell1(顶背驰)/sell3(三卖) 供持仓离场。
    事件产生后再由策略按 buy_style 过滤买点。
    """
    warm = int(params.get("warm", 40))
    n = len(segs)
    buys: List[Dict] = []
    sells: List[Dict] = []
    zs = None          # {'ZD','ZG','armed': None/'up'/'down'}
    buf = []           # 等待中枢形成的段(索引)
    prev_down = None   # 最近一根下行段(索引)
    prev_up = None
    last_buy1 = None   # 最近一买(段索引)
    buy1_low = None
    b2_armed, b2_fired = False, False

    for j in range(n):
        s = segs[j]
        eligible = (s.conf is not None and s.sbar >= warm)
        # ── 背驰 / 一买 / 二买 (下行段侧) ──
        if s.d < 0:
            if prev_down is not None:
                pd = segs[prev_down]
                if s.lo < pd.lo - EPS:
                    # 新低
                    if eligible and s.area < pd.area:
                        ok_ctx = (zs is None) or (s.hi < zs['ZD'] - EPS)
                        if ok_ctx:
                            # zs 非空时一并带出"下跌所离开的那个中枢"(ZD~ZG), 供画图标注中轨
                            buys.append(dict(kind='buy1', exec=s.conf + 1,
                                             price=s.lo, seg=j,
                                             zd=(zs['ZD'] if zs else None),
                                             zg=(zs['ZG'] if zs else None),
                                             born=(zs.get('born') if zs else None),
                                             ref=f"一买背驰 低{s.lo:.2f}<前低{pd.lo:.2f} 面积{s.area:.0f}<{pd.area:.0f}"))
                            last_buy1, buy1_low = j, s.lo
                            b2_armed, b2_fired = True, False
                    # 未产生新一买且跌破旧一买低点 → 旧一买作废
                    if last_buy1 is not None and last_buy1 != j and s.lo < buy1_low - EPS:
                        last_buy1, buy1_low, b2_armed, b2_fired = None, None, False, False
                else:
                    # 抬高低点: 一买后的首个不创新低回调 → 二买
                    if b2_armed and not b2_fired and last_buy1 is not None \
                            and s.lo > buy1_low + EPS and eligible:
                        buys.append(dict(kind='buy2', exec=s.conf + 1,
                                         price=s.lo, seg=j,
                                         ref=f"二买 回调低{s.lo:.2f}不破一买低{buy1_low:.2f}"))
                        b2_armed, b2_fired = False, True
            prev_down = j
        # ── 顶背驰一卖 (上行段侧) ──
        else:
            if prev_up is not None:
                pu = segs[prev_up]
                if s.hi > pu.hi + EPS and eligible and s.area < pu.area:
                    sells.append(dict(kind='sell1', exec=s.conf + 1,
                                      price=s.hi, seg=j,
                                      ref=f"顶背驰 高{s.hi:.2f} 面积{s.area:.0f}<{pu.area:.0f}"))
            prev_up = j

        # ── 中枢: 形成 / 离开 / 三类买卖点 ──
        # 离开语义: 向上离开 = 一根上行段端点(顶)冲破 ZG → 其后下行段为回抽;
        #   回抽低点>ZG → 三买(回抽不回); 回抽落回 ZG 内 → 突破失败, 中枢续延。
        # 向下离开对称: 下行段端点(底)跌破 ZD → 其后上行段 high<ZD → 三卖。
        # 三买/三卖成立后中枢作废, 由后续段重新累积(结构上移/下移)。
        # reanchor_zs (修): 离开后若同向新段(再创新高/新低)先于反向回抽出现,
        #   说明旧中枢已被结构甩开, 立即作废重锚——杜绝"深回落仍够得着远古 ZG"的假三买。
        reroll = bool(params.get("reanchor_zs", True))
        if zs is None:
            buf.append(j)
            if len(buf) >= 3:
                a, b, c = segs[buf[-3]], segs[buf[-2]], segs[buf[-1]]
                lo, hi = max(a.lo, b.lo, c.lo), min(a.hi, b.hi, c.hi)
                if lo < hi - EPS:
                    zs = {'ZD': lo, 'ZG': hi, 'armed': None, 'by': None, 'born': j}
                    buf = []
        else:
            ZD, ZG = zs['ZD'], zs['ZG']
            if zs['armed'] is None:
                if s.d > 0 and s.ep > ZG + EPS:     # 上行段顶冲破 ZG → 向上离开尝试
                    zs['armed'], zs['by'] = 'up', j
                elif s.d < 0 and s.ep < ZD - EPS:   # 下行段底跌破 ZD → 向下离开尝试
                    zs['armed'], zs['by'] = 'down', j
                # 否则与中枢重叠 → 延伸, 中枢不动
            elif zs['armed'] == 'up':
                if s.d < 0:                         # 上行离开后的回抽段
                    if s.lo > ZG + EPS:             # 回抽不回 → 三买
                        if eligible:
                            by = zs.get('by')
                            bh = segs[by].hi if by is not None and 0 <= by < len(segs) else None
                            depth = None if bh is None else (bh - s.lo) / bh * 100
                            buys.append(dict(kind='buy3', exec=s.conf + 1,
                                             price=s.lo, seg=j,
                                             zd=ZD, zg=ZG, born=zs.get('born'), by=by,
                                             bh=bh, depth=depth,
                                             ref=f"三买 离开后回踩{s.lo:.2f}>ZG{ZG:.2f}"))
                        zs, buf = None, [j]         # 结构上移, 从回抽段重新累积
                    else:
                        zs['armed'] = None          # 回抽回到中枢 → 突破失败, 续延
                elif reroll:                        # 再出上行新段、无回抽 → 旧中枢作废重锚
                    zs, buf = None, [j]
                else:
                    zs['armed'] = None
            else:  # 'down'
                if s.d > 0:                         # 向下离开后的反抽段
                    if s.hi < ZD - EPS:             # 反抽不回 → 三卖
                        if eligible:
                            by = zs.get('by')
                            bl = segs[by].lo if by is not None and 0 <= by < len(segs) else None
                            depth = None if bl is None else (s.hi - bl) / bl * 100
                            sells.append(dict(kind='sell3', exec=s.conf + 1,
                                              price=s.hi, seg=j,
                                              zd=ZD, zg=ZG, born=zs.get('born'), by=by,
                                              bl=bl, depth=depth,
                                              ref=f"三卖 离开后反抽{s.hi:.2f}<ZD{ZD:.2f}"))
                        zs, buf = None, [j]         # 结构下移, 从反抽段重新累积
                    else:
                        zs['armed'] = None          # 反抽回到中枢 → 破位失败, 续延
                elif reroll:                        # 再出下行新段、无反抽 → 旧中枢作废重锚
                    zs, buf = None, [j]
                else:
                    zs['armed'] = None
    return buys, sells


def assemble(df: pd.DataFrame, buys: List[Dict], sells: List[Dict],
             tops_pairs: List[tuple], params: Dict) -> pd.DataFrame:
    """day pass: 由预计算买卖事件组装 signal/position (+诊断列)。

    tops_pairs: list of (top_bar, confirm_bar), 供 break5 持仓期局部监视。
    买入只发生在 buy 事件 exec 日 (结构确认次日), 规避引擎当日成交的未来函数。
    """
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    n = len(df)
    if "amount" in df.columns:
        amount = df["amount"].astype(float).to_numpy()
    else:
        amount = (close * df["volume"].astype(float) * 100).to_numpy()
    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().to_numpy()

    amount_min = float(params.get("amount_min", 0.0))
    amt_ok = np.ones(n, bool) if amount_min <= 0 else (amount / 1e8 >= amount_min)

    # 入场双闸 (2026-09-08 一买 lift带+市值 结论, 默认全关=现有行为不变)
    #   lift% = (买入收盘/买点price(信号低点) - 1)*100; entry_lift_max>0 才启用该带。
    #   mcap: 需 df 带 total_shares 列, 或 params["shares"] (股本, 单值)。亿=股×价/1e8。
    entry_lift_min = float(params.get("entry_lift_min", 0.0))
    entry_lift_max = float(params.get("entry_lift_max", 0.0))
    entry_mcap_max = float(params.get("entry_mcap_max", 0.0))
    if "total_shares" in df.columns:
        shares_arr = df["total_shares"].astype(float).to_numpy()
    else:
        sh0 = params.get("shares")
        shares_arr = np.full(n, float(sh0)) if sh0 is not None else np.zeros(n)

    buy_day = np.zeros(n, bool)
    buy_ref = np.full(n, None, dtype=object)
    buy_lo = np.full(n, np.nan)   # exec 日对应的买点 price(=信号低点), 供 lift 入场闸用
    for e in buys:
        ex = e["exec"]
        if ex is not None and 0 <= ex < n and not buy_day[ex]:
            buy_day[ex], buy_ref[ex] = True, e["ref"]
            buy_lo[ex] = float(e["price"]) if e.get("price") is not None else np.nan

    sell_day = np.zeros(n, bool)
    sell_ref = np.full(n, None, dtype=object)
    use_sym = bool(params.get("use_sym_sell", True))
    use_td = bool(params.get("use_top_divergence", True))
    for e in sells:
        if e["kind"] == "sell1" and not use_td:
            continue
        if e["kind"] == "sell3" and not use_sym:
            continue
        ex = e["exec"]
        if ex is not None and 0 <= ex < n and not sell_day[ex]:
            sell_day[ex], sell_ref[ex] = True, e["kind"] + ": " + e["ref"]

    topmap = top_confirm_day_map(high, tops_pairs)

    stop = float(params.get("stop_loss", 0.0))
    trail = float(params.get("trail_pct", 0.0))
    hold_max = int(params.get("hold_max", 0))
    noamp_pct = float(params.get("noamp_pct", 0.0))   # % 单日振幅(高-低/昨收)阈值, 0=关
    noamp_days = int(params.get("noamp_days", 3))
    early_low_after = int(params.get("early_low_after", 0))        # 买后监视窗口, 0=关
    early_low_before = max(1, int(params.get("early_low_before", 3)))
    warm_days = int(params.get("break5_min_hold", 2))
    use_break5 = bool(params.get("use_top_break5", True))
    b5_mode = str(params.get("break5_mode", "fractal"))
    b5_days = max(1, int(params.get("break5_days", 3)))

    signal = np.zeros(n, float)
    position = np.zeros(n, int)
    buy_reason = np.full(n, None, dtype=object)
    sell_reason = np.full(n, None, dtype=object)
    in_pos = False
    entry = 0.0
    entry_low_ref = np.inf        # 早退闸: 买入前 N 日最低 (见 entry 处赋值)
    peak = 0.0
    hold = 0
    amp_max = 0.0                 # 持仓期最大单日振幅(noamp 规则用)
    watcher = False
    w_high = 0.0
    b5_n = 0                     # valid 模式连续收盘 < MA5 的天数

    for i in range(n):
        if not in_pos:
            if buy_day[i] and amt_ok[i]:
                gate_ok = True
                if entry_lift_max > 0 or entry_mcap_max > 0:
                    if entry_lift_max > 0:
                        slo = buy_lo[i]
                        lift = (close[i] / slo - 1.0) * 100.0 if slo > 0 else np.nan
                        if not (entry_lift_min <= lift < entry_lift_max):
                            gate_ok = False
                    if gate_ok and entry_mcap_max > 0:
                        sh = shares_arr[i]
                        cap = (sh * close[i] / 1e8) if sh > 0 else float("inf")
                        if not (cap <= entry_mcap_max):
                            gate_ok = False
                if gate_ok:
                    signal[i], in_pos = 1.0, True
                    entry = float(close[i])
                    peak, hold = entry, 0
                    amp_max = 0.0
                    watcher = False
                    # 早退闸参考: 买入前 early_low_before 个交易日(不含买入日)的最低
                    entry_low_ref = float(low[max(0, i - early_low_before):i].min()) if i > 0 else np.inf
                    buy_reason[i] = buy_ref[i]
        else:
            hold += 1
            if close[i] > peak:
                peak = close[i]
            if noamp_pct > 0 and i > 0:
                prevc = close[i - 1]
                a = (high[i] - low[i]) / prevc * 100.0 if prevc > 0 else 0.0
                if a > amp_max:
                    amp_max = a
            if use_break5 and i in topmap:
                tb, th = topmap[i]
                if not watcher or th >= w_high - EPS:
                    watcher, w_high, b5_n = True, th, 0
            exit_f = False
            if stop > 0 and close[i] <= entry * (1 - stop):
                exit_f, sell_reason[i] = True, f"stop {entry*(1-stop):.2f}"
            if not exit_f and trail > 0 and close[i] <= peak * (1 - trail):
                exit_f, sell_reason[i] = True, f"trail {peak*(1-trail):.2f}"
            if not exit_f and hold_max > 0 and hold >= hold_max:
                exit_f, sell_reason[i] = True, "hold_max"
            # 顶分型破5日线 (课79): fractal=收盘即走; valid=连续收MA5下方(有效跌破)
            if not exit_f and use_break5 and watcher and hold >= warm_days:
                if b5_mode == "valid":
                    if close[i] < ma5[i]:
                        b5_n += 1
                        if b5_n >= b5_days:
                            exit_f, sell_reason[i] = True, f"break5×{b5_days}"
                    else:
                        b5_n = 0
                elif close[i] < ma5[i]:
                    exit_f, sell_reason[i] = True, "break5"
            if not exit_f and sell_day[i]:
                exit_f, sell_reason[i] = True, sell_ref[i]
            # early-low 早退 (由 noamp 改设): 买后第1~early_low_after 个交易日, 盘中低点
            #   跌破"买入前 early_low_before 日最低"→ 当日收盘卖 (盘中触及即算, 用户口径)
            if not exit_f and early_low_after > 0 and 0 < hold <= early_low_after:
                if low[i] < entry_low_ref - EPS:
                    exit_f, sell_reason[i] = True, f"earlylow{early_low_after}d<pre{early_low_before}d低"
            # noamp: 买入后前 noamp_days 个交易日若单日振幅全 < noamp_pct% (反转不活), 到时直接卖
            if not exit_f and noamp_pct > 0 and hold == noamp_days and amp_max < noamp_pct:
                exit_f, sell_reason[i] = True, f"noamp{noamp_days}d<{noamp_pct:g}%"
            if watcher and high[i] > w_high + EPS:
                watcher, b5_n = False, 0
            if exit_f:
                signal[i], in_pos = -1.0, False
                entry, entry_low_ref, hold, amp_max, watcher, b5_n = 0.0, np.inf, 0, 0.0, False, 0
        position[i] = 1 if in_pos else 0

    df = df.copy()
    df["signal"] = signal
    df["position"] = position
    df["buy_reason"] = buy_reason
    df["sell_reason"] = sell_reason
    return df
