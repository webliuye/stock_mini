# -*- coding: utf-8 -*-
"""缠论买点监控页 — 近五日 / 明日可买 / 今日 TAB + 严格一买 两个 TAB。

展示严格三买与严格一买各自独立(不再"无三买才递补一买")。
排序: 收跌优先 → 688/300 > 00 > 60。
买入过滤: 涨停/跌超5% 不买。退出规则: 次日收盘卖(涨停顺延)。

用法: .venv/Scripts/streamlit run chan_monitor.py
"""
import sys
import os

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # exe 目录
from chan_signal_api import recent_chan_signals_both, check_freshness
from refresh_market_data import refresh_market_data

st.set_page_config(page_title="缠论买点监控", page_icon="🎯", layout="wide")

st.title("🎯 缠论买点监控")

# 数据新鲜度
last_date, fresh_msg = check_freshness()
if "⚠️" in fresh_msg or "过期" in fresh_msg:
    st.error(fresh_msg)
else:
    st.info(fresh_msg)

with st.sidebar:
    st.header("口径")
    st.markdown(
        "- **严格三买** (canonical 线段级)\n"
        "- **严格一买** (canonical 线段级, 独立 TAB)\n"
        "- 排序: **收跌优先 → 688/300 > 00 > 60**\n"
        "- 买入过滤: 涨停 / 跌超5% 不买\n"
        "- 退出: 次日收盘卖 (涨停顺延)"
    )
    st.markdown("---")
    refresh = st.button("🔄 重新扫描全市场", use_container_width=True,
                        help="① 清盘中半截bar → ② 增量补全到最新收盘 → ③ 重扫全市场 (约 3-8 分钟)")
    st.caption("① 清盘中缓存 → ② 补全日线 → ③ 重扫")

# 点「重新扫描全市场」: 先补齐数据, 再交给下面的 recent_chan_signals_both(force=True) 重扫
if refresh:
    with st.status("① 清盘中缓存 → ② 补全日线数据 ...", expanded=True) as _status:
        _log_box = st.empty()
        _buf = []

        def _on_log(line):
            _buf.append(line)
            _log_box.code("\n".join(_buf[-25:]), language="text")   # 只留末尾 25 行

        _ok, _msg = refresh_market_data(on_log=_on_log)
        _status.update(label=("✅ " if _ok else "❌ ") + _msg,
                       state="complete" if _ok else "error", expanded=not _ok)

    if _ok:
        st.success("数据已补全 → 开始重扫全市场")
    else:
        st.error(f"数据更新未完成: {_msg} —— 本次仍按现有缓存扫描")

DISPLAY_COLS = {
    "code": "代码", "name": "名称", "kind_label": "买点", "buy_date": "买入日",
    "close": "买入价", "today_ret": "今日涨跌%", "is_down_label": "收跌/收涨",
    "bucket": "状态", "board": "板块", "zg_ext": "zg_ext%", "depth": "depth%",
    "gap": "gap(日)", "ratio": "背驰比",
}
ORDER = ["代码", "名称", "买点", "买入日", "买入价", "今日涨跌%", "收跌/收涨",
         "状态", "板块", "zg_ext%", "depth%", "gap(日)", "背驰比"]


def xq_url(code):
    """6位代码 → 雪球个股页 URL (https://xueqiu.com/S/SH600549)。"""
    code = str(code)
    if code.startswith("6"):
        pre = "SH"
    elif code.startswith(("9", "8", "4")):
        pre = "BJ"
    else:
        pre = "SZ"
    return f"https://xueqiu.com/S/{pre}{code}"


def show_table(df, height=600):
    disp = df.rename(columns=DISPLAY_COLS)
    disp = disp[[c for c in ORDER if c in disp.columns]]
    # 去掉全空列: 一买无 zg_ext/depth/gap, 三买无背驰比
    disp = disp[[c for c in disp.columns if not disp[c].isna().all()]]
    # 代码列 → 可点击的雪球链接
    disp["代码"] = disp["代码"].astype(str).map(xq_url)
    st.dataframe(
        disp,
        use_container_width=True,
        height=height,
        column_config={
            "代码": st.column_config.LinkColumn(
                "代码", display_text=r"(\d{6})", help="点击打开雪球",
            ),
        },
    )


with st.spinner("扫描近 5 日信号 ... (首次约 1 分钟)"):
    df3, df1 = recent_chan_signals_both(days=5, refresh=refresh)

tab5, tab_next, tab1, tab_b1_now, tab_b1_next = st.tabs(
    ["📅 近五日", "🛒 明日可买", "📍 今日", "🎯 一买·今日", "🎯 一买·明日可买"])

with tab5:
    if df3.empty and not df1.empty:
        st.warning("⚠️ 近 5 日无严格三买 → 已递补为**严格一买**")
        df5, primary = df1, "严格一买(递补)"
    else:
        df5, primary = df3, "严格三买"
    st.caption(
        f"主口径: **{primary}** · 共 **{len(df5)}** 笔 · 排序: 收跌优先 → 688/300 > 00 > 60")
    if df5.empty:
        st.info("暂无信号")
    else:
        show_table(df5)

with tab_next:
    st.caption("严格三买 · 结构已确认、下一交易日收盘可买（exec = 末根bar 之后）")
    nxt = df3[df3["bucket"] == "明日可买"]
    if nxt.empty:
        st.info("暂无「明日可买」三买信号")
    else:
        show_table(nxt, height=400)

with tab1:
    st.caption("严格三买 · 仅展示「今日收盘触发」的信号 (exec = 末根bar)")
    today = df3[df3["bucket"] == "今日"]
    if today.empty:
        st.info("今日无「今日收盘触发」的三买")
    else:
        show_table(today, height=400)

with tab_b1_now:
    st.caption("严格一买 · 仅展示「今日收盘触发」的信号 (exec = 末根bar)")
    b1_now = df1[df1["bucket"] == "今日"]
    if b1_now.empty:
        st.info("今日无「今日收盘触发」的严格一买")
    else:
        show_table(b1_now, height=400)

with tab_b1_next:
    st.caption("严格一买 · 结构已确认、下一交易日收盘可买（exec = 末根bar 之后）")
    b1_next = df1[df1["bucket"] == "明日可买"]
    if b1_next.empty:
        st.info("暂无「明日可买」严格一买信号")
    else:
        show_table(b1_next, height=400)

st.markdown("---")
st.caption("严格三买 · K=2 · 收跌优先 → 688/300 > 00 > 60 · 收盘买(涨停/跌超5%不买) · "
           "次日收盘卖(涨停顺延) · 槽空就补 —— 年化 156.6% / 回撤 10.9%")
