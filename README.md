# stock_mini — 缠论买点监控（精简迁移版）

从 `stock` 项目迁移出的「严格三买 / 严格一买」展示页 + 信号 API，最小化到可在 macOS 上独立运行。
**不包含**股票行情缓存（`data/cache/*.parquet`），首次运行需从数据源全量拉取。

## 目录结构
```
stock_mini/
├── exe/chan_monitor.py        # Streamlit 展示页（严格三买 + 严格一买 5 个 TAB）
├── exe/chan_signal_api.py     # 信号 API（全市场 scan_points + 缓存）
├── bt_chan_backtest.py        # 结构构造 / 缓存加载（chan_signal_api 依赖）
├── realtime_snapshot.py       # 盘中实时 bar（AlphaFeed）
├── refresh_market_data.py     # 数据刷新流水线（清盘中bar + 增量补数）
├── _truncate_cache_today.py   # 清盘中半截 bar
├── update_daily_data.py       # 增量补全日线
├── myquant/                   # 精简后的 myquant（仅 chan 策略 + 引擎 + 数据源）
├── data/all_stocks.csv        # 股票名称映射
├── data/all_stocks.parquet    # 股票列表
├── config.yaml                # 数据源 / API key 配置
├── requirements.txt
├── start.sh / stop.sh
```

## macOS 环境搭建
```bash
cd stock_mini
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置数据源
`config.yaml` 已带 AlphaFeed 的 API key（`data.alphafeed_api_key`）。
- 沿用 **AlphaFeed**（付费）：无需改动；`alphafeed` 是私有 pip 包，请确认能 `pip install alphafeed`。
- 改用**免费 akshare**：把 `data.source` 改为 `akshare`，并 `pip install akshare`。

## 首次拉取数据（必须）
`data/cache` 为空，先全量拉取历史日线（2018-01-01 起，约 5000 只，耗时较长）：
```bash
source .venv/bin/activate
.venv/bin/python update_daily_data.py --yes
```

## 启动 / 停止
```bash
chmod +x start.sh stop.sh
./start.sh      # 自动: 刷新数据 → 清信号缓存 → 起页面
# 浏览器打开 http://localhost:8501
./stop.sh
```

## 说明
- 页面首次打开会全市场扫描（约 1 分钟起），信号缓存于 `reports/chan_recent_signals.pkl`。
- 「重新扫描全市场」按钮会先清盘中半截 bar 再增量补数，需联网 + 有效数据源。
