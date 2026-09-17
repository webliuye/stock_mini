#!/usr/bin/env bash
# 缠论买点监控 - macOS 启动脚本 (对应 Windows 的 exe/start.bat)
cd "$(dirname "$0")"          # 切到 stock_mini 根目录

echo "============================================"
echo "  缠论买点监控 - 启动"
echo "============================================"
echo
echo "  [1/3] 刷新数据 (清盘中半截bar + 增量补全到最新收盘) ..."
.venv/bin/python refresh_market_data.py || echo "  [警告] 数据刷新失败, 仍按现有缓存继续"

echo
echo "  [2/3] 清信号缓存 ..."
rm -f reports/chan_recent_signals.pkl

echo
echo "  [3/3] 启动页面 ..."
echo "  启动完成后浏览器手动打开 http://localhost:8501"
echo "  停止: 运行 ./stop.sh 或直接关闭终端"
echo
.venv/bin/streamlit run exe/chan_monitor.py --server.headless false
