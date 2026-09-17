#!/usr/bin/env bash
# 缠论买点监控 - 停止脚本 (杀 8501 端口)
echo "停止缠论买点监控 (端口 8501) ..."
PIDS=$(lsof -ti :8501 2>/dev/null)
if [ -n "$PIDS" ]; then
  kill $PIDS 2>/dev/null && echo "已停止"
else
  echo "没有运行中的 8501 进程"
fi
