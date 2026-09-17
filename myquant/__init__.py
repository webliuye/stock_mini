"""MyQuant - 个人量化交易系统

一个模块化、从想法到实盘的个人量化交易系统。
核心模块:
    - data: 数据获取与存储 (akshare, SQLite)
    - strategy: 策略开发 (双均线, 均值回归等)
    - backtest: 回测引擎 (信号生成→模拟交易→绩效分析)
    - execution: 交易执行 (模拟盘/实盘)
    - risk: 风控管理 (事前/事中/事后)
    - monitor: 系统监控 (日报/告警)
    - utils: 工具函数 (日志/配置/计算)
"""

__version__ = "1.0.0"
__author__ = "MyQuant Team"