# -*- coding: utf-8 -*-
"""市场数据刷新流水线: 先清盘中半截bar, 再增量补全到最新收盘。

步骤 (串行, 任一步失败即中止):
  1) _truncate_cache_today.py   — 删掉 data/cache 里 date==今天 的盘中行并重写文件名,
     让 update_daily_data 判定缓存落后一天, 从而**重拉**今天的完整收盘 bar。
     (盘中跑过 update 时缓存里会留半截 bar, 而 append_today_bar 只在 live.date >
     缓存末日时才追加 → 不清就永远补不回来, 扫描一直用着半截 bar)
  2) update_daily_data.py --yes — 按缓存末日增量拉取, 合并写回 data/cache。

被 chan_monitor.py 的「重新扫描全市场」按钮调用; 也可命令行直跑:
  .venv/Scripts/python refresh_market_data.py

用法:
  from refresh_market_data import refresh_market_data
  ok, msg = refresh_market_data(on_log=print)
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 子进程用的解释器: 优先项目 .venv, 退化到当前解释器 (Windows/macOS 路径自适应)
PYTHON = (PROJECT_ROOT / ".venv"
          / ("Scripts" if os.name == "nt" else "bin")
          / ("python.exe" if os.name == "nt" else "python"))

# (脚本名, 额外参数) —— 顺序即执行顺序
STEPS = [
    ("_truncate_cache_today.py", []),
    ("update_daily_data.py", ["--yes"]),
]


def _python_exe() -> str:
    if PYTHON.exists():
        return str(PYTHON)
    return sys.executable or "python"


def refresh_market_data(on_log=None):
    """跑「清盘中缓存 → 增量补数」流水线。

    Args:
        on_log: 可选回调 on_log(line: str), 逐行收到子进程输出(已去行尾符)。

    Returns:
        (ok, 说明)。ok=False 时说明含失败原因, 且流水线已中止。
    """
    def log(s: str = ""):
        if on_log:
            on_log(s)

    py = _python_exe()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")   # 两个脚本自身也 reconfigure 到 utf-8

    for name, extra in STEPS:
        script = PROJECT_ROOT / name
        if not script.exists():
            msg = f"缺少脚本 {name}"
            log("✗ " + msg)
            return False, msg

        log(f"$ {Path(py).name} {name} {' '.join(extra)}".rstrip())
        try:
            proc = subprocess.Popen(
                [py, str(script), *extra],
                cwd=str(PROJECT_ROOT),          # 脚本内用的是 data/cache、config.yaml 相对路径
                stdin=subprocess.DEVNULL,       # 非交互; update_daily_data 靠 --yes 免确认
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=env,
            )
        except Exception as e:
            msg = f"启动 {name} 失败: {type(e).__name__}: {e}"
            log("✗ " + msg)
            return False, msg

        for line in proc.stdout:
            log(line.rstrip("\n"))
        rc = proc.wait()

        if rc != 0:
            msg = f"{name} 退出码 {rc}"
            log("✗ " + msg)
            return False, msg

    return True, "缓存已补全到最新收盘 bar"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok, msg = refresh_market_data(on_log=print)
    print(("\n✅ " if ok else "\n❌ ") + msg)
    sys.exit(0 if ok else 1)
