"""更新所有股票的每日数据到最新

功能:
1. 扫描 data/cache 目录，查看每只股票缓存数据的最新日期
2. 对比当前日期，增量拉取缺失的每日数据
3. 合并旧数据，更新缓存文件
4. 清理旧的缓存文件
"""
import re
import sys
import time
import threading
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

sys.path.insert(0, ".")

from myquant.data.datasource import DataSource

# 缓存目录
CACHE_DIR = Path("data/cache")

# 缓存文件名正则: {code}_{start}_{end}_{adjust}.parquet
CACHE_PATTERN = re.compile(r"^(\d{6})_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(.+?)\.parquet$")

# AlphaFeed 单请求 symbol 上限。库内默认 100, 但服务端实测放到 200 仍接受
# (300 报 "标的数量超限: 300 (最大: 200)") —— 用它可以把请求数直接减半。
MAX_SYMBOLS_PER_REQUEST = 200

# 服务端限流: 实测 429 消息为 "Rate limit exceeded (60/min)"。取 55 留余量,
# 令牌桶按这个速率放行, 保证不会因为并发 burst 而丢 chunk。
REQUESTS_PER_MINUTE = 55

# 并发路数。真正的节流由令牌桶负责, 这里只要够把管道填满即可。
BATCH_WORKERS = 6


class _RateLimiter:
    """令牌桶限速器(线程安全)。

    按固定间隔放行请求, 使客户端整体速率不超过 per_minute。用来卡在
    AlphaFeed 的 60/min 硬限之下 —— 光靠调低并发并不能解决问题: 并发 12 时
    53 个请求 19 秒打完(≈165/min), 直接超限丢掉 12 个 chunk。
    """

    def __init__(self, per_minute: int):
        self._interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next_at = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._next_at > now:
                time.sleep(self._next_at - now)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self._interval


def _install_rate_limit(api, per_minute: int = REQUESTS_PER_MINUTE) -> None:
    """给 AlphaFeed 客户端的 get() 套上限速, 覆盖库内部所有并发请求。

    库的 klines.batch 用线程池并发取数, 从外面没法逐个请求控制节奏, 所以
    直接包住底层 client.get —— 无论库内起多少线程, 请求都按令牌桶放行。
    顺带也限住了 instrument 名称解析的请求。
    """
    client = api._client
    original_get = client.get
    limiter = _RateLimiter(per_minute)

    def paced_get(*args, **kwargs):
        limiter.acquire()
        return original_get(*args, **kwargs)

    client.get = paced_get


def parse_cache_filename(filename: str) -> Optional[Dict]:
    """解析缓存文件名，提取 code, start_date, end_date, adjust"""
    match = CACHE_PATTERN.match(filename)
    if not match:
        return None
    return {
        "code": match.group(1),
        "start_date": match.group(2),
        "end_date": match.group(3),
        "adjust": match.group(4),
    }


def scan_cache() -> Dict[str, Dict]:
    """扫描缓存目录，返回每只股票的最新缓存信息
    
    Returns:
        {code: {"start_date": str, "end_date": str, "adjust": str, "filepath": Path}}
    """
    cache_info: Dict[str, Dict] = {}
    
    for fpath in CACHE_DIR.glob("*.parquet"):
        info = parse_cache_filename(fpath.name)
        if info is None:
            continue
        
        code = info["code"]
        end_date = info["end_date"]
        
        # 保留每只股票 end_date 最新的那个缓存文件
        if code not in cache_info or end_date > cache_info[code]["end_date"]:
            cache_info[code] = {
                "start_date": info["start_date"],
                "end_date": info["end_date"],
                "adjust": info["adjust"],
                "filepath": fpath,
            }
    
    return cache_info


def get_stock_list() -> pd.DataFrame:
    """从 all_stocks.parquet 读取股票列表"""
    stocks_path = Path("data/all_stocks.parquet")
    if not stocks_path.exists():
        logger.error("data/all_stocks.parquet 不存在，请先运行 get_all_stocks.py")
        return pd.DataFrame()
    return pd.read_parquet(stocks_path)


def to_af_symbol(code: str) -> str:
    """6位代码转 AlphaFeed 格式 (600XXX.SH / 00XXXX.SZ / 30XXXX.SZ / 920XXX.BJ)"""
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith("9") or code.startswith("8") or code.startswith("4"):
        return f"{code}.BJ"
    return f"{code}.SZ"


def calculate_count(fetch_start: str, today_str: str) -> int:
    """估算需要的 K 线数量（AlphaFeed 按 count 返回最新 N 条，需 buffer）。

    请求耗时基本由返回的数据量决定（实测 100 只/次: count=5 → 1.1s,
    count=300 → 1.4s, count=2800 → 3.8s），所以增量更新时没必要为"保险"
    多要几百条 —— 只要 count ≥ 区间内的交易日数, 多要的全是浪费。
    """
    start_dt = datetime.strptime(fetch_start, "%Y-%m-%d")
    end_dt = datetime.strptime(today_str, "%Y-%m-%d")
    diff_days = (end_dt - start_dt).days
    # A股交易日约为自然日的 0.68 (243/365)，取 0.8 留余量；+30 覆盖长假/停牌
    return min(max(int(diff_days * 0.8) + 30, 30), 10000)


def _parse_retry_seconds(err: Exception) -> Optional[float]:
    """从 RateLimitError 消息抓 "Retry after ~16000-24000ms" 的最短等待秒数."""
    import re as _re
    m = _re.search(r"Retry after\s*~?(\d+)(?:\s*-\s*\d+)?\s*ms", str(err))
    return int(m.group(1)) / 1000.0 if m else None


def _fetch_symbol_set(
    api, symbols: List[str], count: int, max_workers: int, label: str
) -> Dict[str, pd.DataFrame]:
    """取一批 symbol, 返回 {symbol: df}。

    库内 klines.batch 自己按 MAX_SYMBOLS_PER_REQUEST 切块 + 线程池并发, 且已对
    429/5xx 做 3 次指数退避重试; 这里只兜底整批彻底失败(返回空 dict)。

    注意: 库内某个 chunk 失败时是**静默吞掉**的 —— 那些 symbol 不会出现在返回
    的 dict 里, 所以调用方必须自己比对缺失。
    """
    from alphafeed import RateLimitError as _RateLimitError

    for attempt in range(3):
        try:
            return api.klines.batch(
                symbols,
                period="1d",
                count=count,
                adjust="forward",
                to_dataframe=True,
                batch_size=MAX_SYMBOLS_PER_REQUEST,
                max_workers=max_workers,
            )
        except _RateLimitError as e:
            if attempt >= 2:
                print(f"    {label} 限流放弃: {e}", flush=True)
                return {}
            wait = _parse_retry_seconds(e) or 20.0
            print(f"    {label} 限流, 冷却 {wait:.0f}s 后重试 (外层 {attempt+1}/2)", flush=True)
            time.sleep(wait + 2.0)
        except Exception as e:
            print(f"    {label} 失败: {type(e).__name__}: {e}", flush=True)
            return {}
    return {}


def _standardize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """trade_date -> date, 只留标准列（与单只 get() 的口径一致）。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["trade_date"])
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("date").reset_index(drop=True)


def batch_fetch(
    api,
    plan: Dict[str, Dict],
    today_str: str,
    max_workers: int = BATCH_WORKERS,
) -> Dict[str, Tuple[Optional[pd.DataFrame], str]]:
    """批量拉取 K 线数据。

    按 fetch_start 分组后整组交给 AlphaFeed 的 klines.batch() —— 该接口内部会
    切块并用线程池并发取。两个要点:

    1. 不要自己切 100 只再逐批调用。传入正好 100 只时库内只切出 1 个 chunk, 会走
       `len(chunks) == 1` 的串行分支, 并发完全用不上 —— 之前就是 53 批纯串行 +
       批间 sleep 2.1s, 光 sleep 就 110 秒。
    2. 但也不能只靠加大并发。服务端限流是 60 请求/分钟(实测 429 消息
       "Rate limit exceeded (60/min)"), 并发 12 时 53 个请求 19 秒打完 ≈165/min,
       直接超限丢掉 12 个 chunk(400 只股票)。所以用令牌桶把整体速率压在
       REQUESTS_PER_MINUTE 之下 —— 全市场增量最少也要 5204/200 ≈ 27 个请求,
       即 ~30 秒, 这是硬下限。

    按 fetch_start 分组还有个好处: 同组共用一个 count, 不会因为批内混进一只
    无缓存的股票 (fetch_start=2018-01-01) 就把整批 100 只的 count 抬到 2800。

    Args:
        api: AlphaFeed 客户端实例
        plan: 拉取计划 {code: {"old_info": ..., "fetch_start": str}}
        today_str: 今天的日期字符串
        max_workers: 并发请求数（节流由令牌桶负责, 这里只决定管道深度）

    Returns:
        {code: (df, symbol)}  df 为 None 表示该股票拉取失败
    """
    results: Dict[str, Tuple[Optional[pd.DataFrame], str]] = {}

    # 按增量起点分组: 同组 fetch_start 一致, 共用一个 count
    groups: Dict[str, list] = defaultdict(list)
    for code, item in plan.items():
        groups[item["fetch_start"]].append(code)

    if not groups:
        return results

    _install_rate_limit(api)
    print(f"  按增量起点分 {len(groups)} 组, "
          f"{MAX_SYMBOLS_PER_REQUEST} 只/请求, 并发 {max_workers} 路, "
          f"限速 {REQUESTS_PER_MINUTE}/min", flush=True)

    for gi, (fetch_start, codes) in enumerate(sorted(groups.items()), 1):
        symbols = [to_af_symbol(c) for c in codes]
        symbol_to_code = {s: c for c, s in zip(codes, symbols)}
        count = calculate_count(fetch_start, today_str)
        label = f"组 {gi}/{len(groups)}"
        n_req = (len(symbols) + MAX_SYMBOLS_PER_REQUEST - 1) // MAX_SYMBOLS_PER_REQUEST

        t0 = time.time()
        dfs_map = _fetch_symbol_set(api, symbols, count, max_workers, label)

        # 库内失败的 chunk 是静默丢掉的, 这里把缺失的 symbol 捞出来补一次。
        # 限流是概率性的, 重取一趟基本都能拿到, 比直接判失败划算。
        missing = [s for s in symbols if s not in dfs_map]
        if missing:
            print(f"    {label} 缺 {len(missing)} 只, 补取一次 ...", flush=True)
            retry_map = _fetch_symbol_set(api, missing, count, max_workers, f"{label} 补取")
            dfs_map.update(retry_map)

        # dfs_map: {symbol: DataFrame}; 仍未出现的按失败处理
        for symbol, df in dfs_map.items():
            code = symbol_to_code.get(symbol)
            if code is None or df is None or df.empty:
                continue
            results[code] = (_standardize_kline_df(df), symbol)

        done_symbols = set(dfs_map.keys())
        for symbol, code in symbol_to_code.items():
            if symbol not in done_symbols:
                results[code] = (None, symbol)

        print(f"    {label} (起点 {fetch_start}, count={count}): {len(codes)} 只, "
              f"成功 {len(done_symbols)} 只, {n_req} 个请求, "
              f"耗时 {time.time()-t0:.1f}s", flush=True)

    return results


def merge_and_save(
    code: str,
    old_info: Optional[Dict],
    new_df: pd.DataFrame,
    today_str: str,
) -> Tuple[bool, str]:
    """合并新旧数据并保存到缓存文件。

    Args:
        code: 6位股票代码
        old_info: 旧的缓存信息（可能为None，表示无缓存）
        new_df: 新拉取的增量数据
        today_str: 今天的日期字符串

    Returns:
        (success, message)
    """
    try:
        if old_info is not None:
            adjust = old_info["adjust"]
            old_filepath = old_info["filepath"]
            try:
                old_df = pd.read_parquet(old_filepath)
            except Exception as e:
                logger.warning(f"  读取旧缓存失败 {code}: {e}，仅保存新数据")
                old_df = pd.DataFrame()
        else:
            adjust = "qfq"
            old_df = pd.DataFrame()

        if old_df is None or old_df.empty:
            merged = new_df.copy()
        else:
            # 确保列一致
            common_cols = [c for c in old_df.columns if c in new_df.columns]
            if "date" not in common_cols:
                common_cols = ["date"] + common_cols
            old_part = old_df[common_cols].copy()
            new_part = new_df[common_cols].copy()
            merged = pd.concat([old_part, new_part], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"])
            merged = merged.drop_duplicates(subset=["date"], keep="last")
            merged = merged.sort_values("date").reset_index(drop=True)

        if merged.empty:
            return True, "无数据"

        # 保存合并后的数据为新缓存文件（实际日期范围）
        actual_start = merged["date"].min().strftime("%Y-%m-%d")
        actual_end = merged["date"].max().strftime("%Y-%m-%d")
        new_cache_file = CACHE_DIR / f"{code}_{actual_start}_{actual_end}_{adjust}.parquet"
        merged.to_parquet(new_cache_file)

        # 删除该股票所有旧缓存文件（除新文件外）
        for old_f in CACHE_DIR.glob(f"{code}_*.parquet"):
            if old_f != new_cache_file:
                try:
                    old_f.unlink()
                except Exception:
                    pass

        new_rows = len(new_df)
        total_rows = len(merged)
        return True, f"新增 {new_rows} 条 → 共 {total_rows} 条 ({actual_start} ~ {actual_end})"

    except Exception as e:
        return False, f"失败: {type(e).__name__}: {e}"


def update_stock_data(
    ds: DataSource,
    code: str,
    old_info: Optional[Dict],
    today_str: str,
) -> Tuple[bool, str]:
    """更新单只股票的每日数据
    
    Args:
        ds: DataSource 实例
        code: 6位股票代码
        old_info: 旧的缓存信息（可能为None，表示无缓存）
        today_str: 今天的日期字符串 "YYYY-MM-DD"
    
    Returns:
        (success, message)
    """
    try:
        if old_info is not None:
            # 有旧缓存：读取旧数据，只拉取增量
            old_end_date = old_info["end_date"]
            start_date = old_info["start_date"]
            adjust = old_info["adjust"]
            old_filepath = old_info["filepath"]
            
            # 如果缓存已经是最新，跳过
            if old_end_date >= today_str:
                return True, f"已是最新 (缓存截止 {old_end_date})"
            
            # 增量拉取：从旧数据最后一天的下一天开始
            old_end_dt = datetime.strptime(old_end_date, "%Y-%m-%d")
            fetch_start = (old_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # 读取旧数据
            try:
                old_df = pd.read_parquet(old_filepath)
            except Exception as e:
                logger.warning(f"  读取旧缓存失败 {code}: {e}，将全量拉取")
                old_df = pd.DataFrame()
                fetch_start = start_date
        else:
            # 无缓存：全量拉取（与全量重拉起点 2018-01-01 对齐）
            start_date = "2018-01-01"
            adjust = "qfq"
            old_df = pd.DataFrame()
            fetch_start = start_date
        
        # 如果 fetch_start > today_str，说明不需要拉取
        if fetch_start > today_str:
            return True, "无需更新"
        
        # 调用 DataSource 获取数据（不读缓存，强制从 API 拉取）
        # 注意：由于 DataSource.get_daily_bars 会先查缓存，而我们要的是增量数据，
        # 缓存中不存在 {code}_{fetch_start}_{today_str}_{adjust}.parquet，
        # 所以它会自动走 API 调用
        new_df = ds.get_daily_bars(
            symbol=code,
            start_date=fetch_start,
            end_date=today_str,
            adjust=adjust,
            use_cache=True,  # 缓存会保存为新的文件名
        )
        
        if new_df is None or new_df.empty:
            return True, f"增量区间 {fetch_start}~{today_str} 无新数据（可能非交易日）"
        
        # 合并新旧数据
        if not old_df.empty:
            # 确保列一致
            common_cols = [c for c in old_df.columns if c in new_df.columns]
            if "date" not in common_cols:
                common_cols = ["date"] + common_cols
            old_part = old_df[common_cols].copy()
            new_part = new_df[common_cols].copy()
            merged = pd.concat([old_part, new_part], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"])
            merged = merged.drop_duplicates(subset=["date"], keep="last")
            merged = merged.sort_values("date").reset_index(drop=True)
        else:
            merged = new_df.copy()
        
        # 保存合并后的数据为新缓存文件
        # 直接用数据中的实际日期范围
        actual_start = merged["date"].min().strftime("%Y-%m-%d")
        actual_end = merged["date"].max().strftime("%Y-%m-%d")
        new_cache_file = CACHE_DIR / f"{code}_{actual_start}_{actual_end}_{adjust}.parquet"
        merged.to_parquet(new_cache_file)
        
        # 删除旧缓存文件（如果和新文件名不同）
        if old_info is not None:
            old_path = old_info["filepath"]
            if old_path != new_cache_file:
                # 同时删除该股票所有旧的缓存文件
                for old_f in CACHE_DIR.glob(f"{code}_*.parquet"):
                    if old_f != new_cache_file:
                        try:
                            old_f.unlink()
                        except Exception:
                            pass
        
        new_rows = len(new_df)
        total_rows = len(merged)
        return True, f"新增 {new_rows} 条 → 共 {total_rows} 条 ({actual_start} ~ {actual_end})"
    
    except Exception as e:
        return False, f"失败: {type(e).__name__}: {e}"


def main(auto_confirm: bool = False):
    """auto_confirm=True 时跳过「>100 只需确认」的交互提问(供程序调用/--yes)。"""
    print("=" * 70)
    print("  股票每日数据更新工具")
    print(f"  当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 加载配置
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_config = config.get("data", {})
    ds = DataSource(data_config)
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 2. 扫描缓存，查看数据更新状态
    print("\n📂 扫描缓存目录...")
    cache_info = scan_cache()
    
    if cache_info:
        end_dates = sorted(set(info["end_date"] for info in cache_info.values()))
        min_date = end_dates[0]
        max_date = end_dates[-1]
        print(f"  缓存股票数: {len(cache_info)} 只")
        print(f"  数据日期范围: {min_date} ~ {max_date}")
        
        if max_date < today_str:
            print(f"  ⚠️ 最新缓存日期 {max_date}，落后于今天 {today_str}，需要更新")
        else:
            print(f"  ✅ 数据已是最新")
        
        # 按日期分组统计
        date_counts = Counter(info["end_date"] for info in cache_info.values())
        print(f"\n  各截止日期股票数量:")
        for d in sorted(date_counts.keys()):
            bar = "█" * min(date_counts[d] // 10, 50)
            print(f"    {d}: {date_counts[d]:>5} 只 {bar}")
    else:
        print("  ⚠️ 缓存目录为空，将全量拉取")
    
    # 3. 获取股票列表
    print("\n📋 读取股票列表...")
    stocks_df = get_stock_list()
    if stocks_df.empty:
        print("  错误: 无法读取股票列表")
        return
    
    codes = stocks_df["代码"].tolist()
    print(f"  共 {len(codes)} 只股票待处理")
    
    # 4. 确认是否继续
    need_update_count = sum(
        1 for code in codes
        if code not in cache_info or cache_info[code]["end_date"] < today_str
    )
    
    if need_update_count == 0:
        print(f"\n✅ 所有股票数据已是最新，无需更新")
        return
    
    print(f"\n🔄 需要更新 {need_update_count} 只股票的数据")
    print(f"   批量拉取 ({MAX_SYMBOLS_PER_REQUEST} 只/请求, 并发 {BATCH_WORKERS} 路, "
          f"限速 {REQUESTS_PER_MINUTE}/min —— 服务端硬限 60/min)")
    
    # 如果有很多需要更新，给个提示
    if need_update_count > 100:
        if auto_confirm:
            print(f"\n  --yes: 自动确认更新 {need_update_count} 只股票")
        else:
            resp = input(f"\n  将更新 {need_update_count} 只股票，是否继续？(y/n): ")
            if resp.lower() != 'y':
                print("  已取消")
                return
    
    # 5. 构建拉取计划（仅需要更新的股票）
    plan: Dict[str, Dict] = {}
    for code in codes:
        old_info = cache_info.get(code)
        # 已是最新的跳过
        if old_info and old_info["end_date"] >= today_str:
            continue
        
        if old_info is not None:
            old_end_dt = datetime.strptime(old_info["end_date"], "%Y-%m-%d")
            fetch_start = (old_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            if fetch_start > today_str:
                continue
        else:
            fetch_start = "2018-01-01"

        plan[code] = {
            "old_info": old_info,
            "fetch_start": fetch_start,
        }
    
    print(f"\n📥 开始批量拉取数据...")
    t0 = time.time()
    
    # 6. 获取 API 实例，批量拉取
    api = ds._get_api()
    fetch_results = batch_fetch(api, plan, today_str)
    
    print(f"\n💾 开始合并并保存缓存...")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    fail_list = []
    
    total_to_process = len(fetch_results)
    for idx, (code, (df, symbol)) in enumerate(fetch_results.items(), 1):
        if idx % 100 == 0 or idx == total_to_process:
            print(f"  进度: {idx}/{total_to_process} | "
                  f"成功 {success_count} | 失败 {fail_count} | 跳过 {skip_count}")
        
        if df is None or df.empty:
            fail_count += 1
            fail_list.append((code, f"批量拉取无数据 ({symbol})"))
            continue
        
        # 只保留增量区间数据
        item = plan[code]
        df = df[(df["date"] >= item["fetch_start"]) & (df["date"] <= today_str)].copy()
        
        if df.empty:
            skip_count += 1
            continue
        
        ok, msg = merge_and_save(code, item["old_info"], df, today_str)
        
        if ok:
            if "无数据" in msg:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            fail_list.append((code, msg))
            logger.error(f"  ❌ {code}: {msg}")
    
    elapsed = time.time() - t0
    
    # 7. 输出结果
    print(f"\n{'='*70}")
    print(f"  更新完成 (耗时 {elapsed:.1f} 秒)")
    print(f"  成功: {success_count} 只")
    print(f"  跳过(已最新): {skip_count} 只")
    print(f"  失败: {fail_count} 只")
    print(f"{'='*70}")
    
    if fail_list:
        print(f"\n  失败列表:")
        for code, msg in fail_list[:20]:
            print(f"    {code}: {msg}")
        if len(fail_list) > 20:
            print(f"    ... 共 {len(fail_list)} 只失败")
    
    # 8. 重新扫描缓存，显示最终状态
    print(f"\n📊 最终缓存状态:")
    final_cache = scan_cache()
    if final_cache:
        end_dates = sorted(set(info["end_date"] for info in final_cache.values()))
        print(f"  缓存股票数: {len(final_cache)} 只")
        print(f"  数据日期范围: {end_dates[0]} ~ {end_dates[-1]}")


if __name__ == "__main__":
    main(auto_confirm="--yes" in sys.argv)