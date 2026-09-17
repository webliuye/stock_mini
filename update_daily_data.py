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
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
from loguru import logger

sys.path.insert(0, ".")

from myquant.data.datasource import DataSource

# 缓存目录
CACHE_DIR = Path("data/cache")

# 缓存文件名正则: {code}_{start}_{end}_{adjust}.parquet
CACHE_PATTERN = re.compile(r"^(\d{6})_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(.+?)\.parquet$")

# 批量请求大小（AlphaFeed batch 接口上限 100）
BATCH_SIZE = 100


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
    """估算需要的 K 线数量（AlphaFeed 按 count 返回最新 N 条，需 buffer）"""
    start_dt = datetime.strptime(fetch_start, "%Y-%m-%d")
    end_dt = datetime.strptime(today_str, "%Y-%m-%d")
    diff_days = (end_dt - start_dt).days
    # 交易日约为自然日 0.7，留 250 条 buffer
    return min(max(int(diff_days * 0.8) + 250, 300), 10000)


def _parse_retry_seconds(err: Exception) -> Optional[float]:
    """从 RateLimitError 消息抓 "Retry after ~16000-24000ms" 的最短等待秒数."""
    import re as _re
    m = _re.search(r"Retry after\s*~?(\d+)(?:\s*-\s*\d+)?\s*ms", str(err))
    return int(m.group(1)) / 1000.0 if m else None


def batch_fetch(
    api,
    plan: Dict[str, Dict],
    today_str: str,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Tuple[Optional[pd.DataFrame], str]]:
    """批量拉取 K 线数据。

    使用 AlphaFeed 的 klines.batch()，每次最多 batch_size 只并发请求，
    大幅减少网络往返次数。AlphaFeed 批次接口限流实测 30/min，客户端内置
    3 次退避重试仍 429 后抛出 RateLimitError —— 这里每批之间留 ~2.1s
    (≈28/min 不触顶) 作限速，撞上 429 时再按 Retry-after 冷区外层重试。

    Args:
        api: AlphaFeed 客户端实例
        plan: 拉取计划 {code: {"old_info": ..., "fetch_start": str}}
        today_str: 今天的日期字符串
        batch_size: 每批股票数

    Returns:
        {code: (df, symbol)}  df 为 None 表示该股票拉取失败
    """
    from alphafeed import RateLimitError as _RateLimitError

    results: Dict[str, Tuple[Optional[pd.DataFrame], str]] = {}

    codes = list(plan.keys())
    total_batches = (len(codes) + batch_size - 1) // batch_size

    for bi in range(total_batches):
        chunk = codes[bi * batch_size : (bi + 1) * batch_size]
        symbols = [to_af_symbol(c) for c in chunk]
        symbol_to_code = {s: c for c, s in zip(chunk, symbols)}

        # 取该批次最早的 fetch_start，保证覆盖所有股票的增量区间
        fetch_start = min(plan[c]["fetch_start"] for c in chunk)
        count = calculate_count(fetch_start, today_str)

        # 批间限速: 30/min 限制留余量
        if bi > 0:
            time.sleep(2.1)

        dfs_map, got = None, False
        for attempt in range(6):
            try:
                dfs_map = api.klines.batch(
                    symbols,
                    period="1d",
                    count=count,
                    adjust="forward",
                    to_dataframe=True,
                )
                got = True
                break
            except _RateLimitError as e:
                if attempt >= 5:
                    print(f"    批次 {bi+1}/{total_batches} 限流放弃: {e}", flush=True)
                    break
                wait = _parse_retry_seconds(e) or 20.0
                print(f"    批次 {bi+1}/{total_batches} 限流, 冷却 {wait:.0f}s 后重试 "
                      f"(外层 {attempt+1}/5)", flush=True)
                time.sleep(wait + 2.0)
            except Exception as e:
                print(f"    批次 {bi+1}/{total_batches} 失败: {type(e).__name__}: {e}", flush=True)
                break

        if not got or dfs_map is None:
            for c in chunk:
                results[c] = (None, to_af_symbol(c))
            continue

        # dfs_map: {symbol: DataFrame}
        for symbol, df in dfs_map.items():
            code = symbol_to_code.get(symbol)
            if code is None or df is None or df.empty:
                continue
            # 标准化列: trade_date -> date（与单只 get() 一致）
            df = df.copy()
            df["date"] = pd.to_datetime(df["trade_date"])
            cols = ["date", "open", "high", "low", "close", "volume", "amount"]
            df = df[[c for c in cols if c in df.columns]]
            df = df.sort_values("date").reset_index(drop=True)
            results[code] = (df, symbol)

        # 该批次中缺失的标记为失败
        done_symbols = set(dfs_map.keys())
        for symbol, code in symbol_to_code.items():
            if symbol not in done_symbols:
                results[code] = (None, symbol)

        ok_count = sum(1 for s in symbol_to_code if s in done_symbols)
        print(f"    批次 {bi+1}/{total_batches}: {len(chunk)} 只，成功 {ok_count} 只", flush=True)

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
        from collections import Counter
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
    print(f"   将批量拉取，每批 {BATCH_SIZE} 只并发请求")
    
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