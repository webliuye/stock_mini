"""获取全量A股（沪市、深市、北交所）股票代码

策略: 生成A股合法代码范围 → alphafeed instruments API 批量验证
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import time
from itertools import chain


def generate_a_share_codes():
    """生成A股全部可能代码范围
    
    A股代码规则:
    - 上海主板: 600000-605999
    - 上海科创板: 688000-689999
    - 深圳主板: 000001-004999
    - 深圳创业板: 300000-301999
    - 北交所: 800000-879999, 920000-929999
    """
    ranges = [
        # (start, end, suffix, label)
        (600000, 605999 + 1, ".SH", "上海主板"),
        (688000, 689999 + 1, ".SH", "科创板"),
        (0, 4999 + 1, ".SZ", "深圳主板"),      # 000001-004999, zerofill
        (300000, 301999 + 1, ".SZ", "创业板"),
        (800000, 879999 + 1, ".BJ", "北交所"),
        (920000, 929999 + 1, ".BJ", "北交所(92)"),
    ]
    
    symbols = []
    for start, end, suffix, label in ranges:
        count = end - start
        for num in range(start, end):
            code = str(num).zfill(6)
            symbols.append(f"{code}{suffix}")
    
    return symbols


def batch_verify_instruments(symbols, api_key, batch_size=200, max_workers=5):
    """使用 alphafeed instruments API 批量验证代码是否有效
    
    返回: list of dicts with keys: symbol, name, code
    """
    from alphafeed import AlphaFeed
    
    client = AlphaFeed(api_key=api_key)
    valid_insts = []
    total = len(symbols)
    
    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        
        try:
            insts = client.instruments.get(batch)
            valid_insts.extend(insts)
        except Exception as e:
            print(f"    ⚠️ 批次 {i//batch_size + 1} 出错: {type(e).__name__}, 跳过")
            time.sleep(0.5)
            continue
        
        # 进度
        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total:
            progress = min(i + batch_size, total)
            pct = progress / total * 100
            print(f"    进度: {progress}/{total} ({pct:.1f}%), 已确认 {len(valid_insts)} 只")
        
        time.sleep(0.05)  # 控制频率
    
    client.close()
    return valid_insts


print("=" * 60)
print("  获取全量A股股票代码")
print("  方案: 生成代码范围 → alphafeed instruments 验证")
print("=" * 60)

# 加载API key
import yaml
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
api_key = config["data"].get("alphafeed_api_key", "")

# 1. 生成A股全部可能的代码
print("\n📋 生成A股代码范围...")
symbols = generate_a_share_codes()
print(f"  候选代码总数: {len(symbols)} 个")

# 2. alphafeed instruments 批量验证
print(f"\n🔍 alphafeed instruments 批量验证...")
valid_insts = batch_verify_instruments(symbols, api_key, batch_size=200)

print(f"\n  ✅ alphafeed 确认有效标的: {len(valid_insts)} 只")

if not valid_insts:
    print("\n  ⚠️ alphafeed 验证无结果，尝试备用方案...")
    
    # 备用: 从已有数据文件恢复
    try:
        import glob, os
        cache_files = glob.glob("data/cache/*.parquet")
        codes = set()
        for f in cache_files:
            name = os.path.basename(f)
            code = name.split("_")[0]
            if len(code) == 6 and code.isdigit():
                codes.add(code)
        
        if codes:
            print(f"  从缓存提取: {len(codes)} 只")
            df = pd.DataFrame({"代码": sorted(codes), "名称": ""})
            df["alphafeed_symbol"] = df["代码"].apply(
                lambda x: f"{x}.SH" if x.startswith("6") or x.startswith("68")
                else (f"{x}.BJ" if x.startswith("8") else f"{x}.SZ")
            )
            df.to_csv("data/all_stocks.csv", index=False, encoding="utf-8-sig")
            df.to_parquet("data/all_stocks.parquet", index=False)
            print(f"  💾 已保存 (缓存): {len(df)} 只")
        else:
            print("  ❌ 缓存也为空")
    except Exception as e:
        print(f"  ❌ 备用方案失败: {e}")
else:
    # 3. 构建 DataFrame
    rows = []
    for inst in valid_insts:
        sym = inst.get("symbol", "")
        name = inst.get("name", "")
        # 从 "600519.SH" 提取纯代码 "600519"
        code = sym.replace(".SH", "").replace(".SZ", "").replace(".BJ", "").replace(".US", "")
        if len(code) == 6 and code.isdigit():
            rows.append({
                "代码": code,
                "名称": name,
                "alphafeed_symbol": sym,
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("代码").reset_index(drop=True)
    
    # 统计
    print(f"\n{'='*60}")
    print(f"  最终结果: {len(df)} 只股票")
    
    sh_all = df[df["代码"].str.startswith("6")].shape[0]
    kcb = df[df["代码"].str.startswith("68")].shape[0]
    sh_main = sh_all - kcb
    sz_main = df[df["代码"].str.match(r"^00").shape[0]]
    sz_gem = df[df["代码"].str.startswith("3")].shape[0]
    bj_92 = df[df["代码"].str.match(r"^92").shape[0]]
    bj_8 = df[df["代码"].str.match(r"^8[0-7]").shape[0]]
    bj_all = bj_92 + bj_8
    
    print(f"\n  📊 市场分布:")
    print(f"     上海主板 (60xxxx):    {sh_main:>5} 只")
    print(f"     科创板   (688xxx):    {kcb:>5} 只")
    print(f"     深圳主板 (00xxxx):    {sz_main:>5} 只")
    print(f"     创业板   (30xxxx):    {sz_gem:>5} 只")
    print(f"     北交所   (8xxxxx/92xxxxx): {bj_all:>5} 只")
    print(f"     ──────────────────────────")
    print(f"     合计:                  {len(df):>5} 只")
    
    # ST 统计
    st_count = df["名称"].str.contains(r"ST|\*ST|退", na=False, regex=True).sum()
    n_count = df["名称"].str.match(r"^N", na=False).sum()
    print(f"\n  🏷️  ST/退市: {st_count} 只 | N新股: {n_count} 只")
    
    # 保存
    df.to_csv("data/all_stocks.csv", index=False, encoding="utf-8-sig")
    df.to_parquet("data/all_stocks.parquet", index=False)
    print(f"\n  💾 已保存: data/all_stocks.csv ({len(df)}行)")
    print(f"  💾 已保存: data/all_stocks.parquet")
    
    # 预览
    print(f"\n  📋 预览 (前10):")
    print(df[["代码", "名称"]].head(10).to_string(index=False))
    print(f"\n  📋 预览 (后10):")
    print(df[["代码", "名称"]].tail(10).to_string(index=False))

print(f"\n{'='*60}\n  完成\n{'='*60}")