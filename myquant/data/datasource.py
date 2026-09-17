"""数据源模块 - 多数据源行情数据获取"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger


class DataSource:
    """数据源 - 市场行情数据获取
    
    支持数据源:
    - alphafeed: 专业A股/美股数据（默认，需API key）
    - akshare: 免费A股数据
    - tushare: 需要token
    - yfinance: 美股数据
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.source = config.get("source", "alphafeed")
        self.cache_dir = Path(config.get("cache_dir", "./data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = None
        
        # AlphaFeed 专用配置
        if self.source == "alphafeed":
            self._alphafeed_api_key = config.get("alphafeed_api_key", "")
            self._alphafeed_timeout = config.get("alphafeed_timeout", 30)
        
    def _get_api(self):
        """获取数据源API实例（延迟加载）"""
        if self._api is None:
            if self.source == "alphafeed":
                from alphafeed import AlphaFeed
                api_key = self._alphafeed_api_key or None
                self._api = AlphaFeed(
                    api_key=api_key,
                    timeout=self._alphafeed_timeout,
                )
            elif self.source == "akshare":
                import akshare as ak
                self._api = ak
            elif self.source == "tushare":
                import tushare as ts
                token = self.config.get("tushare_token", "78d5102db91829371b112ef58345512771818588412e6bcab10bf40f")
                if token:
                    ts.set_token(token)
                self._api = ts
            elif self.source == "yfinance":
                import yfinance as yf
                self._api = yf
            else:
                raise ValueError(f"不支持的数据源: {self.source}")
        return self._api
    
    def get_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """获取日线行情数据
        
        Args:
            symbol: 股票代码，如 "000001" 或 "600519"
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期，默认为今天
            adjust: 复权方式 (qfq=前复权, hfq=后复权, None=不复权)
            use_cache: 是否使用本地缓存
            
        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, amount]
        """
        api = self._get_api()
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        # 尝试从缓存读取
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}_{adjust}.parquet"
        if use_cache and cache_file.exists():
            logger.info(f"从缓存读取 {symbol} 数据: {cache_file}")
            return pd.read_parquet(cache_file)
        
        logger.info(f"从 {self.source} 获取 {symbol} ({start_date} ~ {end_date})")
        
        if self.source == "alphafeed":
            df = self._get_alphafeed_daily(symbol, start_date, end_date, adjust)
        elif self.source == "akshare":
            df = self._get_akshare_daily(symbol, start_date, end_date, adjust)
        elif self.source == "tushare":
            df = self._get_tushare_daily(symbol, start_date, end_date)
        else:
            df = self._get_yfinance_daily(symbol, start_date, end_date)
        
        if df is None or df.empty:
            logger.warning(f"未获取到 {symbol} 的数据")
            return pd.DataFrame()
        
        # 标准化列名
        df = self._standardize_columns(df)
        
        # 保存缓存
        if use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file)
            logger.info(f"数据已缓存: {cache_file}")
        
        return df
    
    def _get_alphafeed_daily(
        self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """从AlphaFeed获取日线数据
        
        AlphaFeed 提供专业级A股/美股行情数据。
        symbol 格式示例：
        - A股: "600519.SH", "000001.SZ"
        - 美股: "AAPL.US"
        
        adjust 映射:
        - qfq/hfq -> forward/backward
        - None/不复权 -> none
        """
        api = self._get_api()
        
        # 转换symbol格式：6位代码 + 市场后缀
        # AlphaFeed 要求: 600XXX.SH / 00XXXX.SZ / 30XXXX.SZ
        symbol_af = symbol.upper()
        if ".SH" not in symbol_af and ".SZ" not in symbol_af and ".US" not in symbol_af:
            if symbol.startswith("6"):
                symbol_af = f"{symbol}.SH"
            elif symbol.startswith("0") or symbol.startswith("3"):
                symbol_af = f"{symbol}.SZ"
            else:
                symbol_af = f"{symbol}.US"
        
        # 转换复权参数
        adjust_map = {"qfq": "forward", "hfq": "backward", None: "none", "none": "none"}
        af_adjust = adjust_map.get(adjust, "forward")
        
        try:
            # 计算需要的K线数量（AlphaFeed 返回最新count条，需要额外buffer）
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            diff_days = (end_dt - start_dt).days
            count = min(max(diff_days * 2 + 250, 500), 10000)
            
            # 当调获取（先按count取，再过滤日期范围）
            df = api.klines.get(
                symbol_af,
                period="1d",
                count=count,
                adjust=af_adjust,
                to_dataframe=True,
            )
            
            if df is None or df.empty:
                logger.warning(f"AlphaFeed 返回空数据: {symbol_af}")
                return None
            
            # AlphaFeed DataFrame 列: symbol, name, timestamp, trade_date, trade_time, open, high, low, close, volume, amount
            # 使用 trade_date（已为正确日期）创建date字段
            df = df.reset_index(drop=True)
            df["date"] = pd.to_datetime(df["trade_date"])
            
            # 只保留需要的列
            cols = ["date", "open", "high", "low", "close", "volume", "amount"]
            df = df[[c for c in cols if c in df.columns]]
            
            # 过滤日期范围
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            df = df[mask].sort_values("date").reset_index(drop=True)
            
            logger.info(f"  ✅ AlphaFeed 获取 {symbol_af} 成功: {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"AlphaFeed获取 {symbol} (->{symbol_af}) 失败: {type(e).__name__}: {e}")
            return None
    
    def _get_akshare_daily(
        self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """从akshare获取A股日线数据
        
        使用多重备选接口，提高数据获取成功率:
        1. stock_zh_a_hist (akshare封装 - 东方财富)
        2. _fetch_eastmoney_direct (requests直连 - 绕过curl_cffi问题)
        3. stock_zh_a_daily (akshare - 新浪)
        """
        api = self._get_api()
        errors = []
        
        # 接口1: akshare 官方接口
        try:
            df = api.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust,
            )
            if df is not None and not df.empty:
                df.columns = [
                    "date", "open", "close", "high", "low", "volume",
                    "amount", "amplitude", "change_pct", "change_amount", "turnover"
                ]
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as e:
            errors.append(f"akshare官方: {type(e).__name__}")
        
        # 接口2: requests 直连东方财富 (绕过 curl_cffi TLS问题)
        try:
            logger.info(f"  -> 使用requests直连东方财富API")
            df = self._fetch_from_eastmoney(symbol, start_date, end_date, adjust)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            errors.append(f"直连东方财富: {type(e).__name__}")
        
        logger.error(f"所有接口均失败: {'; '.join(errors)}")
        return None
    
    def _fetch_from_eastmoney(
        self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """使用requests直连东方财富API
        
        绕过akshare的curl_cffi库（某些网络环境下被服务器重置）
        """
        import requests
        import json
        
        # 市场代码: 1=上海, 0=深圳
        secid = f"1.{symbol}" if symbol.startswith("6") else f"0.{symbol}"
        fqt = "1" if adjust == "qfq" else "2" if adjust == "hfq" else "0"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://quote.eastmoney.com/",
        }
        
        # 计算需要的K线数量（粗略估算）
        import datetime
        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days
            limit = min(max(days + 100, 300), 8000)  # 最多8000条
        except:
            limit = 1000
        
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",  # 日K
            "fqt": fqt,    # 复权
            "end": "20500101",
            "lmt": str(limit),
        }
        
        r = requests.get(url, params=params, headers=headers, timeout=15)
        data = r.json()
        
        if not data.get("data") or not data["data"].get("klines"):
            logger.warning(f"东方财富直连接口返回空数据")
            return None
        
        klines = data["data"]["klines"]
        
        # 解析K线数据
        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) >= 11:
                rows.append({
                    "date": parts[0],
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "volume": float(parts[5]),
                    "amount": float(parts[6]),
                    "amplitude": float(parts[7]),
                    "change_pct": float(parts[8]),
                    "change_amount": float(parts[9]),
                    "turnover": float(parts[10]),
                })
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        
        # 过滤日期范围
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df[mask].sort_values("date").reset_index(drop=True)
        
        logger.info(f"  ✅ 直连获取 {symbol} 成功: {len(df)} 条")
        return df
    
    def _get_tushare_daily(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """从tushare获取日线数据"""
        api = self._get_api()
        try:
            df = api.daily(
                ts_code=symbol,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "trade_date": "date",
                    "vol": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"tushare获取 {symbol} 失败: {e}")
            return None
    
    def _get_yfinance_daily(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """从yfinance获取美股日线数据"""
        api = self._get_api()
        try:
            ticker = api.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                df = df.reset_index()
                df = df.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })
            return df
        except Exception as e:
            logger.error(f"yfinance获取 {symbol} 失败: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名为小写英文"""
        col_mapping = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "change_pct",
            "涨跌额": "change_amount", "换手率": "turnover",
            "trade_date": "date", "trade_time": "trade_time",
            "timestamp": "timestamp", "symbol": "symbol", "name": "name",
            "vol": "volume",
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
        
        # 确保包含核心列
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"数据缺少列: {col}")
        
        return df
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        api = self._get_api()
        if self.source == "akshare":
            try:
                df = api.stock_zh_a_spot_em()  # 实时行情快照
                return df[["代码", "名称", "最新价", "涨跌幅", "成交量", "成交额"]]
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
                try:
                    df = api.stock_info_a_code_name()
                    return df
                except Exception as e2:
                    logger.error(f"获取股票列表(备用)失败: {e2}")
        return pd.DataFrame()
    
    def get_index_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取指数行情数据
        
        Args:
            symbol: 指数代码，如 "000300" (沪深300), "000001" (上证指数)
            start_date: 开始日期
            end_date: 结束日期
        """
        api = self._get_api()
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        try:
            df = api.stock_zh_index_daily(
                symbol=f"sh{symbol}" if symbol.startswith("0") else symbol,
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "date": "date", "open": "open", "close": "close",
                    "high": "high", "low": "low", "volume": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                # 过滤日期范围
                mask = (df["date"] >= start_date) & (df["date"] <= end_date)
                df = df[mask]
            return df
        except Exception as e:
            logger.error(f"获取指数 {symbol} 失败: {e}")
            return pd.DataFrame()
    
    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情快照
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            实时行情DataFrame
        """
        api = self._get_api()
        if self.source == "akshare":
            try:
                df = api.stock_zh_a_spot_em()
                # 筛选需要的股票
                result = df[df["代码"].isin(symbols)].copy()
                return result
            except Exception as e:
                logger.error(f"获取实时行情失败: {e}")
        return pd.DataFrame()