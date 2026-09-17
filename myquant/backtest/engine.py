"""回测引擎 - 核心模拟交易执行"""
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from myquant.strategy.base import BaseStrategy


@dataclass
class Trade:
    """单笔交易记录"""
    date: str
    symbol: str
    direction: str  # BUY / SELL
    price: float
    volume: int
    amount: float
    commission: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    strategy: str = ""
    notes: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本统计
    symbol: str = ""
    strategy_name: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0
    
    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    
    # 数据
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)
    round_pnls: List[float] = field(default_factory=list)  # 已收口各轮总已实现盈亏(round-trip口径)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    
    def summary(self) -> str:
        """生成回测结果摘要"""
        return f"""
╔════════════════════════════════════════╗
║         回测结果报告                    ║
╠════════════════════════════════════════╣
║ 标的: {self.symbol:<20s}        ║
║ 策略: {self.strategy_name:<30s} ║
╠════════════════════════════════════════╣
║ 初始资金: {self.initial_capital:>12,.2f}          ║
║ 最终资金: {self.final_capital:>12,.2f}          ║
║ 总收益率: {self.total_return_pct:>11.2%}          ║
╠════════════════════════════════════════╣
║ 总交易次数: {self.total_trades:>5}                  ║
║ 胜率: {self.win_rate:>12.2%}          ║
║ 盈亏比: {self.profit_factor:>10.2f}               ║
║ 最大回撤: {self.max_drawdown_pct:>11.2%}          ║
║ 夏普比率: {self.sharpe_ratio:>10.2f}               ║
╚════════════════════════════════════════╝
"""


class BacktestEngine:
    """回测引擎
    
    负责:
    1. 加载历史数据
    2. 运行策略生成信号
    3. 模拟交易执行（考虑滑点、手续费）
    4. 生成绩效报告
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        bt_config = config.get("backtest", {})
        
        self.initial_capital = bt_config.get("initial_capital", 100000.0)
        self.commission = bt_config.get("commission", 0.0003)       # 手续费
        self.slippage = bt_config.get("slippage", 0.001)            # 滑点
        self.stamp_duty = bt_config.get("stamp_duty", 0.001)        # 印花税(卖出)
        self.min_volume = 100  # A股最小交易单位100股
        
        # 运行时状态
        self.capital = self.initial_capital
        self.position = 0           # 持仓数量
        self.avg_cost = 0.0         # 持仓均价
        self.equity = []            # 每日净值序列
        self.trades: List[Trade] = []
        self.trade_count = 0
        self._pending_round_pnl = 0.0       # 当前轮(一次BUY到清零)累计已实现盈亏
        self._round_pnls: List[float] = []  # 已收口各轮的round-trip总盈亏, 供指标统计
        
        logger.info(
            f"回测引擎初始化: 资金={self.initial_capital:,.2f}, "
            f"手续费={self.commission:.4f}, "
            f"滑点={self.slippage:.4f}"
        )
    
    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "",
        verbose: bool = True,
    ) -> BacktestResult:
        """运行回测
        
        Args:
            strategy: 策略实例
            data: 历史行情数据
            symbol: 标的代码
            verbose: 是否打印详细日志
            
        Returns:
            回测结果对象
        """
        logger.info(f"开始回测: {strategy.name} @ {symbol}")
        
        # 1. 重置状态
        self._reset()
        
        # 2. 运行策略生成信号
        signals = strategy.on_data(data)
        if signals is None or signals.empty:
            logger.warning("策略未生成任何信号")
            return BacktestResult()
        
        # 3. 模拟逐日交易
        equity_records = []
        dates = signals["date"] if "date" in signals.columns else signals.index
        
        for i in range(len(signals)):
            row = signals.iloc[i]
            current_date = dates.iloc[i] if hasattr(dates, 'iloc') else dates[i]
            current_price = float(row["close"])
            signal = float(row.get("signal", 0.0))
            
            # 执行交易信号
            if signal == 1.0 and self.position == 0:
                # 买入开仓
                self._execute_buy(current_date, current_price, signal)
                
            elif signal == -1.0 and self.position > 0:
                # 卖出平仓(exit_frac<1=部分平仓, 默认1.0全清; 存量策略无此列不受影响)
                exit_frac = float(row.get("exit_frac", 1.0))
                self._execute_sell(current_date, current_price, exit_frac)
            
            # 更新持仓市值
            position_value = self.position * current_price
            total_equity = self.capital + position_value
            self.equity = total_equity
            
            equity_records.append({
                "date": current_date,
                "price": current_price,
                "position": self.position,
                "capital": self.capital,
                "position_value": position_value,
                "total_equity": total_equity,
                "signal": signal,
            })
        
        # 4. 计算绩效
        equity_df = pd.DataFrame(equity_records)
        result = self._calculate_performance(equity_df, strategy, symbol)
        
        if verbose:
            logger.info(f"\n{result.summary()}")
        
        return result
    
    def _execute_buy(self, date: str, price: float, signal: float):
        """执行买入"""
        self._pending_round_pnl = 0.0     # 新的一轮开始, 重置轮内累计(引擎仅空仓时买入)
        price_with_slip = price * (1 + self.slippage)  # 买入滑点向上
        max_volume = int(self.capital * 0.95 / (price_with_slip * self.min_volume))
        volume = max_volume * self.min_volume
        
        if volume <= 0:
            return
        
        amount = price_with_slip * volume
        commission = max(amount * self.commission, 5.0)  # 最低5元
        total_cost = amount + commission
        
        if total_cost > self.capital:
            return
        
        self.capital -= total_cost
        self.position = volume
        self.avg_cost = price_with_slip
        self.trade_count += 1
        
        trade = Trade(
            date=date,
            symbol="",
            direction="BUY",
            price=price_with_slip,
            volume=volume,
            amount=amount,
            commission=commission,
            strategy="",
        )
        self.trades.append(trade)
        
        logger.debug(f"[{date}] 买入 {volume}股 @ {price_with_slip:.2f}, "
                     f"手续费={commission:.2f}, 剩余资金={self.capital:.2f}")
    
    def _execute_sell(self, date: str, price: float, exit_frac: float = 1.0):
        """执行卖出 (支持部分平仓)

        exit_frac=1.0: 全清(逻辑与旧版逐位一致); 0<exit_frac<1: 卖对应比例股数并保留剩余持仓,
        avg_cost 仅在清仓时归零 (剩余半仓按原成本继续持有)。持仓恒为100股整手。
        """
        price_with_slip = price * (1 - self.slippage)  # 卖出滑点向下

        if self.position <= 0:
            return

        if exit_frac >= 0.999999:
            volume = self.position                  # 全清
        else:
            # 部分平仓: 按整手上取整 (int(x+0.5) 避开 round() 银行家舍入), 剩余保持整手可后续一次清掉
            lots = int(self.position * exit_frac / self.min_volume + 0.5)
            lots = max(1, lots)
            volume = lots * self.min_volume
            if volume > self.position:
                volume = self.position

        amount = price_with_slip * volume
        commission = max(amount * self.commission, 5.0)
        stamp = amount * self.stamp_duty  # 印花税
        total_cost = commission + stamp

        cost_basis = volume * self.avg_cost
        pnl = amount - total_cost - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0

        self.capital += amount - total_cost
        self.position -= volume
        self._pending_round_pnl += pnl
        if self.position == 0:                      # 本轮(一次BUY到清零)收口, 记录该轮总盈亏
            self._round_pnls.append(self._pending_round_pnl)
            self._pending_round_pnl = 0.0
            self.avg_cost = 0.0
        # 部分平仓后 avg_cost 保留, 剩余持仓继续按原成本持有

        trade = Trade(
            date=date,
            symbol="",
            direction="SELL",
            price=price_with_slip,
            volume=volume,
            amount=amount,
            commission=commission,
            pnl=pnl,
            pnl_pct=pnl_pct,
            notes=f"印花税={stamp:.2f}" + ("" if exit_frac >= 0.999999 else f", exit_frac={exit_frac}"),
        )
        self.trades.append(trade)

        logger.debug(f"[{date}] 卖出 {volume}股 @ {price_with_slip:.2f}, "
                     f"盈亏={pnl:.2f}({pnl_pct:.2%}), "
                     f"资金={self.capital:.2f}")
    
    def _calculate_performance(
        self, equity_df: pd.DataFrame, strategy: BaseStrategy, symbol: str
    ) -> BacktestResult:
        """计算回测绩效指标"""
        result = BacktestResult()
        result.symbol = symbol
        result.strategy_name = strategy.name
        result.initial_capital = self.initial_capital
        # 最终资金 = 现金 + 持仓市值（如果仍有持仓）
        if self.position > 0 and not equity_df.empty:
            last_price = equity_df["price"].iloc[-1]
            position_value = self.position * last_price
            result.final_capital = self.capital + position_value
        else:
            result.final_capital = self.capital
        result.trades = self.trades
        
        # 总收益
        result.total_return = result.final_capital - self.initial_capital
        result.total_return_pct = result.total_return / self.initial_capital
        
        # 交易统计 (round-trip口径: 一轮=一次BUY到剩余股数清零的所有SELL累计已实现盈亏)
        # 全仓1:1时每轮恰一次SELL, 与旧的"按SELL逐笔"统计逐项一致; 期末未清空的开仓轮不进轮集但计入分母
        buy_trades = [t for t in self.trades if t.direction == "BUY"]
        result.total_trades = len(buy_trades)        # 口径不变 = 入场/开仓次数 (存量脚本依赖)
        result.round_pnls = list(self._round_pnls)

        winning = [p for p in self._round_pnls if p > 0]
        losing = [p for p in self._round_pnls if p <= 0]
        result.winning_trades = len(winning)
        result.losing_trades = len(losing)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        result.avg_win = float(np.mean(winning)) if winning else 0
        result.avg_loss = float(np.mean(losing)) if losing else 0

        # 盈亏比
        total_profit = sum(winning)
        total_loss = abs(sum(losing))
        result.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 最大回撤
        if not equity_df.empty:
            equity_curve = equity_df["total_equity"].values
            peak = equity_curve[0]
            max_dd = 0
            max_dd_pct = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = peak - value
                dd_pct = dd / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct
            result.max_drawdown = max_dd
            result.max_drawdown_pct = max_dd_pct
        
        # 年化收益率和夏普比率
        if not equity_df.empty and len(equity_df) > 1:
            equity_df["return"] = equity_df["total_equity"].pct_change()
            daily_returns = equity_df["return"].dropna()
            
            if len(daily_returns) > 0:
                trading_days = len(daily_returns)
                annual_return = (result.final_capital / self.initial_capital) ** (252 / trading_days) - 1
                
                # 计算波动率
                result.volatility = daily_returns.std() * np.sqrt(252)
                
                # 夏普比率（假设无风险利率3%）
                excess_returns = daily_returns.mean() * 252 - 0.03
                result.sharpe_ratio = excess_returns / result.volatility if result.volatility > 0 else 0
                
                result.daily_returns = daily_returns
                result.equity_curve = equity_df
        
        return result
    
    def _reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0
        self.avg_cost = 0.0
        self.equity = []
        self.trades = []
        self.trade_count = 0
        self._pending_round_pnl = 0.0
        self._round_pnls = []