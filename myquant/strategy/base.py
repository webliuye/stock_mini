"""策略基类 - 所有策略的抽象接口"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from loguru import logger


class BaseStrategy(ABC):
    """策略抽象基类
    
    所有具体策略需继承此类并实现 generate_signals 方法。
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self._signals = None
        
        logger.info(f"策略 [{self.name}] 初始化, 参数: {self.params}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号
        
        Args:
            data: 行情DataFrame，至少包含 [date, open, high, low, close, volume]
            
        Returns:
            带信号列的DataFrame，需包含 'signal' 列:
            - 1.0: 买入
            - 0.0: 持仓/观望
            - -1.0: 卖出
        """
        pass
    
    def get_signal(self) -> float:
        """获取最新信号
        
        Returns:
            当前信号值
        """
        if self._signals is not None and not self._signals.empty:
            return self._signals["signal"].iloc[-1]
        return 0.0
    
    def get_signals(self) -> Optional[pd.DataFrame]:
        """获取完整信号序列"""
        return self._signals
    
    def on_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据处理入口（在回测/实盘中调用）
        
        Args:
            data: 最新行情数据
            
        Returns:
            带信号的DataFrame
        """
        if data.empty:
            return data
        
        self._signals = self.generate_signals(data)
        return self._signals
    
    def reset(self):
        """重置策略状态"""
        self._signals = None
        logger.info(f"策略 [{self.name}] 已重置")
    
    def __repr__(self) -> str:
        return f"Strategy(name='{self.name}', params={self.params})"