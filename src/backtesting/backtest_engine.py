"""
Backtesting Engine
Realistic backtesting with transaction costs and proper execution modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum

from src.core.constants import TradingConstants as TC, DataQualityLimits as DQL

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """Order representation"""
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    account_id: Optional[str] = None

    def __post_init__(self):
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires limit_price")
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"{self.order_type} requires stop_price")


@dataclass
class Fill:
    """Trade execution representation"""
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float
    slippage: float
    order: Order
    account_id: Optional[str] = None


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT

    @property
    def value(self) -> float:
        return self.quantity * self.avg_price

    def update(self, fill: Fill) -> float:
        """Update position with fill, return realized P&L"""
        realized = 0.0

        if self.quantity == 0:
            # Opening position
            self.quantity = fill.quantity
            self.avg_price = fill.price
        elif np.sign(self.quantity) == np.sign(fill.quantity):
            # Adding to position
            total_value = self.value + fill.quantity * fill.price
            self.quantity += fill.quantity
            self.avg_price = total_value / self.quantity if self.quantity != 0 else 0
        else:
            # Reducing or reversing position
            if abs(fill.quantity) <= abs(self.quantity):
                # Partial close
                realized = fill.quantity * (self.avg_price - fill.price)
                self.quantity += fill.quantity
            else:
                # Full close and reverse
                close_qty = -self.quantity
                realized = close_qty * (self.avg_price - fill.price)

                # New position
                new_qty = fill.quantity + close_qty
                self.quantity = new_qty
                self.avg_price = fill.price

        self.realized_pnl += realized
        self.total_commission += fill.commission

        return realized


@dataclass
class BacktestResult:
    """Backtest results container"""
    # Time series data
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame

    # Summary statistics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float

    # Trade statistics
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    var_95: float
    cvar_95: float

    # Costs
    total_commission: float
    total_slippage: float

    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine with realistic execution modeling
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = TC.COMMISSION_PER_SHARE,
                 min_commission: float = TC.MIN_COMMISSION,
                 slippage_bps: float = TC.SLIPPAGE_BPS,
                 market_impact_bps: float = TC.MARKET_IMPACT_BPS):
        """
        Initialize backtesting engine

        Args:
            initial_capital: Starting capital
            commission_rate: Commission per share
            min_commission: Minimum commission per trade
            slippage_bps: Slippage in basis points
            market_impact_bps: Market impact in basis points
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        self.market_impact_bps = market_impact_bps

        # State tracking
        self.reset()

    def reset(self):
        """Reset engine state"""
        self.capital = self.initial_capital
        self.positions = {}  # symbol -> Position
        self.equity_curve = []
        self.trades = []
        self.pending_orders = []
        self.current_timestamp = None

    def run_backtest(self,
                    data: pd.DataFrame,
                    signal_func: Callable,
                    position_sizer: Optional[Callable] = None,
                    risk_manager: Optional[Callable] = None) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            data: Historical data with columns [date, symbol, price, ...]
            signal_func: Function that generates signals (data_slice) -> List[Order]
            position_sizer: Function to size positions (signal, capital, risk) -> quantity
            risk_manager: Function to manage risk (positions, capital) -> List[Order]

        Returns:
            BacktestResult with performance metrics
        """
        self.reset()

        # Ensure data is sorted by date
        data = data.sort_values('date')
        dates = data['date'].unique()

        logger.info(f"Running backtest from {dates[0]} to {dates[-1]}")

        # Main backtest loop
        for date in dates:
            self.current_timestamp = pd.Timestamp(date)
            daily_data = data[data['date'] == date]

            # 1. Process pending orders
            fills = self._process_orders(daily_data)

            # 2. Update positions with fills
            for fill in fills:
                self._update_position(fill)

            # 3. Mark-to-market positions
            self._mark_to_market(daily_data)

            # 4. Apply risk management
            if risk_manager:
                risk_orders = risk_manager(self.positions, self.capital)
                self.pending_orders.extend(risk_orders)

            # 5. Generate new signals
            signals = signal_func(daily_data)

            # 6. Size positions
            if position_sizer:
                for signal in signals:
                    signal.quantity = position_sizer(
                        signal,
                        self.capital,
                        self._calculate_portfolio_risk()
                    )

            # 7. Submit new orders
            self.pending_orders.extend(signals)

            # 8. Record equity
            total_value = self._calculate_portfolio_value(daily_data)
            self.equity_curve.append({
                'date': date,
                'capital': self.capital,
                'positions_value': total_value - self.capital,
                'total_equity': total_value
            })

        # Create results
        return self._create_results()

    def _process_orders(self, market_data: pd.DataFrame) -> List[Fill]:
        """Process pending orders against market data"""
        fills = []
        remaining_orders = []

        for order in self.pending_orders:
            fill = self._try_fill_order(order, market_data)

            if fill:
                fills.append(fill)
                self.trades.append(fill)
            else:
                # Keep unfilled orders (e.g., limit orders)
                remaining_orders.append(order)

        self.pending_orders = remaining_orders
        return fills

    def _try_fill_order(self, order: Order, market_data: pd.DataFrame) -> Optional[Fill]:
        """Try to fill an order given market data"""

        # Get relevant market data
        symbol_data = market_data[market_data['symbol'] == order.symbol] if 'symbol' in market_data.columns else market_data

        if symbol_data.empty:
            return None

        # For simplicity, use first row (can be enhanced for tick data)
        row = symbol_data.iloc[0]
        market_price = row.get('price', row.get('close', row.get('net_pnl', 0)))

        # Check if order can be filled
        can_fill = False
        fill_price = market_price

        if order.order_type == OrderType.MARKET:
            can_fill = True

        elif order.order_type == OrderType.LIMIT:
            if order.quantity > 0:  # Buy limit
                can_fill = market_price <= order.limit_price
                fill_price = min(market_price, order.limit_price)
            else:  # Sell limit
                can_fill = market_price >= order.limit_price
                fill_price = max(market_price, order.limit_price)

        elif order.order_type == OrderType.STOP:
            if order.quantity > 0:  # Buy stop
                can_fill = market_price >= order.stop_price
            else:  # Sell stop
                can_fill = market_price <= order.stop_price

        if not can_fill:
            return None

        # Calculate execution costs
        slippage = self._calculate_slippage(order, market_price)
        commission = self._calculate_commission(order)

        # Adjust fill price for slippage
        if order.quantity > 0:
            fill_price += slippage
        else:
            fill_price -= slippage

        # Create fill
        fill = Fill(
            timestamp=self.current_timestamp,
            symbol=order.symbol,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage * abs(order.quantity),
            order=order,
            account_id=order.account_id
        )

        return fill

    def _calculate_slippage(self, order: Order, market_price: float) -> float:
        """Calculate slippage based on order size and market impact"""

        # Base slippage
        slippage_pct = self.slippage_bps / 10000

        # Add market impact for large orders
        # Simple model: impact increases with square root of size
        size_factor = np.sqrt(abs(order.quantity) / 100)  # Normalize by 100 shares
        market_impact_pct = self.market_impact_bps / 10000 * size_factor

        total_slippage_pct = slippage_pct + market_impact_pct

        return market_price * total_slippage_pct

    def _calculate_commission(self, order: Order) -> float:
        """Calculate commission for order"""
        commission = abs(order.quantity) * self.commission_rate
        return max(commission, self.min_commission)

    def _update_position(self, fill: Fill):
        """Update position with fill"""
        symbol = fill.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]
        realized_pnl = position.update(fill)

        # Update capital
        self.capital -= fill.quantity * fill.price  # Cash outflow for buys
        self.capital -= fill.commission  # Commission always reduces capital
        self.capital += realized_pnl  # Add realized P&L

        # Remove position if flat
        if position.quantity == 0:
            del self.positions[symbol]

    def _mark_to_market(self, market_data: pd.DataFrame):
        """Update unrealized P&L for all positions"""

        for symbol, position in self.positions.items():
            symbol_data = market_data[market_data['symbol'] == symbol] if 'symbol' in market_data.columns else market_data

            if not symbol_data.empty:
                current_price = symbol_data.iloc[0].get('price', symbol_data.iloc[0].get('close', 0))
                position.unrealized_pnl = position.quantity * (current_price - position.avg_price)

    def _calculate_portfolio_value(self, market_data: pd.DataFrame) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.value + pos.unrealized_pnl for pos in self.positions.values())
        return self.capital + positions_value

    def _calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calculate current portfolio risk metrics"""

        # Position concentration
        total_value = sum(abs(pos.value) for pos in self.positions.values())
        max_position = max((abs(pos.value) for pos in self.positions.values()), default=0)

        concentration = max_position / (total_value + TC.MIN_VARIANCE)

        # Exposure
        long_exposure = sum(pos.value for pos in self.positions.values() if pos.side == PositionSide.LONG)
        short_exposure = abs(sum(pos.value for pos in self.positions.values() if pos.side == PositionSide.SHORT))
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        return {
            'concentration': concentration,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'n_positions': len(self.positions)
        }

    def _create_results(self) -> BacktestResult:
        """Create backtest results from recorded data"""

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)

        # Calculate returns
        returns = equity_df['total_equity'].pct_change().dropna()

        # Calculate metrics
        total_days = len(equity_df)
        years = total_days / TC.TRADING_DAYS_PER_YEAR

        total_return = (equity_df['total_equity'].iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        volatility = returns.std() * np.sqrt(TC.TRADING_DAYS_PER_YEAR)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(TC.TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0

        sharpe_ratio = (annual_return - TC.RISK_FREE_RATE) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - TC.RISK_FREE_RATE) / downside_deviation if downside_deviation > 0 else 0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0

        for date, dd in drawdown.items():
            if dd < 0 and drawdown_start is None:
                drawdown_start = date
                current_duration = 1
            elif dd < 0:
                current_duration += 1
            elif dd == 0 and drawdown_start is not None:
                max_duration = max(max_duration, current_duration)
                drawdown_start = None
                current_duration = 0

        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Trade statistics
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        if not trades_df.empty:
            # Calculate trade P&L
            trade_pnl = []
            for i, trade in trades_df.iterrows():
                # Simple P&L calculation (can be enhanced)
                if i > 0 and trade['symbol'] == trades_df.iloc[i-1]['symbol']:
                    pnl = -trade['quantity'] * (trade['price'] - trades_df.iloc[i-1]['price'])
                    trade_pnl.append(pnl)
                else:
                    trade_pnl.append(0)

            trades_df['pnl'] = trade_pnl
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]

            n_trades = len(trades_df)
            win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

            total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

            total_commission = trades_df['commission'].sum()
            total_slippage = trades_df['slippage'].sum()
        else:
            n_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_commission = 0
            total_slippage = 0

        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # Create positions DataFrame
        positions_data = []
        for symbol, position in self.positions.items():
            positions_data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl
            })
        positions_df = pd.DataFrame(positions_data)

        return BacktestResult(
            equity_curve=equity_df['total_equity'],
            returns=returns,
            positions=positions_df,
            trades=trades_df,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_duration,
            calmar_ratio=calmar_ratio,
            n_trades=n_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            volatility=volatility,
            downside_deviation=downside_deviation,
            var_95=var_95,
            cvar_95=cvar_95,
            total_commission=total_commission,
            total_slippage=total_slippage,
            metadata={
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_days': total_days
            }
        )
