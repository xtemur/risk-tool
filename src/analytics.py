"""
Trader Analytics Module for Risk Management MVP
Generates comprehensive performance, risk, and behavioral analytics
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.dates as mdates
from io import BytesIO
import base64

from src.database import Database

logger = logging.getLogger(__name__)


class TraderAnalytics:
    """Comprehensive analytics for day traders"""

    def __init__(self):
        self.db = Database()
        self.plots_dir = Path('data/analytics_plots')
        self.plots_dir.mkdir(exist_ok=True)

        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def calculate_performance_metrics(self, totals_df: pd.DataFrame, period: str = "monthly") -> Dict:
        """Calculate core performance metrics"""

        if totals_df.empty:
            return {}

        # Basic metrics
        total_days = len(totals_df)
        trading_days = len(totals_df[totals_df['net_pnl'] != 0])
        winning_days = len(totals_df[totals_df['net_pnl'] > 0])
        losing_days = len(totals_df[totals_df['net_pnl'] < 0])

        # Performance calculations
        win_rate = (winning_days / trading_days * 100) if trading_days > 0 else 0

        total_pnl = totals_df['net_pnl'].sum()
        avg_daily_pnl = totals_df['net_pnl'].mean()

        # Profit factor
        gross_profit = totals_df[totals_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(totals_df[totals_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Risk metrics
        daily_returns = totals_df['net_pnl']
        volatility = daily_returns.std()

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (avg_daily_pnl / volatility) if volatility > 0 else 0

        # Drawdown analysis
        cumulative_pnl = daily_returns.cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = drawdown.min()

        # Recovery analysis
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        is_in_drawdown = current_drawdown < -10  # $10 threshold

        return {
            'total_days': total_days,
            'trading_days': trading_days,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'is_in_drawdown': is_in_drawdown,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'best_day': daily_returns.max(),
            'worst_day': daily_returns.min(),
            'winning_days': winning_days,
            'losing_days': losing_days
        }

    def calculate_risk_metrics(self, totals_df: pd.DataFrame) -> Dict:
        """Calculate advanced risk metrics"""

        if totals_df.empty:
            return {}

        daily_returns = totals_df['net_pnl']

        # Value at Risk (VaR) - 5% and 1%
        var_5 = np.percentile(daily_returns, 5)
        var_1 = np.percentile(daily_returns, 1)

        # Expected Shortfall (CVaR)
        cvar_5 = daily_returns[daily_returns <= var_5].mean()

        # Consecutive losing days
        losing_streaks = []
        current_streak = 0
        for pnl in daily_returns:
            if pnl < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    losing_streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            losing_streaks.append(current_streak)

        max_losing_streak = max(losing_streaks) if losing_streaks else 0

        # Recovery metrics
        cumulative_pnl = daily_returns.cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max

        # Time to recovery (days from drawdown to new high)
        recovery_times = []
        in_drawdown = False
        drawdown_start = 0

        for i, dd in enumerate(drawdown):
            if dd < -50 and not in_drawdown:  # Start of significant drawdown
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:  # Recovery to new high
                recovery_times.append(i - drawdown_start)
                in_drawdown = False

        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0

        return {
            'var_5_percent': var_5,
            'var_1_percent': var_1,
            'cvar_5_percent': cvar_5,
            'max_losing_streak': max_losing_streak,
            'avg_recovery_time': avg_recovery_time,
            'downside_deviation': daily_returns[daily_returns < 0].std()
        }

    def calculate_behavior_metrics(self, totals_df: pd.DataFrame, fills_df: pd.DataFrame) -> Dict:
        """Calculate trading behavior patterns"""

        behavior_metrics = {}

        # Trading frequency analysis
        if not totals_df.empty:
            avg_daily_orders = totals_df['orders_count'].mean()
            avg_daily_fills = totals_df['fills_count'].mean()

            # Overtrading indicator (high activity, poor performance)
            high_activity_days = totals_df[totals_df['orders_count'] > totals_df['orders_count'].quantile(0.8)]
            overtrading_pnl = high_activity_days['net_pnl'].mean()

            behavior_metrics.update({
                'avg_daily_orders': avg_daily_orders,
                'avg_daily_fills': avg_daily_fills,
                'overtrading_pnl': overtrading_pnl,
                'high_activity_days': len(high_activity_days)
            })

        # Time-of-day analysis
        if not fills_df.empty:
            fills_df['hour'] = pd.to_datetime(fills_df['datetime']).dt.hour

            # Group by hour and calculate P&L (approximate from fills)
            hourly_activity = fills_df.groupby('hour').agg({
                'quantity': 'sum',
                'total_fees': 'sum'
            })

            # Best and worst trading hours
            if len(hourly_activity) > 0:
                peak_activity_hour = hourly_activity['quantity'].abs().idxmax()
                behavior_metrics['peak_trading_hour'] = peak_activity_hour

                # Morning vs afternoon performance
                morning_activity = fills_df[fills_df['hour'] < 12]['quantity'].abs().sum()
                afternoon_activity = fills_df[fills_df['hour'] >= 12]['quantity'].abs().sum()

                behavior_metrics.update({
                    'morning_activity_pct': morning_activity / (morning_activity + afternoon_activity) * 100,
                    'afternoon_activity_pct': afternoon_activity / (morning_activity + afternoon_activity) * 100
                })

        # Symbol diversification
        if not fills_df.empty:
            unique_symbols = fills_df['symbol'].nunique()
            total_volume = fills_df['quantity'].abs().sum()

            # Concentration risk (Herfindahl index)
            symbol_weights = fills_df.groupby('symbol')['quantity'].sum().abs()
            symbol_weights = symbol_weights / symbol_weights.sum()
            concentration_index = (symbol_weights ** 2).sum()

            behavior_metrics.update({
                'symbols_traded': unique_symbols,
                'concentration_risk': concentration_index,
                'diversification_score': 1 - concentration_index
            })

        return behavior_metrics

    def calculate_efficiency_metrics(self, totals_df: pd.DataFrame) -> Dict:
        """Calculate trading efficiency metrics"""

        if totals_df.empty:
            return {}

        # Fee efficiency
        total_fees = totals_df['total_fees'].sum()
        gross_pnl = totals_df['gross_pnl'].sum()
        fee_ratio = (total_fees / abs(gross_pnl) * 100) if gross_pnl != 0 else 0

        # Hit rate analysis
        winning_trades = totals_df[totals_df['net_pnl'] > 0]
        losing_trades = totals_df[totals_df['net_pnl'] < 0]

        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0

        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Consistency score (based on daily P&L standard deviation)
        pnl_std = totals_df['net_pnl'].std()
        avg_pnl = totals_df['net_pnl'].mean()
        consistency_score = 100 / (1 + pnl_std / max(abs(avg_pnl), 1))

        return {
            'fee_efficiency': fee_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'consistency_score': consistency_score,
            'total_fees_paid': total_fees
        }

    def generate_trader_analytics(self, account_id: str, lookback_days: int = 30) -> Dict:
        """Generate comprehensive analytics for a single trader"""

        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Get trader data
        totals_df, fills_df = self.db.get_trader_data(
            account_id,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if totals_df.empty:
            return {'account_id': account_id, 'error': 'No data available'}

        # Calculate all metrics
        performance = self.calculate_performance_metrics(totals_df)
        risk = self.calculate_risk_metrics(totals_df)
        behavior = self.calculate_behavior_metrics(totals_df, fills_df)
        efficiency = self.calculate_efficiency_metrics(totals_df)

        return {
            'account_id': account_id,
            'period_days': lookback_days,
            'data_date_range': (totals_df['date'].min(), totals_df['date'].max()),
            'performance': performance,
            'risk': risk,
            'behavior': behavior,
            'efficiency': efficiency
        }

    def generate_peer_comparison(self, lookback_days: int = 30) -> Dict:
        """Generate peer comparison analytics"""

        traders_df = self.db.get_all_traders()
        all_analytics = []

        for _, trader in traders_df.iterrows():
            analytics = self.generate_trader_analytics(str(trader['account_id']), lookback_days)
            if 'error' not in analytics:
                analytics['trader_name'] = trader['trader_name']
                all_analytics.append(analytics)

        if not all_analytics:
            return {}

        # Create comparison DataFrame
        comparison_data = []
        for analytics in all_analytics:
            if analytics['performance']:
                comparison_data.append({
                    'account_id': analytics['account_id'],
                    'trader_name': analytics['trader_name'],
                    'total_pnl': analytics['performance']['total_pnl'],
                    'win_rate': analytics['performance']['win_rate'],
                    'sharpe_ratio': analytics['performance']['sharpe_ratio'],
                    'profit_factor': analytics['performance']['profit_factor'],
                    'max_drawdown': analytics['performance']['max_drawdown'],
                    'avg_daily_pnl': analytics['performance']['avg_daily_pnl'],
                    'consistency_score': analytics['efficiency']['consistency_score']
                })

        if not comparison_data:
            return {}

        df = pd.DataFrame(comparison_data)

        # Calculate rankings and percentiles
        rankings = {}
        for metric in ['total_pnl', 'win_rate', 'sharpe_ratio', 'profit_factor', 'consistency_score']:
            df[f'{metric}_rank'] = df[metric].rank(ascending=False, method='min')
            df[f'{metric}_percentile'] = df[metric].rank(pct=True) * 100

        # Max drawdown ranking (lower is better)
        df['max_drawdown_rank'] = df['max_drawdown'].rank(ascending=True, method='min')
        df['max_drawdown_percentile'] = (1 - df['max_drawdown'].rank(pct=True)) * 100

        return {
            'total_traders': len(df),
            'comparison_data': df.to_dict('records'),
            'top_performers': {
                'by_pnl': df.nlargest(3, 'total_pnl')[['trader_name', 'total_pnl']].to_dict('records'),
                'by_sharpe': df.nlargest(3, 'sharpe_ratio')[['trader_name', 'sharpe_ratio']].to_dict('records'),
                'by_consistency': df.nlargest(3, 'consistency_score')[['trader_name', 'consistency_score']].to_dict('records')
            }
        }

    def create_performance_chart(self, totals_df: pd.DataFrame, trader_name: str) -> str:
        """Create performance visualization"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Cumulative P&L
        cumulative_pnl = totals_df['net_pnl'].cumsum()
        axes[0,0].plot(totals_df['date'], cumulative_pnl, linewidth=2, color='steelblue')
        axes[0,0].set_title(f'Cumulative P&L - {trader_name}')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)

        # Daily P&L histogram
        axes[0,1].hist(totals_df['net_pnl'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].axvline(totals_df['net_pnl'].mean(), color='red', linestyle='--', label=f'Mean: ${totals_df["net_pnl"].mean():.2f}')
        axes[0,1].set_title('Daily P&L Distribution')
        axes[0,1].set_xlabel('Daily P&L ($)')
        axes[0,1].legend()

        # Rolling metrics
        window = min(7, len(totals_df) // 4)
        if window > 1:
            rolling_pnl = totals_df['net_pnl'].rolling(window).mean()
            rolling_vol = totals_df['net_pnl'].rolling(window).std()

            axes[1,0].plot(totals_df['date'], rolling_pnl, label=f'{window}-day Avg P&L', color='green')
            axes[1,0].set_title('Rolling Performance')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].tick_params(axis='x', rotation=45)

            axes[1,1].plot(totals_df['date'], rolling_vol, label=f'{window}-day Volatility', color='orange')
            axes[1,1].set_title('Rolling Volatility')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save and return base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    def create_comparison_chart(self, comparison_data: List[Dict]) -> str:
        """Create peer comparison visualization"""

        if not comparison_data:
            return ""

        df = pd.DataFrame(comparison_data)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Total P&L ranking
        top_10 = df.nlargest(10, 'total_pnl')
        axes[0,0].barh(range(len(top_10)), top_10['total_pnl'], color='steelblue')
        axes[0,0].set_yticks(range(len(top_10)))
        axes[0,0].set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in top_10['trader_name']])
        axes[0,0].set_title('Top 10 - Total P&L')
        axes[0,0].set_xlabel('P&L ($)')

        # Win Rate vs Sharpe Ratio scatter
        axes[0,1].scatter(df['win_rate'], df['sharpe_ratio'], alpha=0.7, color='lightcoral')
        axes[0,1].set_xlabel('Win Rate (%)')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].set_title('Win Rate vs Sharpe Ratio')
        axes[0,1].grid(True, alpha=0.3)

        # Profit Factor distribution
        axes[1,0].hist(df['profit_factor'].clip(0, 5), bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,0].set_xlabel('Profit Factor')
        axes[1,0].set_ylabel('Number of Traders')
        axes[1,0].set_title('Profit Factor Distribution')

        # Consistency vs Performance
        axes[1,1].scatter(df['consistency_score'], df['total_pnl'], alpha=0.7, color='gold')
        axes[1,1].set_xlabel('Consistency Score')
        axes[1,1].set_ylabel('Total P&L ($)')
        axes[1,1].set_title('Consistency vs Performance')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save and return base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    def calculate_advanced_trading_metrics(self, totals_df):
        """Calculate state-of-the-art trading metrics for professional day traders"""

        if totals_df.empty or len(totals_df) < 5:  # Need minimum 5 days
            print(f"Insufficient data for advanced metrics: {len(totals_df)} days")
            return {}

        daily_returns = totals_df['net_pnl']

        # Check if we have any non-zero returns
        non_zero_returns = daily_returns[daily_returns != 0]
        if len(non_zero_returns) < 3:
            print("Insufficient non-zero returns for advanced metrics")
            return {}

        print(f"Calculating advanced metrics for {len(daily_returns)} days, {len(non_zero_returns)} non-zero")

        # Simple Omega Ratio (gains vs losses)
        gains = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]

        if len(losses) > 0 and losses.sum() < 0:
            omega_ratio = gains.sum() / abs(losses.sum())
        elif len(gains) > 0:
            omega_ratio = 5.0  # High value when no losses
        else:
            omega_ratio = 1.0

        omega_ratio = min(omega_ratio, 10.0)  # Cap at reasonable value

        # Simple Hurst Exponent (autocorrelation proxy)
        hurst_exp = 0.5  # Default to random walk
        if len(daily_returns) >= 7:
            try:
                autocorr = daily_returns.autocorr(lag=1)
                if not pd.isna(autocorr):
                    hurst_exp = 0.5 + autocorr * 0.3  # Scale to reasonable range
                    hurst_exp = max(0.1, min(0.9, hurst_exp))  # Clamp
            except:
                hurst_exp = 0.5

        # Information Coefficient (simple momentum correlation)
        information_coefficient = 0
        if len(daily_returns) > 3:
            try:
                momentum_signal = daily_returns.rolling(2, min_periods=1).mean().shift(1)
                actual_direction = (daily_returns > 0).astype(int)
                predicted_direction = (momentum_signal > 0).astype(int)

                valid_mask = ~(pd.isna(momentum_signal) | pd.isna(daily_returns))
                if valid_mask.sum() > 2:
                    correlation = np.corrcoef(predicted_direction[valid_mask], actual_direction[valid_mask])[0, 1]
                    information_coefficient = correlation if not np.isnan(correlation) else 0
            except:
                information_coefficient = 0

        # Tail Expectation (worst 20% of days)
        tail_threshold = np.percentile(daily_returns, 20)
        tail_returns = daily_returns[daily_returns <= tail_threshold]
        tail_expectation = tail_returns.mean() if len(tail_returns) > 0 else daily_returns.min()

        # Sterling Ratio (return per average drawdown)
        cumulative_pnl = daily_returns.cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdowns = cumulative_pnl - rolling_max

        avg_drawdown = abs(drawdowns.mean()) if len(drawdowns) > 0 else 1
        if avg_drawdown == 0:
            avg_drawdown = 1

        annualized_return = daily_returns.mean() * 252
        sterling_ratio = annualized_return / avg_drawdown

        print(f"Advanced metrics calculated - Omega: {omega_ratio:.2f}, Hurst: {hurst_exp:.3f}, IC: {information_coefficient:.3f}")

        return {
            'omega_ratio': omega_ratio,
            'hurst_exponent': hurst_exp,
            'information_coefficient': information_coefficient,
            'tail_expectation': tail_expectation,
            'sterling_ratio': sterling_ratio,
            'current_rolling_sharpe': 0,
            'avg_rolling_sharpe': 0,
            'sharpe_stability': 0,
            'kappa3': 0,
            'returns_skewness': 0,
            'returns_kurtosis': 0,
            'upside_capture': 1.0,
            'downside_capture': 1.0,
            'burke_ratio': 0,
            'martin_ratio': 0
        }


    def calculate_advanced_trading_metrics(self, totals_df):
        """Calculate advanced trading metrics - handles edge cases properly"""

        if totals_df.empty or len(totals_df) < 3:
            return self._empty_advanced_metrics()

        daily_returns = totals_df['net_pnl']
        print(f"DEBUG: Calculating advanced metrics for {len(daily_returns)} days")
        print(f"DEBUG: P&L range: ${daily_returns.min():.2f} to ${daily_returns.max():.2f}")

        # Basic stats
        gains = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]

        print(f"DEBUG: {len(gains)} winning days, {len(losses)} losing days")

        # 1. Omega Ratio (handles no-loss scenario)
        if len(losses) > 0 and losses.sum() < 0:
            omega_ratio = gains.sum() / abs(losses.sum())
        elif len(gains) > 0:
            omega_ratio = 10.0  # Excellent - no losses
        else:
            omega_ratio = 1.0

        omega_ratio = min(omega_ratio, 10.0)
        print(f"DEBUG: Omega ratio = {omega_ratio:.2f}")

        # 2. Hurst Exponent (trend vs mean reversion)
        hurst_exp = 0.5  # Default random walk
        if len(daily_returns) >= 5:
            try:
                # Simple autocorrelation approach
                autocorr = daily_returns.autocorr(lag=1)
                if not pd.isna(autocorr):
                    hurst_exp = 0.5 + autocorr * 0.3
                    hurst_exp = max(0.1, min(0.9, hurst_exp))
            except:
                pass

        print(f"DEBUG: Hurst exponent = {hurst_exp:.3f}")

        # 3. Information Coefficient (market timing)
        information_coefficient = 0
        if len(daily_returns) > 3:
            try:
                # Simple momentum signal
                momentum = daily_returns.rolling(2, min_periods=1).mean().shift(1)
                actual_up = (daily_returns > daily_returns.median()).astype(int)
                predicted_up = (momentum > momentum.median()).astype(int)

                valid_mask = ~pd.isna(momentum)
                if valid_mask.sum() > 2:
                    correlation = np.corrcoef(predicted_up[valid_mask], actual_up[valid_mask])[0, 1]
                    information_coefficient = correlation if not np.isnan(correlation) else 0
            except:
                pass

        print(f"DEBUG: Information coefficient = {information_coefficient:.3f}")

        # 4. Sortino Ratio (handles no downside case)
        mean_return = daily_returns.mean()
        if len(losses) > 0:
            downside_deviation = losses.std()
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        else:
            # All gains - use overall volatility but give high score
            volatility = daily_returns.std()
            sortino_ratio = mean_return / (volatility * 0.5) if volatility > 0 else 5.0

        print(f"DEBUG: Sortino ratio = {sortino_ratio:.3f}")

        # 5. Calmar Ratio (handles no drawdown case)
        cumulative = daily_returns.cumsum()
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        max_drawdown = drawdown.min()

        annualized_return = mean_return * 252
        if max_drawdown < -1:  # Meaningful drawdown
            calmar_ratio = annualized_return / abs(max_drawdown)
        elif annualized_return > 0:
            calmar_ratio = 20.0  # High value for no drawdown
        else:
            calmar_ratio = 0

        print(f"DEBUG: Calmar ratio = {calmar_ratio:.3f}")

        # 6. Kelly Criterion (handles all-win case)
        kelly_criterion = 0
        if len(gains) > 0:
            win_rate = len(gains) / len(daily_returns)
            avg_win = gains.mean()

            if len(losses) > 0:
                avg_loss = abs(losses.mean())
                kelly_criterion = win_rate - ((1 - win_rate) * avg_win / avg_loss)
            else:
                # All wins case - use conservative estimate
                estimated_loss = avg_win * 0.2  # Assume 20% loss potential
                kelly_criterion = win_rate - ((1 - win_rate) * avg_win / estimated_loss)

            kelly_criterion = max(0, min(kelly_criterion, 0.5))  # Conservative cap

        print(f"DEBUG: Kelly criterion = {kelly_criterion:.3f}")

        # 7. Sterling Ratio
        avg_drawdown = abs(drawdown.mean()) if len(drawdown) > 0 else 1
        if avg_drawdown == 0:
            avg_drawdown = 1
        sterling_ratio = annualized_return / avg_drawdown

        # 8. Tail Expectation
        tail_percentile = max(10, 100 // max(len(daily_returns), 1))
        tail_threshold = np.percentile(daily_returns, tail_percentile)
        tail_returns = daily_returns[daily_returns <= tail_threshold]
        tail_expectation = tail_returns.mean() if len(tail_returns) > 0 else daily_returns.min()

        result = {
            'omega_ratio': round(omega_ratio, 3),
            'hurst_exponent': round(hurst_exp, 3),
            'information_coefficient': round(information_coefficient, 3),
            'sortino_ratio': round(sortino_ratio, 3),
            'calmar_ratio': round(calmar_ratio, 3),
            'kelly_criterion': round(kelly_criterion, 3),
            'sterling_ratio': round(sterling_ratio, 3),
            'tail_expectation': round(tail_expectation, 2),
            'current_rolling_sharpe': 0,
            'avg_rolling_sharpe': 0,
            'sharpe_stability': 0,
            'kappa3': 0,
            'returns_skewness': 0,
            'returns_kurtosis': 0,
            'upside_capture': 1.0,
            'downside_capture': 1.0,
            'burke_ratio': 0,
            'martin_ratio': 0
        }

        print(f"DEBUG: Returning {len(result)} advanced metrics")
        return result

    def _empty_advanced_metrics(self):
        """Return empty advanced metrics"""
        return {
            'omega_ratio': 0,
            'hurst_exponent': 0.5,
            'information_coefficient': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'kelly_criterion': 0,
            'sterling_ratio': 0,
            'tail_expectation': 0,
            'current_rolling_sharpe': 0,
            'avg_rolling_sharpe': 0,
            'sharpe_stability': 0,
            'kappa3': 0,
            'returns_skewness': 0,
            'returns_kurtosis': 0,
            'upside_capture': 1.0,
            'downside_capture': 1.0,
            'burke_ratio': 0,
            'martin_ratio': 0
        }
