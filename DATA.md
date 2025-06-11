# DATA.md - Risk Tool Database Schema Documentation

## Overview

The risk-tool MVP uses SQLite database with three main tables to store and analyze day trader prop reports. This document explains each column's purpose and identifies which columns are essential for behavioral signal extraction and profit maximization.

## Database Tables

### 1. accounts Table

Stores basic account information and metadata.

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `account_id` | TEXT PRIMARY KEY | Unique identifier for each trader account | **CRITICAL** - Primary key for all joins | **KEEP** |
| `account_name` | TEXT | Human-readable name for the account | **USEFUL** - For reporting and identification | **KEEP** |
| `account_type` | TEXT | Type of account (Equities/Options) | **CRITICAL** - Different account types have different risk profiles | **KEEP** |
| `created_at` | TIMESTAMP | When account was added to system | **MODERATE** - Useful for tracking trader lifecycle | **KEEP** |
| `updated_at` | TIMESTAMP | Last update timestamp | **LOW** - Mainly for debugging | **CONSIDER DROPPING** |

### 2. account_daily_summary Table

Daily aggregated trading metrics from summaryByDate reports. This is the primary table for feature engineering.

#### Core Trading Metrics

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `id` | INTEGER PRIMARY KEY | Auto-incrementing primary key | **SYSTEM** - Database management | **KEEP** |
| `account_id` | TEXT | Foreign key to accounts table | **CRITICAL** - Links to trader | **KEEP** |
| `date` | DATE | Trading date | **CRITICAL** - Time series analysis | **KEEP** |
| `type` | TEXT | Daily type indicator (Eq/Op) | **MODERATE** - Redundant with account_type | **CONSIDER DROPPING** |
| `orders` | INTEGER | Number of orders placed | **HIGH** - Activity indicator, overtrading signal | **KEEP** |
| `fills` | INTEGER | Number of executed trades | **HIGH** - Execution rate, activity level | **KEEP** |
| `qty` | INTEGER | Total shares/contracts traded | **HIGH** - Volume indicator, position sizing behavior | **KEEP** |
| `gross` | REAL | Gross P&L before fees | **CRITICAL** - Raw trading performance | **KEEP** |
| `net` | REAL | Net P&L after all fees | **CRITICAL** - Actual profitability | **KEEP** |

#### Fee Breakdown

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `comm` | REAL | Commission fees | **MODERATE** - Cost analysis | **KEEP** |
| `ecn_fee` | REAL | ECN fees | **LOW** - Minor component | **DROP** |
| `sec` | REAL | SEC fees | **LOW** - Regulatory, predictable | **DROP** |
| `orf` | REAL | ORF fees | **LOW** - Minor component | **DROP** |
| `cat` | REAL | CAT fees | **LOW** - Minor component | **DROP** |
| `taf` | REAL | TAF fees | **LOW** - Minor component | **DROP** |
| `ftt` | REAL | FTT fees | **LOW** - Minor component | **DROP** |
| `nscc` | REAL | NSCC fees | **LOW** - Minor component | **DROP** |
| `acc` | REAL | ACC fees | **LOW** - Minor component | **DROP** |
| `clr` | REAL | CLR fees | **LOW** - Minor component | **DROP** |
| `misc` | REAL | Miscellaneous fees | **LOW** - Unpredictable | **DROP** |
| `trade_fees` | REAL | Total trading fees | **HIGH** - Aggregated cost metric | **KEEP** |

#### Account-Type Specific Fees

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `fee_software_md` | REAL | Software & market data (Equities only) | **MODERATE** - Fixed cost component | **KEEP** |
| `fee_vat` | REAL | VAT fees (Equities only) | **LOW** - Tax component | **DROP** |
| `fee_daily_interest` | REAL | Daily interest (Options only) | **HIGH** - Cost of carry indicator | **KEEP** |

#### Adjusted Metrics

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `adj_fees` | REAL | Adjusted fees | **MODERATE** - Normalized costs | **KEEP** |
| `adj_net` | REAL | Adjusted net P&L | **HIGH** - Normalized performance | **KEEP** |

#### Portfolio Metrics

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `unrealized_delta` | REAL | Change in unrealized P&L | **CRITICAL** - Position holding behavior | **KEEP** |
| `total_delta` | REAL | Total P&L change | **CRITICAL** - Daily performance | **KEEP** |
| `transfer_deposit` | REAL | Deposits/withdrawals | **MODERATE** - Capital management | **KEEP** |
| `transfers` | REAL | Account transfers | **MODERATE** - Capital flows | **KEEP** |
| `cash` | REAL | Cash balance | **HIGH** - Liquidity, margin usage | **KEEP** |
| `unrealized` | REAL | Unrealized P&L | **CRITICAL** - Open position risk | **KEEP** |
| `end_balance` | REAL | End of day balance | **CRITICAL** - Account health | **KEEP** |
| `created_at` | TIMESTAMP | Record creation time | **LOW** - System tracking | **DROP** |

### 3. fills Table

Individual trade executions - granular data for detailed behavioral analysis.

#### Trade Identification

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `id` | INTEGER PRIMARY KEY | Auto-incrementing primary key | **SYSTEM** | **KEEP** |
| `account_id` | TEXT | Foreign key to accounts | **CRITICAL** | **KEEP** |
| `datetime` | TIMESTAMP | Exact execution time | **CRITICAL** - Intraday patterns | **KEEP** |
| `order_id` | TEXT | Order identifier | **MODERATE** - Order analysis | **KEEP** |
| `fill_id` | TEXT | Unique fill identifier | **HIGH** - Deduplication | **KEEP** |
| `propreports_id` | INTEGER | Source system ID | **LOW** - Debugging only | **DROP** |

#### Trade Details

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `side` | TEXT | Buy/Sell/Transfer (B/S/T) | **CRITICAL** - Direction analysis | **KEEP** |
| `quantity` | INTEGER | Number of shares/contracts | **CRITICAL** - Position sizing | **KEEP** |
| `symbol` | TEXT | Trading symbol | **CRITICAL** - Asset selection behavior | **KEEP** |
| `price` | REAL | Execution price | **CRITICAL** - Entry/exit analysis | **KEEP** |
| `route` | TEXT | Execution route | **MODERATE** - Execution quality | **KEEP** |
| `liquidity` | TEXT | Liquidity indicator | **HIGH** - Market impact analysis | **KEEP** |

#### Fees (Granular)

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `commission` | REAL | Trade commission | **MODERATE** | **KEEP** |
| `total_fees` | REAL | Total fees for this trade | **HIGH** - Aggregated metric | **KEEP** |
| `ecn_fee`, `sec_fee`, `orf_fee`, `cat_fee`, `taf_fee`, `ftt_fee`, `nscc_fee`, `acc_fee`, `clr_fee`, `misc_fee` | REAL | Individual fee components | **LOW** - Too granular for MVP | **DROP ALL** |

#### Other Fields

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| `currency` | TEXT | Trading currency | **LOW** - Usually USD | **DROP** |
| `status` | TEXT | Trade status | **MODERATE** - Data quality | **KEEP** |

### 4. data_loads Table (System Table)

Tracks data import history - useful for debugging but not for analysis.

| Column | Type | Description | Usefulness | Action |
|--------|------|-------------|------------|--------|
| All columns | Various | Data load tracking | **SYSTEM ONLY** | **IGNORE FOR ANALYSIS** |

## Recommended Feature Engineering Focus

### High-Value Columns for Behavioral Signals

1. **Performance Metrics**
   - `gross`, `net`, `adj_net` - Core profitability signals
   - `unrealized_delta`, `total_delta` - Risk-taking behavior
   - `unrealized`, `end_balance` - Position management

2. **Activity Indicators**
   - `orders`, `fills`, `qty` - Trading frequency and volume
   - `fills/orders` ratio - Execution efficiency
   - Time patterns from `fills.datetime`

3. **Risk Behavior**
   - `cash`, `unrealized` - Leverage usage
   - `symbol` concentration - Diversification
   - `side` patterns - Directional bias

4. **Cost Awareness**
   - `trade_fees/gross` ratio - Fee sensitivity
   - `commission` patterns - Broker relationship

## Data Quality Considerations

1. **Missing Values**
   - Account-type specific columns will be NULL for other account types
   - This is expected behavior, not missing data

2. **Temporal Consistency**
   - Ensure no look-ahead bias in feature calculation
   - Use only past data for predictions

3. **Cross-Validation**
   - Implement time-series aware CV splits
   - Account for trader lifecycle (some traders may stop trading)

## Implementation Recommendations

1. **Drop Granular Fee Columns**: Individual fee components add complexity without significant predictive value
2. **Create Derived Features**: Ratios and rolling statistics from core columns
3. **Focus on Behavioral Patterns**: Time-of-day analysis, streak analysis, drawdown behavior
4. **Normalize by Account Type**: Equities and Options accounts have different characteristics
5. **Build Features Incrementally**: Start with daily summaries, add fills data for advanced features later

## Next Steps

1. Create feature engineering pipeline focusing on high-value columns
2. Implement data validation to ensure consistency
3. Build behavioral signal extractors from temporal patterns
4. Design risk scoring system based on profitability and volatility metrics
