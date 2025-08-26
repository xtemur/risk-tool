# Refactoring Guide - Risk Tool

## Overview
This guide documents the refactoring of the risk tool codebase to improve maintainability, testability, and scalability. The refactoring follows clean architecture principles and SOLID design patterns.

## Completed Refactoring

### 1. Repository Pattern Implementation ✅
**Location:** `src/repositories/`

- **Base Repository** (`base.py`): Abstract interface for all repositories
- **TraderRepository** (`trader_repository.py`): Handles all trader data access
- **ModelRepository** (`model_repository.py`): Manages ML model storage and retrieval
- **FillsRepository** (`fills_repository.py`): Manages trading fills/transactions

**Benefits:**
- Centralized data access logic
- Easy to mock for testing
- Consistent interface across all data operations

### 2. Domain Models ✅
**Location:** `src/models/domain/`

- **Trader Models** (`trader.py`): `Trader`, `TradingMetrics`, `TraderProfile`
- **Risk Models** (`risk.py`): `RiskAssessment`, `RiskLevel`, `RiskAlert`
- **Prediction Models** (`prediction.py`): `Prediction`, `PredictionResult`

**Benefits:**
- Strong typing and validation
- Business logic encapsulated in models
- Clear data structures

### 3. Service Layer ✅
**Location:** `src/services/`

- **RiskService**: Risk assessment business logic
- **TraderService**: Trader management operations
- **PredictionService**: Prediction generation
- **MetricsService**: Metrics calculation

**Benefits:**
- Separation of business logic from infrastructure
- Reusable service methods
- Easy to test business logic

### 4. Dependency Injection ✅
**Location:** `src/container.py`

- **ServiceContainer**: Central DI container
- Singleton pattern for service instances
- Easy service resolution

**Benefits:**
- Loose coupling between components
- Easy to swap implementations
- Simplified testing with mock services

### 5. Configuration Management ✅
**Location:** `src/config/config_manager.py`

- **ConfigManager**: Centralized configuration
- Environment variable support
- Configuration validation

**Benefits:**
- Single source of truth for config
- Environment-specific settings
- Runtime configuration updates

### 6. Custom Exceptions ✅
**Location:** `src/exceptions.py`

- Domain-specific exceptions
- Detailed error information
- Consistent error handling

**Benefits:**
- Clear error semantics
- Better debugging
- Proper error propagation

### 7. Constants Extraction ✅
**Location:** `src/constants.py`

- All magic numbers and strings extracted
- Centralized thresholds and limits
- Path configurations

**Benefits:**
- Easy to modify thresholds
- No hardcoded values
- Better maintainability

## Migration Guide

### Step 1: Update Imports
Replace direct database access with repositories:

```python
# Before
import sqlite3
conn = sqlite3.connect('data/risk_tool.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM traders WHERE id = ?", (trader_id,))

# After
from src.container import get_container
container = get_container()
trader = container.trader_repository.find_by_id(trader_id)
```

### Step 2: Use Service Layer
Replace business logic with service calls:

```python
# Before
def calculate_risk(trader_id):
    # 100 lines of mixed data access and business logic
    ...

# After
from src.container import get_container
container = get_container()
risk_assessment = container.risk_service.assess_trader_risk(trader_id)
```

### Step 3: Adopt Domain Models
Use domain models instead of dictionaries:

```python
# Before
trader_data = {
    'id': 123,
    'name': 'Trader A',
    'metrics': {...}
}

# After
from src.models.domain import Trader, TradingMetrics
trader = Trader(id=123, name='Trader A')
metrics = TradingMetrics(bat_30d=45.5, wl_ratio=1.2, sharpe=0.8)
```

### Step 4: Handle Exceptions
Use custom exceptions:

```python
# Before
try:
    model = load_model(trader_id)
except Exception as e:
    print(f"Error: {e}")

# After
from src.exceptions import ModelNotFoundError, handle_exception
try:
    model = model_repository.load_model(trader_id)
except ModelNotFoundError as e:
    handle_exception(e, logger)
```

## Example: Refactoring a Function

### Before (Monolithic)
```python
def generate_report(trader_id):
    # Database connection
    conn = sqlite3.connect('data/risk_tool.db')

    # Data fetching
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM fills WHERE trader_id = ?", (trader_id,))
    fills = cursor.fetchall()

    # Business logic
    total_pnl = sum(f[5] for f in fills)  # Magic index
    win_rate = len([f for f in fills if f[5] > 0]) / len(fills) * 100

    # Risk calculation
    if win_rate < 40:  # Magic number
        risk_level = "HIGH"
    else:
        risk_level = "LOW"

    # Report generation
    report = f"Trader {trader_id}: PnL={total_pnl}, Risk={risk_level}"

    return report
```

### After (Clean Architecture)
```python
def generate_report(trader_id: int) -> str:
    # Get container with all dependencies
    container = get_container()

    # Use services for business logic
    trader = container.trader_service.get_trader(trader_id)
    metrics = container.metrics_service.calculate_trader_metrics(trader_id)
    risk_assessment = container.risk_service.assess_trader_risk(trader_id)

    # Format report using domain models
    report = (
        f"Trader {trader.display_name}: "
        f"PnL=${metrics.total_pnl:,.2f}, "
        f"Risk={risk_assessment.risk_level.value}"
    )

    return report
```

## Testing Strategy

### Unit Tests
Test individual components in isolation:

```python
def test_trader_repository():
    # Mock database
    mock_db = Mock()
    repo = TraderRepository(mock_db)

    # Test find_by_id
    trader = repo.find_by_id(123)
    assert trader.id == 123
```

### Integration Tests
Test service interactions:

```python
def test_risk_service():
    # Create test container
    container = ServiceContainer(env='test')

    # Test risk assessment
    assessment = container.risk_service.assess_trader_risk(123)
    assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
```

## Next Steps

### Still To Refactor
1. **SignalGenerator** - Break down the monolithic class
2. **CausalImpactEvaluation** - Split into smaller services
3. **TraderSpecificTrainer** - Apply strategy pattern

### Recommended Improvements
1. Add caching layer for frequently accessed data
2. Implement event-driven architecture for real-time updates
3. Add API layer with proper REST endpoints
4. Implement comprehensive logging strategy
5. Add monitoring and metrics collection

## Performance Considerations

### Caching
The `ModelRepository` includes basic caching:
```python
model_repo.clear_cache()  # Clear when needed
```

### Database Connections
Repositories manage connections efficiently:
- Use context managers for automatic cleanup
- Connection pooling for high-throughput scenarios

### Lazy Loading
Services load data only when needed:
```python
profile = trader_service.get_trader_profile(trader_id)  # Loads on demand
```

## Conclusion

This refactoring provides:
- **Better Maintainability**: Clear separation of concerns
- **Improved Testability**: Easy to mock and test
- **Enhanced Scalability**: Easy to add new features
- **Reduced Coupling**: Components are independent
- **Better Documentation**: Clear interfaces and types

The codebase is now more professional, maintainable, and ready for future enhancements.
