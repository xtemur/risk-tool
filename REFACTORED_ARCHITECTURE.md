# Refactored Risk Tool Architecture

## Overview
This document describes the refactored clean architecture implementation for the risk tool, following SOLID principles and industry best practices.

## Architecture Layers

### 1. Domain Layer (`src/models/domain/`)
Pure business logic and entities with no external dependencies.

**Components:**
- `trader.py` - Trader, TradingMetrics, TraderProfile entities
- `risk.py` - RiskAssessment, RiskLevel, RiskAlert entities
- `prediction.py` - Prediction, PredictionResult, PredictionStatus entities

**Key Features:**
- Immutable data classes
- Business logic encapsulated in entities
- No framework dependencies

### 2. Data Access Layer (`src/repositories/`)
Abstracts all data access behind a consistent interface.

**Components:**
- `base.py` - Abstract Repository interface
- `trader_repository.py` - Trader data access
- `model_repository.py` - ML model storage/retrieval
- `fills_repository.py` - Trading fills/transactions

**Key Features:**
- Consistent CRUD interface
- Database-agnostic design
- Connection management
- Query optimization

### 3. Service Layer (`src/services/`)
Orchestrates business operations using repositories and domain models.

**Components:**
- `risk_service.py` - Risk assessment logic
- `trader_service.py` - Trader management
- `prediction_service.py` - ML predictions
- `metrics_service.py` - Metrics calculations

**Key Features:**
- Business logic coordination
- Transaction boundaries
- Cross-cutting concerns

### 4. Infrastructure Layer

#### Configuration (`src/config/`)
- `config_manager.py` - Centralized configuration management
- Environment variable support
- Configuration validation

#### Dependency Injection (`src/container.py`)
- `ServiceContainer` - IoC container
- Singleton pattern for services
- Service resolution

#### Exception Handling (`src/exceptions.py`)
- Domain-specific exceptions
- Detailed error information
- Consistent error handling

#### Constants (`src/constants.py`)
- Centralized constants
- Magic number elimination
- Threshold definitions

## Design Patterns Used

### Repository Pattern
Encapsulates data access logic and provides a more object-oriented view of the persistence layer.

```python
from src.repositories import TraderRepository

repo = TraderRepository(db_path)
trader = repo.find_by_id(123)
metrics = repo.get_trader_metrics(123, lookback_days=30)
```

### Dependency Injection
Loose coupling between components through constructor injection.

```python
from src.container import get_container

container = get_container()
risk_service = container.risk_service  # Dependencies injected automatically
```

### Service Layer Pattern
Defines application's boundary and orchestrates business operations.

```python
assessment = risk_service.assess_trader_risk(trader_id)
alerts = risk_service.generate_risk_alerts([assessment])
```

### Domain-Driven Design
Rich domain models with business logic.

```python
metrics = TradingMetrics(bat_30d=45, wl_ratio=1.2, sharpe=0.8)
risk_score = metrics.get_risk_score()  # Business logic in domain
is_high_risk = metrics.is_high_risk()
```

## Usage Examples

### Basic Usage
```python
from src.container import get_container

# Get container with all dependencies
container = get_container(env='production')

# Access services
risk_service = container.risk_service
trader_service = container.trader_service

# Perform operations
trader = trader_service.get_trader(123)
assessment = risk_service.assess_trader_risk(123)
```

### Error Handling
```python
from src.exceptions import TraderNotFoundError, ModelNotFoundError

try:
    assessment = risk_service.assess_trader_risk(trader_id)
except TraderNotFoundError as e:
    logger.error(f"Trader not found: {e.details}")
except ModelNotFoundError as e:
    logger.error(f"Model not found: {e.details}")
```

### Configuration
```python
from src.config import ConfigManager

config = ConfigManager(env='production')
db_path = config.database.path
risk_threshold = config.risk.high_risk_sharpe
```

## Testing Strategy

### Unit Tests
Test individual components in isolation:
```python
def test_trading_metrics():
    metrics = TradingMetrics(bat_30d=25, wl_ratio=0.5, sharpe=-0.2)
    assert metrics.is_high_risk() == True
    assert metrics.get_risk_score() > 70
```

### Integration Tests
Test service interactions:
```python
def test_risk_assessment():
    container = ServiceContainer(env='test')
    assessment = container.risk_service.assess_trader_risk(123)
    assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
```

### Test Coverage
Run comprehensive test suite:
```bash
python tests/test_refactoring.py
```

## Migration Guide

### Step 1: Update Imports
Replace direct imports with container-based access:
```python
# Before
import sqlite3
conn = sqlite3.connect('data/risk_tool.db')

# After
from src.container import get_container
container = get_container()
repo = container.trader_repository
```

### Step 2: Use Domain Models
Replace dictionaries with domain models:
```python
# Before
trader_data = {'id': 123, 'name': 'Trader A'}

# After
from src.models.domain import Trader
trader = Trader(id=123, name='Trader A')
```

### Step 3: Implement Service Layer
Move business logic to services:
```python
# Before
# 100 lines of mixed logic

# After
assessment = risk_service.assess_trader_risk(trader_id)
```

## Benefits

### Maintainability
- Clear separation of concerns
- Single responsibility principle
- Easy to understand and modify

### Testability
- Mock dependencies easily
- Test in isolation
- High test coverage achievable

### Scalability
- Add new features without affecting existing code
- Swap implementations easily
- Database/framework agnostic

### Type Safety
- Strong typing throughout
- IDE autocomplete support
- Catch errors at development time

## File Structure
```
src/
├── config/
│   ├── __init__.py
│   └── config_manager.py
├── models/
│   ├── __init__.py
│   └── domain/
│       ├── __init__.py
│       ├── trader.py
│       ├── risk.py
│       └── prediction.py
├── repositories/
│   ├── __init__.py
│   ├── base.py
│   ├── trader_repository.py
│   ├── model_repository.py
│   └── fills_repository.py
├── services/
│   ├── __init__.py
│   ├── risk_service.py
│   ├── trader_service.py
│   ├── prediction_service.py
│   └── metrics_service.py
├── constants.py
├── container.py
├── exceptions.py
└── refactored_signal_generator.py

tests/
└── test_refactoring.py

examples/
├── __init__.py
└── using_refactored_code.py
```

## Next Steps

1. **Gradual Migration**: Migrate existing components one at a time
2. **Add Caching**: Implement caching layer for frequently accessed data
3. **API Layer**: Add REST API endpoints using the service layer
4. **Event System**: Implement event-driven updates
5. **Monitoring**: Add performance monitoring and metrics

## Conclusion

This refactored architecture provides a solid foundation for maintaining and extending the risk tool. It follows industry best practices, is fully tested, and ready for production use.
