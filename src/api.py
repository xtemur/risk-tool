"""FastAPI application for Trader Risk Management System."""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import os
import pickle
from pathlib import Path

from config import get_config
from utils import get_logger, log_signal_generation, log_error, log_system_event
from models.signal_generator import DeploymentReadySignals


# Initialize configuration and logger
config = get_config()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trader Risk Management API",
    description="Production API for generating daily risk signals for traders",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# Pydantic models
class TraderSignalRequest(BaseModel):
    """Request model for single trader signal."""
    trader_id: str = Field(..., description="Unique trader identifier")
    signal_date: Optional[date] = Field(
        default=None,
        description="Date for signal generation (defaults to today)"
    )
    include_features: bool = Field(
        default=False,
        description="Include feature values in response"
    )


class BatchSignalRequest(BaseModel):
    """Request model for batch signal generation."""
    trader_ids: List[str] = Field(..., description="List of trader IDs")
    signal_date: Optional[date] = Field(
        default=None,
        description="Date for signal generation (defaults to today)"
    )


class RiskSignal(BaseModel):
    """Response model for risk signal."""
    trader_id: str
    signal_date: str
    risk_level: int = Field(..., description="0=Low, 1=Neutral, 2=High")
    risk_label: str = Field(..., description="Human-readable risk label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    features: Optional[Dict[str, float]] = None


class SignalBatch(BaseModel):
    """Response model for batch signals."""
    generated_at: datetime
    signal_count: int
    signals: List[RiskSignal]


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    models_loaded: int
    last_signal_generated: Optional[datetime] = None
    version: str


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    trader_id: str
    accuracy: float
    f1_score: float
    last_updated: datetime
    training_samples: int
    feature_importance: Dict[str, float]


# Global signal generator instance
signal_generator: Optional[DeploymentReadySignals] = None


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify API key if required."""
    if not config.api.api_key_required:
        return True

    api_key = os.getenv("RISK_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    if credentials.credentials != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global signal_generator

    try:
        logger.info("Starting Trader Risk Management API")
        log_system_event("api_startup", "Risk Management API starting up", {})

        # Initialize signal generator
        signal_generator = DeploymentReadySignals()

        # Load models
        models_loaded = signal_generator.load_production_models()

        logger.info(f"API started successfully, {models_loaded} models loaded")
        log_system_event(
            "api_ready",
            "Risk Management API ready",
            {"models_loaded": models_loaded}
        )

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        log_error("api_startup_failed", str(e), {"component": "api"})
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Trader Risk Management API")
    log_system_event("api_shutdown", "Risk Management API shutting down", {})


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Trader Risk Management API",
        "version": config.app.version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        if signal_generator is None:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        return HealthCheck(
            status="healthy",
            timestamp=datetime.utcnow(),
            models_loaded=len(signal_generator.models),
            last_signal_generated=signal_generator.last_signal_time,
            version=config.app.version
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/signals/generate", response_model=RiskSignal)
async def generate_signal(
    request: TraderSignalRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Generate risk signal for a single trader."""
    try:
        if signal_generator is None:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        # Generate signal
        signal_date = request.signal_date or date.today()
        signal_data = signal_generator.generate_signal_for_trader(
            trader_id=request.trader_id,
            signal_date=signal_date,
            include_features=request.include_features
        )

        if signal_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for trader {request.trader_id}"
            )

        # Log signal generation
        log_signal_generation(
            trader_id=request.trader_id,
            signal=signal_data['risk_level'],
            confidence=signal_data['confidence'],
            features=signal_data.get('features')
        )

        # Create response
        return RiskSignal(
            trader_id=request.trader_id,
            signal_date=str(signal_date),
            risk_level=signal_data['risk_level'],
            risk_label=signal_data['risk_label'],
            confidence=signal_data['confidence'],
            recommendation=signal_data['recommendation'],
            features=signal_data.get('features')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate signal: {e}")
        log_error(
            "signal_generation_failed",
            str(e),
            {"trader_id": request.trader_id}
        )
        raise HTTPException(status_code=500, detail="Signal generation failed")


@app.post("/signals/batch", response_model=SignalBatch)
async def generate_batch_signals(
    request: BatchSignalRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Generate risk signals for multiple traders."""
    try:
        if signal_generator is None:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        signal_date = request.signal_date or date.today()
        signals = []

        for trader_id in request.trader_ids:
            try:
                signal_data = signal_generator.generate_signal_for_trader(
                    trader_id=trader_id,
                    signal_date=signal_date,
                    include_features=False
                )

                if signal_data:
                    signals.append(RiskSignal(
                        trader_id=trader_id,
                        signal_date=str(signal_date),
                        risk_level=signal_data['risk_level'],
                        risk_label=signal_data['risk_label'],
                        confidence=signal_data['confidence'],
                        recommendation=signal_data['recommendation']
                    ))

            except Exception as e:
                logger.warning(f"Failed to generate signal for {trader_id}: {e}")

        return SignalBatch(
            generated_at=datetime.utcnow(),
            signal_count=len(signals),
            signals=signals
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch signal generation failed: {e}")
        raise HTTPException(status_code=500, detail="Batch generation failed")


@app.get("/models", response_model=List[str])
async def list_models(authenticated: bool = Depends(verify_api_key)):
    """List all available trader models."""
    try:
        if signal_generator is None:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        return list(signal_generator.models.keys())

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model list")


@app.get("/models/{trader_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(
    trader_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """Get performance metrics for a specific trader model."""
    try:
        if signal_generator is None:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        metrics = signal_generator.get_model_metrics(trader_id)

        if metrics is None:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for trader {trader_id}"
            )

        return ModelMetrics(**metrics)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.post("/models/reload")
async def reload_models(authenticated: bool = Depends(verify_api_key)):
    """Reload all models from disk."""
    try:
        if signal_generator is None:
            raise HTTPException(status_code=503, detail="Signal generator not initialized")

        models_loaded = signal_generator.reload_models()

        log_system_event(
            "models_reloaded",
            "Models reloaded successfully",
            {"models_loaded": models_loaded}
        )

        return {
            "status": "success",
            "models_loaded": models_loaded,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Failed to reload models: {e}")
        raise HTTPException(status_code=500, detail="Model reload failed")


# Add custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {"error": "Resource not found", "path": str(request.url)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    log_error("internal_server_error", str(exc), {"path": str(request.url)})
    return {"error": "Internal server error", "message": "Please try again later"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.app.environment == "development"
    )
