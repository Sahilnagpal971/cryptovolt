"""
CryptoVolt Backend Main Application
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api import routes
from app.core.database import engine, Base
from app.core.init_db import init_db, verify_database_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events
    """
    # Startup event
    logger.info("Starting CryptoVolt API Server")
    
    # Verify database connection and initialize
    if verify_database_connection():
        init_db()
    else:
        logger.error("⚠ Database connection failed - some features may not work")
    
    yield
    
    # Shutdown event
    logger.info("Shutting down CryptoVolt API Server")


# Create FastAPI application
app = FastAPI(
    title="CryptoVolt API",
    description="AI-based algorithmic trading system with sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trust headers from proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)


# Include routers
app.include_router(routes.health.router)
app.include_router(routes.auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(routes.users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(routes.strategies.router, prefix="/api/v1/strategies", tags=["strategies"])
app.include_router(routes.market_data.router, prefix="/api/v1/market", tags=["market"])
app.include_router(routes.sentiment.router, prefix="/api/v1/sentiment", tags=["sentiment"])
app.include_router(routes.signals.router, prefix="/api/v1/signals", tags=["signals"])
app.include_router(routes.trades.router, prefix="/api/v1/trades", tags=["trades"])
app.include_router(routes.models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(routes.alerts.router, prefix="/api/v1/alerts", tags=["alerts"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
