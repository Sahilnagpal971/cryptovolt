"""Health check routes"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "CryptoVolt API"}


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CryptoVolt API",
        "version": "1.0.0",
        "description": "AI-based algorithmic trading system"
    }
