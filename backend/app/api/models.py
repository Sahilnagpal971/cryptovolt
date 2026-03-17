"""ML model management routes"""
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import ModelCreate, ModelResponse
from app.models.database import Model
from fastapi import HTTPException

router = APIRouter()


@router.post("/", response_model=ModelResponse)
async def register_model(model: ModelCreate, db: Session = Depends(get_db)):
    """Register new ML model"""
    db_model = Model(
        name=model.name,
        model_type=model.model_type,
        version=model.version,
        accuracy=model.accuracy,
        precision=model.precision,
        recall=model.recall,
        auc=model.auc,
        trained_on=datetime.utcnow(),
        extra_data=model.metadata,
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    return {
        "model_id": db_model.model_id,
        "name": db_model.name,
        "model_type": db_model.model_type,
        "version": db_model.version,
        "accuracy": db_model.accuracy,
        "precision": db_model.precision,
        "recall": db_model.recall,
        "auc": db_model.auc,
        "metadata": db_model.extra_data,
        "trained_on": db_model.trained_on,
        "created_at": db_model.created_at,
    }


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get model by ID"""
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "model_id": model.model_id,
        "name": model.name,
        "model_type": model.model_type,
        "version": model.version,
        "accuracy": model.accuracy,
        "precision": model.precision,
        "recall": model.recall,
        "auc": model.auc,
        "metadata": model.extra_data,
        "trained_on": model.trained_on,
        "created_at": model.created_at,
    }
