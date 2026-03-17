"""Alert management routes"""
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import AlertCreate, AlertResponse
from app.models.database import Alert, User
from fastapi import HTTPException

router = APIRouter()


@router.post("/", response_model=AlertResponse)
async def create_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    """Create new alert"""
    user = db.query(User).filter(User.user_id == alert.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    now = datetime.utcnow()
    db_alert = Alert(
        user_id=alert.user_id,
        alert_type=alert.alert_type,
        message=alert.message,
        is_read=False,
        timestamp=now,
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)

    return {
        "alert_id": db_alert.alert_id,
        "user_id": db_alert.user_id,
        "alert_type": db_alert.alert_type,
        "message": db_alert.message,
        "is_read": db_alert.is_read,
        "timestamp": db_alert.timestamp,
        "created_at": db_alert.created_at,
    }


@router.get("/{user_id}")
async def get_alerts(user_id: int, limit: int = 50, db: Session = Depends(get_db)):
    """Get alerts for user"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    rows = (
        db.query(Alert)
        .filter(Alert.user_id == user_id)
        .order_by(Alert.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "user_id": user_id,
        "alerts": [
            {
                "alert_id": r.alert_id,
                "user_id": r.user_id,
                "alert_type": r.alert_type,
                "message": r.message,
                "is_read": r.is_read,
                "timestamp": r.timestamp,
                "created_at": r.created_at,
            }
            for r in rows
        ],
    }
