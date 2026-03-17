"""Authentication routes"""
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.config import settings
from app.core.security import create_access_token, verify_password
from app.models.database import User

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    """User login endpoint"""
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive")

    token = create_access_token(
        subject=str(user.user_id),
        secret_key=settings.SECRET_KEY,
        expires_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        extra_claims={"username": user.username},
    )
    return {"access_token": token, "token_type": "bearer"}


@router.post("/logout")
async def logout():
    """User logout endpoint"""
    return {"message": "Successfully logged out"}
