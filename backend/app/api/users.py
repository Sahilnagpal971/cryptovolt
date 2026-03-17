"""User management routes"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.base import UserCreate, UserResponse
from app.models.database import User
from app.core.security import hash_password

router = APIRouter()


@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    existing = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username or email already registered")

    db_user = User(
        username=user.username,
        email=user.email,
        password_hash=hash_password(user.password),
        is_active=True,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
