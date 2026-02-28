"""
Database utility functions for common operations
"""
import logging
from typing import List, Optional, Type, TypeVar
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)

# Generic type for database models
T = TypeVar('T')


class DatabaseService:
    """
    Generic database service for CRUD operations
    """
    
    @staticmethod
    def create(db: Session, model_class: Type[T], **kwargs) -> T:
        """Create a new record"""
        try:
            db_object = model_class(**kwargs)
            db.add(db_object)
            db.commit()
            db.refresh(db_object)
            logger.info(f"Created new {model_class.__name__}: {db_object}")
            return db_object
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating {model_class.__name__}: {str(e)}")
            raise
    
    @staticmethod
    def read(db: Session, model_class: Type[T], id: int) -> Optional[T]:
        """Read a record by ID"""
        try:
            return db.query(model_class).filter(
                model_class.__table__.c.get(f"{model_class.__tablename__[:-1]}_id") == id
            ).first()
        except Exception as e:
            logger.error(f"Error reading {model_class.__name__}: {str(e)}")
            return None
    
    @staticmethod
    def update(db: Session, model_class: Type[T], id: int, **kwargs) -> Optional[T]:
        """Update a record"""
        try:
            db_object = db.query(model_class).filter(
                model_class.__table__.c.get(f"{model_class.__tablename__[:-1]}_id") == id
            ).first()
            
            if db_object:
                for key, value in kwargs.items():
                    setattr(db_object, key, value)
                db.commit()
                db.refresh(db_object)
                logger.info(f"Updated {model_class.__name__} {id}")
            
            return db_object
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating {model_class.__name__}: {str(e)}")
            raise
    
    @staticmethod
    def delete(db: Session, model_class: Type[T], id: int) -> bool:
        """Delete a record"""
        try:
            db_object = db.query(model_class).filter(
                model_class.__table__.c.get(f"{model_class.__tablename__[:-1]}_id") == id
            ).first()
            
            if db_object:
                db.delete(db_object)
                db.commit()
                logger.info(f"Deleted {model_class.__name__} {id}")
                return True
            
            return False
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting {model_class.__name__}: {str(e)}")
            raise
    
    @staticmethod
    def list_all(db: Session, model_class: Type[T], skip: int = 0, limit: int = 100) -> List[T]:
        """List all records with pagination"""
        try:
            return db.query(model_class).offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error listing {model_class.__name__}: {str(e)}")
            return []
    
    @staticmethod
    def bulk_create(db: Session, model_class: Type[T], objects: List[dict]) -> List[T]:
        """Create multiple records"""
        try:
            instances = [model_class(**obj) for obj in objects]
            db.add_all(instances)
            db.commit()
            logger.info(f"Created {len(instances)} new {model_class.__name__} records")
            return instances
        except Exception as e:
            db.rollback()
            logger.error(f"Error bulk creating {model_class.__name__}: {str(e)}")
            raise


class QueryHelper:
    """
    Helper class for building complex queries
    """
    
    @staticmethod
    def filter_by_symbol(db: Session, model_class: Type[T], symbol: str) -> List[T]:
        """Filter records by symbol (for market data, sentiment, etc.)"""
        try:
            return db.query(model_class).filter(model_class.symbol == symbol).all()
        except Exception as e:
            logger.error(f"Error filtering by symbol: {str(e)}")
            return []
    
    @staticmethod
    def filter_by_date_range(db: Session, model_class: Type[T], start_date, end_date, date_column: str = "timestamp"):
        """Filter records by date range"""
        try:
            return db.query(model_class).filter(
                and_(
                    getattr(model_class, date_column) >= start_date,
                    getattr(model_class, date_column) <= end_date
                )
            ).all()
        except Exception as e:
            logger.error(f"Error filtering by date range: {str(e)}")
            return []
    
    @staticmethod
    def search_text(db: Session, model_class: Type[T], search_term: str, text_columns: List[str]) -> List[T]:
        """Search across multiple text columns"""
        try:
            filters = [
                getattr(model_class, col).ilike(f"%{search_term}%") 
                for col in text_columns 
                if hasattr(model_class, col)
            ]
            return db.query(model_class).filter(or_(*filters)).all()
        except Exception as e:
            logger.error(f"Error searching text: {str(e)}")
            return []


class DatabaseStats:
    """
    Get database and table statistics
    """
    
    @staticmethod
    def get_table_row_count(db: Session, model_class: Type[T]) -> int:
        """Get row count for a table"""
        try:
            return db.query(model_class).count()
        except Exception as e:
            logger.error(f"Error getting row count: {str(e)}")
            return 0
    
    @staticmethod
    def get_all_table_stats(db: Session, models: List[Type[T]]) -> dict:
        """Get statistics for all tables"""
        stats = {}
        for model in models:
            try:
                stats[model.__tablename__] = db.query(model).count()
            except Exception as e:
                logger.error(f"Error getting stats for {model.__tablename__}: {str(e)}")
                stats[model.__tablename__] = 0
        return stats


def get_db_session() -> Session:
    """Get a database session (for use in scripts)"""
    return SessionLocal()


def close_db_session(db: Session):
    """Close a database session"""
    if db:
        db.close()
