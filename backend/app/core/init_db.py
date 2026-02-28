"""
Database initialization and setup utilities
"""
import logging
from sqlalchemy import text, inspect
from app.core.database import engine, Base, SessionLocal
from app.models.database import User, TradingStrategy, MarketData, SentimentData, Signal, Trade, Model, Alert

logger = logging.getLogger(__name__)


def init_db():
    """
    Initialize the database by creating all tables if they don't exist
    """
    try:
        logger.info("Initializing database...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created/verified")
        
        # Verify tables exist
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        required_tables = [
            'users',
            'trading_strategies', 
            'market_data',
            'sentiment_data',
            'signals',
            'trades',
            'models',
            'alerts'
        ]
        
        missing_tables = [t for t in required_tables if t not in existing_tables]
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
        else:
            logger.info(f"✓ All required tables exist: {', '.join(required_tables)}")
        
        # Create default admin user if it doesn't exist
        create_default_admin()
        
        logger.info("✓ Database initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
        return False


def create_default_admin():
    """
    Create a default admin user if none exists
    """
    try:
        db = SessionLocal()
        
        # Check if any users exist
        existing_user = db.query(User).filter(User.username == "admin").first()
        
        if not existing_user:
            logger.info("Creating default admin user...")
            # In production, use proper password hashing
            admin_user = User(
                username="admin",
                email="admin@cryptovolt.local",
                password_hash="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5YmMxSUvar.OC",  # password: admin123
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            logger.info("✓ Default admin user created (username: admin, password: admin123)")
        else:
            logger.info("✓ Admin user already exists")
        
        db.close()
        
    except Exception as e:
        logger.error(f"Error creating default admin: {str(e)}")


def drop_all_tables():
    """
    Drop all tables from the database (for development/testing only)
    WARNING: This will delete all data!
    """
    try:
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=engine)
        logger.warning("✓ All tables dropped")
    except Exception as e:
        logger.error(f"Error dropping tables: {str(e)}")


def reset_database():
    """
    Reset database: drop all tables and recreate them (for development/testing only)
    WARNING: This will delete all data!
    """
    try:
        logger.warning("Resetting database...")
        drop_all_tables()
        init_db()
        logger.warning("✓ Database reset complete")
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")


def verify_database_connection():
    """
    Verify that the database connection is working
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("✓ Database connection verified")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {str(e)}")
        return False


if __name__ == "__main__":
    # For manual database initialization
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if verify_database_connection():
        init_db()
    else:
        logger.error("Cannot initialize database - connection failed")
