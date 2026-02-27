"""
Database initialization script
Creates all tables in the PostgreSQL database
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import Base, engine
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def init_database():
    """Initialize database with all tables"""
    print("=" * 60)
    print("CryptoVolt Database Initialization")
    print("=" * 60)
    
    print(f"\nDatabase URL: {settings.DATABASE_URL}")
    print("Creating tables...")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("\n✅ All tables created successfully!")
        print("\nTables created:")
        print("-" * 60)
        
        # List created tables
        inspector = None
        try:
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            if tables:
                for table_name in tables:
                    columns = inspector.get_columns(table_name)
                    print(f"\n📋 Table: {table_name}")
                    for col in columns:
                        col_type = str(col['type'])
                        nullable = "NULL" if col['nullable'] else "NOT NULL"
                        print(f"   - {col['name']}: {col_type} ({nullable})")
            else:
                print("No tables found. You may need to define models in your application.")
        except Exception as e:
            print(f"Note: Could not list tables: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Database initialization complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error initializing database:")
        print(f"   {type(e).__name__}: {e}")
        print("\n" + "=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    init_database()
