#!/usr/bin/env python
"""
Database Management Script
Usage: python manage_db.py [command]

Commands:
  init       - Initialize database (create tables)
  reset      - Reset database (drop and recreate - DANGEROUS)
  verify     - Verify database connection
  stats      - Show database statistics
  seed       - Seed database with sample data
  clean      - Clean database of old records
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.init_db import (
    init_db, 
    verify_database_connection, 
    reset_database,
    drop_all_tables,
    create_default_admin
)
from app.core.database import SessionLocal
from app.models.database import User, TradingStrategy, MarketData, SentimentData, Alert
from app.services.database_service import DatabaseStats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_init():
    """Initialize database"""
    logger.info("=" * 60)
    logger.info("INITIALIZING DATABASE")
    logger.info("=" * 60)
    if init_db():
        logger.info("✓ Database initialization successful!")
        return 0
    else:
        logger.error("✗ Database initialization failed!")
        return 1


def cmd_verify():
    """Verify database connection"""
    logger.info("=" * 60)
    logger.info("VERIFYING DATABASE CONNECTION")
    logger.info("=" * 60)
    if verify_database_connection():
        logger.info("✓ Database connection successful!")
        return 0
    else:
        logger.error("✗ Database connection failed!")
        return 1


def cmd_reset():
    """Reset database (DROP ALL DATA)"""
    logger.info("=" * 60)
    logger.warning("WARNING: This will DELETE ALL DATA from the database!")
    logger.info("=" * 60)
    response = input("Are you sure? Type 'yes' to continue: ")
    
    if response.lower() == 'yes':
        reset_database()
        logger.warning("✓ Database reset complete!")
        return 0
    else:
        logger.info("✗ Reset cancelled")
        return 1


def cmd_stats():
    """Show database statistics"""
    logger.info("=" * 60)
    logger.info("DATABASE STATISTICS")
    logger.info("=" * 60)
    
    try:
        db = SessionLocal()
        
        # Define all models
        models = [User, TradingStrategy, MarketData, SentimentData, Alert]
        
        # Get statistics
        stats = DatabaseStats.get_all_table_stats(db, models)
        
        logger.info(f"\n{'Table':<25} {'Record Count':>15}")
        logger.info("-" * 42)
        
        total_records = 0
        for table_name, count in stats.items():
            logger.info(f"{table_name:<25} {count:>15}")
            total_records += count
        
        logger.info("-" * 42)
        logger.info(f"{'TOTAL':<25} {total_records:>15}\n")
        
        # Show recent records
        logger.info("\nRecent Users:")
        recent_users = db.query(User).order_by(User.created_at.desc()).limit(5).all()
        for user in recent_users:
            logger.info(f"  - {user.username} ({user.email}) - {user.created_at}")
        
        db.close()
        return 0
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return 1


def cmd_clean():
    """Clean old records from database"""
    logger.info("=" * 60)
    logger.info("CLEANING OLD DATABASE RECORDS")
    logger.info("=" * 60)
    
    try:
        db = SessionLocal()
        
        # Delete old sentiment data (older than 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        old_sentiment_count = db.query(SentimentData).filter(
            SentimentData.created_at < cutoff_date
        ).delete()
        
        # Delete old market data (older than 90 days)
        cutoff_date_market = datetime.utcnow() - timedelta(days=90)
        old_market_count = db.query(MarketData).filter(
            MarketData.created_at < cutoff_date_market
        ).delete()
        
        db.commit()
        logger.info(f"✓ Deleted {old_sentiment_count} old sentiment records")
        logger.info(f"✓ Deleted {old_market_count} old market records")
        
        db.close()
        return 0
        
    except Exception as e:
        logger.error(f"Error cleaning database: {str(e)}")
        return 1


def cmd_seed():
    """Seed database with sample data"""
    logger.info("=" * 60)
    logger.info("SEEDING DATABASE WITH SAMPLE DATA")
    logger.info("=" * 60)
    
    try:
        create_default_admin()
        logger.info("✓ Sample data seeded successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error seeding database: {str(e)}")
        return 1


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    
    command = sys.argv[1].lower()
    
    commands = {
        'init': cmd_init,
        'verify': cmd_verify,
        'reset': cmd_reset,
        'stats': cmd_stats,
        'clean': cmd_clean,
        'seed': cmd_seed,
    }
    
    if command not in commands:
        logger.error(f"Unknown command: {command}")
        print(__doc__)
        return 1
    
    try:
        return commands[command]()
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
