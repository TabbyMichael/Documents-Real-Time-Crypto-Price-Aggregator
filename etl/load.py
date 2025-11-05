# etl/load.py
import os
from sqlalchemy import create_engine
import pandas as pd
from typing import Optional
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean data before database insertion.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        pd.DataFrame: Validated and cleaned DataFrame
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger.info("Starting data validation")
    
    # Check if required columns exist
    required_columns = ['symbol', 'date', 'close', 'current_price', 'price_change_pct']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from DataFrame: {missing_columns}")
    
    # Store original row count
    original_rows = len(df)
    logger.info(f"Validating {original_rows} rows of data")
    
    # Validate data types
    # Ensure symbol is string
    df['symbol'] = df['symbol'].astype(str)
    
    # Ensure numeric columns are numeric
    numeric_columns = ['close', 'current_price', 'price_change_pct', 'vol']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing required values
    df = df.dropna(subset=['symbol', 'date', 'close', 'current_price', 'price_change_pct'])
    
    # Validate price values are positive or zero
    df = df.loc[df['close'] >= 0]
    df = df.loc[df['current_price'] >= 0]
    
    # Limit string lengths
    df['symbol'] = df['symbol'].str.slice(0, 50)
    
    # Remove duplicates based on symbol and date
    df = df.drop_duplicates(subset=['symbol', 'date'], keep='first')
    
    # Log validation results
    validated_rows = len(df)
    removed_rows = original_rows - validated_rows
    if removed_rows > 0:
        logger.warning(f"Removed {removed_rows} invalid rows during validation")
    
    logger.info(f"Data validation complete. {validated_rows} rows passed validation")
    return df

def load_to_db(df: pd.DataFrame, database_url: Optional[str] = None):
    """
    Load validated data to database.
    
    Args:
        df (pd.DataFrame): DataFrame to load
        database_url (str, optional): Database URL. If not provided, will use DATABASE_URL from environment or default to SQLite.
    """
    # Get database URL from environment if not provided
    if not database_url:
        database_url = os.getenv('DATABASE_URL', 'sqlite:///crypto_data.db')
    
    # Validate that we have a database URL
    if not database_url:
        raise ValueError("No database URL provided and DATABASE_URL not set in environment")
        
    logger.info(f"Connecting to database: {database_url}")
    engine = create_engine(database_url, pool_pre_ping=True)

    # Validate data before insertion
    try:
        df = validate_data(df)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data validation: {e}")
        raise

    # Create table if not exists and upsert latest values
    try:
        with engine.begin() as conn:
            # for simplicity replace table (you can change to upsert)
            df.to_sql('crypto_prices', conn, if_exists='replace', index=False)
        logger.info(f"Successfully loaded {len(df)} rows to database")
    except Exception as e:
        logger.error(f"Error loading data to database: {e}")
        raise