# api/routes/crypto_routes.py
from fastapi import APIRouter, HTTPException, Path, Query
from sqlalchemy import create_engine, text
from pydantic import BaseModel, validator
import pandas as pd
import os
from dotenv import load_dotenv
import re

load_dotenv()

router = APIRouter()
database_url = os.getenv('DATABASE_URL', 'sqlite:///crypto_data.db')
engine = create_engine(database_url)

class SymbolRequest(BaseModel):
    symbol: str
    
    @validator('symbol')
    def validate_symbol(cls, v):
        # Validate that symbol contains only alphanumeric characters and underscores
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Symbol must contain only alphanumeric characters and underscores')
        # Limit symbol length
        if len(v) > 50:
            raise ValueError('Symbol must be less than 50 characters')
        # Convert to lowercase
        return v.lower()

@router.get('/latest')
def get_latest(
    limit: int = Query(50, ge=1, le=1000, description="Number of records to return (1-1000)")
):
    """
    Get the latest cryptocurrency prices from the database.
    
    Args:
        limit (int): Number of records to return (1-1000), default is 50
        
    Returns:
        list: A list of dictionaries containing the latest cryptocurrency price data.
        
    Raises:
        HTTPException: If there's an error retrieving data from the database.
    """
    try:
        # Use parameterized query to prevent SQL injection
        query = text("SELECT * FROM crypto_prices ORDER BY date DESC LIMIT :limit")
        df = pd.read_sql(query, con=engine, params={"limit": limit})
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve latest cryptocurrency data. Please try again later.")

@router.get('/metrics')
def get_metrics():
    """
    Get aggregated metrics for all cryptocurrencies.
    
    Returns:
        list: A list of dictionaries containing average price change and volatility for each symbol.
        
    Raises:
        HTTPException: If there's an error calculating metrics.
    """
    try:
        # Use parameterized query to prevent SQL injection
        query = text("SELECT symbol, AVG(price_change_pct) as avg_change, AVG(vol) as avg_vol FROM crypto_prices GROUP BY symbol")
        df = pd.read_sql(query, con=engine)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to calculate cryptocurrency metrics. Please try again later.")

@router.get('/symbols')
def get_symbols():
    """
    Get list of available cryptocurrency symbols.
    
    Returns:
        list: A list of available cryptocurrency symbols.
        
    Raises:
        HTTPException: If there's an error retrieving symbols.
    """
    try:
        # Use parameterized query to prevent SQL injection
        query = text("SELECT DISTINCT symbol FROM crypto_prices")
        df = pd.read_sql(query, con=engine)
        return df['symbol'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve cryptocurrency symbols. Please try again later.")

@router.get('/symbol/{symbol}')
def get_symbol_data(
    symbol: str = Path(..., title="Cryptocurrency symbol", description="The cryptocurrency symbol to retrieve data for", example="bitcoin", 
                      regex=r'^[a-zA-Z0-9_]+$'),
    limit: int = Query(1, ge=1, le=100, description="Number of records to return (1-100)")
):
    """
    Get detailed data for a specific cryptocurrency symbol.
    
    Args:
        symbol (str): The cryptocurrency symbol to retrieve data for.
        limit (int): Number of records to return (1-100), default is 1
        
    Returns:
        dict or list: A dictionary containing detailed data for the specified symbol, or a list if limit > 1.
        
    Raises:
        HTTPException: If the symbol is not found or there's an error retrieving data.
    """
    # Additional validation
    if len(symbol) > 50:
        raise HTTPException(status_code=400, detail="Symbol must be less than 50 characters")
    
    try:
        # Use parameterized query to prevent SQL injection
        if limit == 1:
            query = text("SELECT * FROM crypto_prices WHERE symbol = :symbol ORDER BY date DESC LIMIT 1")
            df = pd.read_sql(query, con=engine, params={"symbol": symbol.lower()})
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Cryptocurrency symbol '{symbol}' not found.")
            return df.to_dict(orient='records')[0]
        else:
            query = text("SELECT * FROM crypto_prices WHERE symbol = :symbol ORDER BY date DESC LIMIT :limit")
            df = pd.read_sql(query, con=engine, params={"symbol": symbol.lower(), "limit": limit})
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Cryptocurrency symbol '{symbol}' not found.")
            return df.to_dict(orient='records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data for '{symbol}'. Please try again later.")

@router.get('/gemini/symbols')
def get_gemini_symbols():
    """
    Get list of available symbols from Gemini API.
    
    Returns:
        list: A list of available symbols on the Gemini exchange.
        
    Raises:
        HTTPException: If there's an error fetching symbols from Gemini API.
    """
    try:
        import requests
        url = "https://api.gemini.com/v1/symbols"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch available symbols from Gemini API. Please try again later.")