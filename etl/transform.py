# etl/transform.py
import pandas as pd
import numpy as np
import logging
import uuid
import time
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_correlation_id():
    """Generate a unique correlation ID for request tracing."""
    return str(uuid.uuid4())

def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality metrics for the transformed DataFrame.
    
    Args:
        df (pd.DataFrame): Transformed DataFrame
        
    Returns:
        dict: Data quality metrics
    """
    correlation_id = get_correlation_id()
    logger.info(f"[{correlation_id}] Calculating data quality metrics", extra={
        'correlation_id': correlation_id,
        'step': 'calculate_data_quality_score'
    })
    
    if df.empty:
        return {
            'completeness': 0.0,
            'consistency': 0.0,
            'freshness': 0.0,
            'overall_score': 0.0,
            'record_count': 0
        }
    
    # Completeness: Percentage of non-null values
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = df.count().sum()
    completeness = non_null_cells / total_cells if total_cells > 0 else 0
    
    # Consistency: Check for price anomalies (negative prices)
    numeric_columns = ['close', 'current_price']
    consistency_checks = []
    for col in numeric_columns:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            consistency_checks.append(1 - (negative_count / len(df)) if len(df) > 0 else 1)
    
    consistency = np.mean(consistency_checks) if consistency_checks else 1.0
    
    # Freshness: Percentage of recent data (assuming date column exists)
    freshness = 1.0  # Default to 100% fresh
    if 'date' in df.columns:
        try:
            # Convert to datetime if not already
            df_dates = pd.to_datetime(df['date'])
            # Calculate days since most recent record
            most_recent = df_dates.max()
            days_old = (pd.Timestamp.now() - most_recent).days
            # Freshness score: 100% if 0 days old, decreasing as data gets older
            freshness = max(0, 1 - (days_old / 30))  # Assume data older than 30 days is stale
        except Exception as e:
            logger.warning(f"[{correlation_id}] Error calculating freshness: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'calculate_data_quality_score',
                'warning': 'FreshnessCalculationError'
            })
    
    # Overall score
    overall_score = (completeness + consistency + freshness) / 3
    
    logger.info(f"[{correlation_id}] Data quality metrics calculated", extra={
        'correlation_id': correlation_id,
        'step': 'calculate_data_quality_score',
        'completeness': completeness,
        'consistency': consistency,
        'freshness': freshness,
        'overall_score': overall_score
    })
    
    return {
        'completeness': completeness,
        'consistency': consistency,
        'freshness': freshness,
        'overall_score': overall_score,
        'record_count': len(df)
    }

def transform_data(historical_df: pd.DataFrame, live_json: dict) -> pd.DataFrame:
    """
    Transform historical and live data into final format for database storage.
    
    Args:
        historical_df (pd.DataFrame): Historical cryptocurrency data
        live_json (dict): Live cryptocurrency data from API
        
    Returns:
        pd.DataFrame: Transformed data ready for database storage
    """
    correlation_id = get_correlation_id()
    logger.info(f"[{correlation_id}] Starting data transformation", extra={
        'correlation_id': correlation_id,
        'step': 'transform_data',
        'historical_records': len(historical_df),
        'live_records': len(live_json)
    })
    
    start_time = time.time()
    
    # ensure consistent symbol column
    if 'symbol' not in historical_df.columns:
        if 'id' in historical_df.columns:
            historical_df = historical_df.rename(columns={'id': 'symbol'})
            logger.info(f"[{correlation_id}] Renamed 'id' column to 'symbol'", extra={
                'correlation_id': correlation_id,
                'step': 'transform_data'
            })

    # Latest close per symbol
    historical_latest = (
        historical_df.sort_values(['symbol', 'date'])
        .groupby('symbol')
        .last()
        .reset_index()
    )
    logger.info(f"[{correlation_id}] Extracted latest historical data for {len(historical_latest)} symbols", extra={
        'correlation_id': correlation_id,
        'step': 'transform_data',
        'symbols_processed': len(historical_latest)
    })

    live_rows = []
    for k, v in live_json.items():
        row = {'symbol': k, 'current_price': v.get('usd')}
        # Add additional data from Gemini if available
        if 'bid' in v:
            row['bid'] = v.get('bid')
        if 'ask' in v:
            row['ask'] = v.get('ask')
        if 'volume' in v:
            # Extract USD volume if available
            volume_data = v.get('volume', {})
            if 'USD' in volume_data:
                row['volume_usd'] = volume_data.get('USD')
        live_rows.append(row)
        
    live_df = pd.DataFrame(live_rows)
    logger.info(f"[{correlation_id}] Processed {len(live_rows)} live data records", extra={
        'correlation_id': correlation_id,
        'step': 'transform_data',
        'live_records_processed': len(live_rows)
    })

    merged = historical_latest.merge(live_df, on='symbol', how='left')
    logger.info(f"[{correlation_id}] Merged data for {len(merged)} symbols", extra={
        'correlation_id': correlation_id,
        'step': 'transform_data',
        'merged_records': len(merged)
    })

    # Calculate price change percentage
    merged['price_change_pct'] = ((merged['current_price'] - merged['close']) / merged['close']) * 100

    # compute simple 10-day volatility if historical has enough rows
    if 'close' in historical_df.columns:
        try:
            vol = (
                historical_df.assign(ret=historical_df.groupby('symbol')['close'].pct_change())
                .groupby('symbol')['ret']
                .rolling(10)
                .std()
                .reset_index(level=0, drop=True)
            )
            # pick last volatility per symbol
            last_vol = (
                pd.DataFrame({'symbol': historical_df['symbol'], 'vol': vol})
                .groupby('symbol')
                .last()
                .reset_index()
            )
            merged = merged.merge(last_vol, on='symbol', how='left')
            logger.info(f"[{correlation_id}] Calculated volatility data", extra={
                'correlation_id': correlation_id,
                'step': 'transform_data'
            })
        except Exception as e:
            logger.warning(f"[{correlation_id}] Error calculating volatility: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'transform_data',
                'warning': 'VolatilityCalculationError',
                'error_details': str(e)
            })
            merged['vol'] = None
    else:
        merged['vol'] = None

    # Tidy columns
    columns_to_keep = ['symbol', 'date', 'close', 'current_price', 'price_change_pct', 'vol']
    # Add Gemini specific columns if they exist
    for col in ['bid', 'ask', 'volume_usd']:
        if col in merged.columns:
            columns_to_keep.append(col)
    
    merged = merged[columns_to_keep]
    
    # Ensure we return a DataFrame
    if not isinstance(merged, pd.DataFrame):
        merged = pd.DataFrame(merged)
    
    # Calculate data quality metrics
    quality_metrics = calculate_data_quality_score(merged)
    duration = time.time() - start_time
    
    logger.info(f"[{correlation_id}] Data transformation complete. Final dataset has {len(merged)} rows and {len(merged.columns)} columns. Quality score: {quality_metrics['overall_score']:.2f}", extra={
        'correlation_id': correlation_id,
        'step': 'transform_data',
        'final_rows': len(merged),
        'final_columns': len(merged.columns),
        'duration': duration,
        'quality_score': quality_metrics['overall_score']
    })
    
    # Add quality metrics to the result for monitoring
    merged.attrs['data_quality'] = quality_metrics
    
    return merged