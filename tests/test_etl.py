# tests/test_etl.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from etl.transform import transform_data
from etl.load import validate_data, load_to_db
from etl.extract import extract_kaggle_data, extract_live_data
import os

def test_transform_basic():
    """
    Test basic transformation functionality.
    """
    hist = pd.DataFrame([
        {'symbol': 'bitcoin', 'date': '2025-01-01', 'close': 40000},
        {'symbol': 'bitcoin', 'date': '2025-01-02', 'close': 41000},
    ])
    live = {'bitcoin': {'usd': 42000}}
    out = transform_data(hist, live)
    assert 'price_change_pct' in out.columns
    assert out.loc[out['symbol']=='bitcoin', 'current_price'].iloc[0] == 42000

def test_transform_with_gemini_data():
    """
    Test transformation with Gemini-specific data fields.
    """
    hist = pd.DataFrame([
        {'symbol': 'bitcoin', 'date': '2025-01-01', 'close': 40000},
        {'symbol': 'bitcoin', 'date': '2025-01-02', 'close': 41000},
    ])
    live = {
        'bitcoin': {
            'usd': 42000,
            'bid': 41990,
            'ask': 42010,
            'volume': {'USD': 1000000}
        }
    }
    out = transform_data(hist, live)
    assert 'price_change_pct' in out.columns
    assert 'bid' in out.columns
    assert 'ask' in out.columns
    assert 'volume_usd' in out.columns
    assert out.loc[out['symbol']=='bitcoin', 'current_price'].iloc[0] == 42000

def test_validate_data_success():
    """
    Test successful data validation.
    """
    df = pd.DataFrame([
        {'symbol': 'bitcoin', 'date': '2025-01-01', 'close': 40000, 'current_price': 42000, 'price_change_pct': 5.0},
        {'symbol': 'ethereum', 'date': '2025-01-01', 'close': 2000, 'current_price': 2100, 'price_change_pct': 5.0},
    ])
    validated_df = validate_data(df)
    assert len(validated_df) == 2

def test_validate_data_missing_columns():
    """
    Test data validation with missing required columns.
    """
    df = pd.DataFrame([
        {'symbol': 'bitcoin', 'date': '2025-01-01', 'close': 40000},
    ])
    with pytest.raises(ValueError, match="Required columns missing"):
        validate_data(df)

def test_validate_data_negative_prices():
    """
    Test data validation filters out negative prices.
    """
    df = pd.DataFrame([
        {'symbol': 'bitcoin', 'date': '2025-01-01', 'close': -40000, 'current_price': 42000, 'price_change_pct': 5.0},
        {'symbol': 'ethereum', 'date': '2025-01-01', 'close': 2000, 'current_price': 2100, 'price_change_pct': 5.0},
    ])
    validated_df = validate_data(df)
    assert len(validated_df) == 1
    assert validated_df.iloc[0]['symbol'] == 'ethereum'

@patch('etl.extract.os.path.exists')
@patch('etl.extract.pd.read_csv')
def test_extract_kaggle_data_success(mock_read_csv, mock_exists):
    """
    Test successful Kaggle data extraction.
    """
    # Mock that the file exists
    mock_exists.return_value = True
    
    mock_df = pd.DataFrame({
        'Symbol': ['bitcoin', 'ethereum'],
        'Date': ['2025-01-01', '2025-01-01'],
        'Open': [40000, 2000],
        'High': [41000, 2100],
        'Low': [39000, 1900],
        'Close': [40500, 2050],
        'Volume': [1000, 2000]
    })
    mock_read_csv.return_value = mock_df
    
    result = extract_kaggle_data('fake_path.csv')
    assert len(result) == 2
    assert 'symbol' in result.columns
    assert 'close' in result.columns

@patch('etl.extract.os.path.exists')
def test_extract_kaggle_data_file_not_found(mock_exists):
    """
    Test Kaggle data extraction with file not found.
    """
    # Mock that the file does not exist
    mock_exists.return_value = False
    
    with pytest.raises(FileNotFoundError):
        extract_kaggle_data('nonexistent_file.csv')

@patch('etl.extract._fetch_gemini_data_with_retry')
def test_extract_live_data_success(mock_fetch):
    """
    Test successful live data extraction from Gemini.
    """
    mock_fetch.return_value = {'last': '42000'}
    
    result = extract_live_data(['bitcoin'])
    assert 'bitcoin' in result
    assert result['bitcoin']['usd'] == 42000

@patch('etl.extract._fetch_gemini_data_with_retry')
def test_extract_live_data_timeout(mock_fetch):
    """
    Test live data extraction with timeout.
    """
    # Mock the retry logic to eventually fail
    mock_fetch.side_effect = Exception("Timeout")
    
    result = extract_live_data(['bitcoin'])
    # With retry logic, we might still get a result if caching is involved
    # But in this test, we expect it to be empty because all retries fail
    assert 'bitcoin' not in result or result['bitcoin']['usd'] is None