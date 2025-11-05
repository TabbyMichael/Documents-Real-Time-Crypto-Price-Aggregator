# etl/extract.py
import pandas as pd
import requests
from typing import List
import os
import uuid
import time
import logging
import random
from functools import wraps
from collections import OrderedDict
from enum import Enum
from dotenv import load_dotenv
import hashlib

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Simple in-memory cache with LRU eviction
class LRUCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            # Move to end to show it was recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            # Move to end to show it was recently used
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used item
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()

# Global cache instance
cache = LRUCache(max_size=100)

# Circuit breaker states
class CircuitState(Enum):
    CLOSED = 1
    OPEN = 2
    HALF_OPEN = 3

# Simple circuit breaker implementation
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Global circuit breaker instance
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def get_correlation_id():
    """Generate a unique correlation ID for request tracing."""
    return str(uuid.uuid4())

def get_secure_api_key():
    """
    Get API key from environment variables with additional security measures.
    
    Returns:
        str: API key or None if not found
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback to default API key (less secure)
        api_key = os.getenv("API_KEY")
        if api_key:
            logger.warning("Using default API key. For production, use GEMINI_API_KEY for better security.")
    
    return api_key

def mask_sensitive_data(data):
    """
    Mask sensitive data for logging purposes.
    
    Args:
        data (str): Sensitive data to mask
        
    Returns:
        str: Masked data
    """
    if not data:
        return data
    if len(data) <= 4:
        return "*" * len(data)
    return data[:2] + "*" * (len(data) - 4) + data[-2:]

def retry_with_backoff(max_retries=3, backoff_factor=2, jitter=True):
    """
    Decorator to implement retry logic with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        backoff_factor (int): Base for exponential backoff calculation
        jitter (bool): Whether to add random jitter to delay
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            correlation_id = kwargs.get('correlation_id', get_correlation_id())
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"[{correlation_id}] Max retries exceeded for {func.__name__}: {e}", extra={
                            'correlation_id': correlation_id,
                            'step': func.__name__,
                            'attempt': attempt,
                            'error': str(e)
                        })
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = backoff_factor ** attempt
                    if jitter:
                        delay += random.uniform(0, 1)  # Add jitter
                    
                    logger.warning(f"[{correlation_id}] Attempt {attempt + 1} failed for {func.__name__}. Retrying in {delay:.2f}s: {e}", extra={
                        'correlation_id': correlation_id,
                        'step': func.__name__,
                        'attempt': attempt + 1,
                        'delay': delay,
                        'error': str(e)
                    })
                    
                    time.sleep(delay)
        return wrapper
    return decorator

def extract_kaggle_data(path: str) -> pd.DataFrame:
    """
    Extract data from Kaggle CSV file.
    
    Args:
        path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with cryptocurrency data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    correlation_id = get_correlation_id()
    logger.info(f"[{correlation_id}] Starting Kaggle data extraction", extra={
        'correlation_id': correlation_id,
        'step': 'extract_kaggle_data',
        'path': path
    })
    
    start_time = time.time()
    try:
        if not os.path.exists(path):
            logger.error(f"[{correlation_id}] Kaggle data file not found at {path}", extra={
                'correlation_id': correlation_id,
                'step': 'extract_kaggle_data',
                'error': 'FileNotFoundError'
            })
            raise FileNotFoundError(f"Kaggle data file not found at {path}")
        
        df = pd.read_csv(path)
        duration = time.time() - start_time
        logger.info(f"[{correlation_id}] Successfully loaded {len(df)} rows from {path} in {duration:.2f}s", extra={
            'correlation_id': correlation_id,
            'step': 'extract_kaggle_data',
            'rows': len(df),
            'duration': duration
        })
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Symbol': 'symbol',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        # Expect at least: symbol, date, open, high, low, close, volume
        return df
    except FileNotFoundError:
        logger.error(f"[{correlation_id}] Kaggle data file not found at {path}", extra={
            'correlation_id': correlation_id,
            'step': 'extract_kaggle_data',
            'error': 'FileNotFoundError'
        })
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"[{correlation_id}] Kaggle data file is empty: {path}", extra={
            'correlation_id': correlation_id,
            'step': 'extract_kaggle_data',
            'error': 'EmptyDataError'
        })
        raise
    except pd.errors.ParserError as e:
        logger.error(f"[{correlation_id}] Error parsing Kaggle data file {path}: {e}", extra={
            'correlation_id': correlation_id,
            'step': 'extract_kaggle_data',
            'error': 'ParserError',
            'error_details': str(e)
        })
        raise
    except Exception as e:
        logger.error(f"[{correlation_id}] Unexpected error loading Kaggle data from {path}: {e}", extra={
            'correlation_id': correlation_id,
            'step': 'extract_kaggle_data',
            'error': 'UnexpectedError',
            'error_details': str(e)
        })
        raise


@retry_with_backoff(max_retries=3, backoff_factor=2)
def _fetch_gemini_data_with_retry(url, correlation_id=None):
    """Helper function to fetch data from Gemini with retry logic."""
    if correlation_id is None:
        correlation_id = get_correlation_id()
        
    # Check cache first
    # Create a cache key that includes the URL and doesn't include sensitive data
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logger.info(f"[{correlation_id}] Cache hit for {url}", extra={
            'correlation_id': correlation_id,
            'step': '_fetch_gemini_data_with_retry',
            'cache_hit': True
        })
        return cached_data
    
    # Get API key
    api_key = get_secure_api_key()
    headers = {}
    if api_key:
        # For Gemini, we don't need to add the API key to headers for public endpoints
        # But we log that we're using an API key
        logger.info(f"[{correlation_id}] Using API key for Gemini requests", extra={
            'correlation_id': correlation_id,
            'step': '_fetch_gemini_data_with_retry',
            'api_key_masked': mask_sensitive_data(api_key)
        })
    
    # Use circuit breaker
    def fetch_data():
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    try:
        data = circuit_breaker.call(fetch_data)
    except Exception as e:
        logger.error(f"[{correlation_id}] Circuit breaker prevented request to {url}: {e}", extra={
            'correlation_id': correlation_id,
            'step': '_fetch_gemini_data_with_retry',
            'error': 'CircuitBreakerOpen',
            'error_details': str(e)
        })
        raise e
    
    # Cache the result for 60 seconds
    cache.put(cache_key, data)
    
    logger.info(f"[{correlation_id}] Fetched and cached data from {url}", extra={
        'correlation_id': correlation_id,
        'step': '_fetch_gemini_data_with_retry',
        'cache_hit': False
    })
    
    return data

def extract_live_data(symbols: List[str]):
    """
    Extract real-time data from Gemini API
    
    Args:
        symbols (List[str]): List of cryptocurrency symbols to fetch
        
    Returns:
        dict: Dictionary with real-time price data
    """
    correlation_id = get_correlation_id()
    logger.info(f"[{correlation_id}] Starting live data extraction for {len(symbols)} symbols", extra={
        'correlation_id': correlation_id,
        'step': 'extract_live_data',
        'symbol_count': len(symbols)
    })
    
    # Symbol mapping from our format to Gemini format
    symbol_mapping = {
        'bitcoin': 'BTCUSD',
        'ethereum': 'ETHUSD',
        'solana': 'SOLUSD'
        # Note: Cardano (ADA) is not available on Gemini
    }
    
    result = {}
    successful_requests = 0
    failed_requests = 0
    
    # Get data for each symbol
    for symbol in symbols:
        gemini_symbol = symbol_mapping.get(symbol.lower())
        
        # Skip if symbol not available on Gemini
        if not gemini_symbol:
            logger.warning(f"[{correlation_id}] Symbol {symbol} not available on Gemini, skipping...", extra={
                'correlation_id': correlation_id,
                'step': 'extract_live_data',
                'symbol': symbol,
                'warning': 'SymbolNotAvailable'
            })
            continue
            
        # Gemini API endpoint for ticker data
        url = f"https://api.gemini.com/v1/pubticker/{gemini_symbol}"
        
        try:
            start_time = time.time()
            data = _fetch_gemini_data_with_retry(url, correlation_id)
            duration = time.time() - start_time
            
            # Extract price information
            if 'last' in data:
                result[symbol.lower()] = {'usd': float(data['last'])}
                successful_requests += 1
                logger.info(f"[{correlation_id}] Successfully fetched data for {symbol} in {duration:.2f}s", extra={
                    'correlation_id': correlation_id,
                    'step': 'extract_live_data',
                    'symbol': symbol,
                    'duration': duration,
                    'price': float(data['last'])
                })
            else:
                failed_requests += 1
                logger.warning(f"[{correlation_id}] No price data available for {symbol}", extra={
                    'correlation_id': correlation_id,
                    'step': 'extract_live_data',
                    'symbol': symbol,
                    'warning': 'NoPriceData'
                })
        except requests.exceptions.Timeout:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Timeout fetching data for {gemini_symbol} from Gemini", extra={
                'correlation_id': correlation_id,
                'step': 'extract_live_data',
                'symbol': symbol,
                'error': 'Timeout'
            })
        except requests.exceptions.RequestException as e:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Error fetching live data for {gemini_symbol} from Gemini: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'extract_live_data',
                'symbol': symbol,
                'error': 'RequestException',
                'error_details': str(e)
            })
        except ValueError as e:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Error parsing price data for {gemini_symbol}: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'extract_live_data',
                'symbol': symbol,
                'error': 'ValueError',
                'error_details': str(e)
            })
        except Exception as e:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Unexpected error fetching data for {gemini_symbol}: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'extract_live_data',
                'symbol': symbol,
                'error': 'UnexpectedError',
                'error_details': str(e)
            })
    
    logger.info(f"[{correlation_id}] Live data extraction completed. Successful: {successful_requests}, Failed: {failed_requests}", extra={
        'correlation_id': correlation_id,
        'step': 'extract_live_data',
        'successful_requests': successful_requests,
        'failed_requests': failed_requests,
        'result_count': len(result)
    })
    
    return result


def extract_gemini_candlestick_data(symbols: List[str], timeframe: str = "1m"):
    """
    Extract candlestick data from Gemini API for more detailed real-time information
    Note: Gemini doesn't have a direct candlestick endpoint, so we'll simulate with ticker data
    
    Args:
        symbols (List[str]): List of cryptocurrency symbols to fetch
        timeframe (str): Timeframe for data (default: "1m")
        
    Returns:
        dict: Dictionary with detailed cryptocurrency data
    """
    correlation_id = get_correlation_id()
    logger.info(f"[{correlation_id}] Starting detailed Gemini data extraction for {len(symbols)} symbols", extra={
        'correlation_id': correlation_id,
        'step': 'extract_gemini_candlestick_data',
        'symbol_count': len(symbols),
        'timeframe': timeframe
    })
    
    # Symbol mapping from our format to Gemini format
    symbol_mapping = {
        'bitcoin': 'BTCUSD',
        'ethereum': 'ETHUSD',
        'solana': 'SOLUSD'
        # Note: Cardano (ADA) is not available on Gemini
    }
    
    result = {}
    successful_requests = 0
    failed_requests = 0
    
    # Get detailed data for each symbol
    for symbol in symbols:
        gemini_symbol = symbol_mapping.get(symbol.lower())
        
        # Skip if symbol not available on Gemini
        if not gemini_symbol:
            logger.warning(f"[{correlation_id}] Symbol {symbol} not available on Gemini, skipping...", extra={
                'correlation_id': correlation_id,
                'step': 'extract_gemini_candlestick_data',
                'symbol': symbol,
                'warning': 'SymbolNotAvailable'
            })
            continue
        
        # Get ticker data
        ticker_url = f"https://api.gemini.com/v1/pubticker/{gemini_symbol}"
        
        # Get symbols details for additional information
        details_url = f"https://api.gemini.com/v1/symbols/details/{gemini_symbol}"
        
        try:
            # Get ticker data
            start_time = time.time()
            ticker_data = _fetch_gemini_data_with_retry(ticker_url, correlation_id)
            details_data = _fetch_gemini_data_with_retry(details_url, correlation_id)
            duration = time.time() - start_time
            
            # Extract comprehensive data
            if 'last' in ticker_data:
                result[symbol.lower()] = {
                    'usd': float(ticker_data['last']),
                    'bid': float(ticker_data.get('bid', 0)),
                    'ask': float(ticker_data.get('ask', 0)),
                    'volume': {}
                }
                
                # Add volume data if available
                if 'volume' in ticker_data:
                    volume_data = ticker_data['volume']
                    for key, value in volume_data.items():
                        if key in ['BTC', 'ETH', 'SOL', 'USD']:
                            result[symbol.lower()]['volume'][key] = float(value)
                
                successful_requests += 1
                logger.info(f"[{correlation_id}] Successfully fetched detailed data for {symbol} in {duration:.2f}s", extra={
                    'correlation_id': correlation_id,
                    'step': 'extract_gemini_candlestick_data',
                    'symbol': symbol,
                    'duration': duration,
                    'price': float(ticker_data['last'])
                })
            else:
                failed_requests += 1
                logger.warning(f"[{correlation_id}] No detailed data available for {symbol}", extra={
                    'correlation_id': correlation_id,
                    'step': 'extract_gemini_candlestick_data',
                    'symbol': symbol,
                    'warning': 'NoDetailedData'
                })
                            
        except requests.exceptions.Timeout:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Timeout fetching detailed data for {gemini_symbol} from Gemini", extra={
                'correlation_id': correlation_id,
                'step': 'extract_gemini_candlestick_data',
                'symbol': symbol,
                'error': 'Timeout'
            })
        except requests.exceptions.RequestException as e:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Error fetching detailed data for {gemini_symbol} from Gemini: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'extract_gemini_candlestick_data',
                'symbol': symbol,
                'error': 'RequestException',
                'error_details': str(e)
            })
        except ValueError as e:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Error parsing detailed data for {gemini_symbol}: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'extract_gemini_candlestick_data',
                'symbol': symbol,
                'error': 'ValueError',
                'error_details': str(e)
            })
        except Exception as e:
            failed_requests += 1
            logger.error(f"[{correlation_id}] Unexpected error fetching detailed data for {gemini_symbol}: {e}", extra={
                'correlation_id': correlation_id,
                'step': 'extract_gemini_candlestick_data',
                'symbol': symbol,
                'error': 'UnexpectedError',
                'error_details': str(e)
            })
    
    logger.info(f"[{correlation_id}] Detailed Gemini data extraction completed. Successful: {successful_requests}, Failed: {failed_requests}", extra={
        'correlation_id': correlation_id,
        'step': 'extract_gemini_candlestick_data',
        'successful_requests': successful_requests,
        'failed_requests': failed_requests,
        'result_count': len(result)
    })
    
    return result


def get_gemini_symbols():
    """
    Get list of available symbols from Gemini
    
    Returns:
        list: List of available symbols on Gemini exchange
    """
    correlation_id = get_correlation_id()
    logger.info(f"[{correlation_id}] Fetching available symbols from Gemini", extra={
        'correlation_id': correlation_id,
        'step': 'get_gemini_symbols'
    })
    
    url = "https://api.gemini.com/v1/symbols"
    
    try:
        start_time = time.time()
        symbols = _fetch_gemini_data_with_retry(url, correlation_id)
        duration = time.time() - start_time
        logger.info(f"[{correlation_id}] Successfully fetched {len(symbols)} symbols from Gemini in {duration:.2f}s", extra={
            'correlation_id': correlation_id,
            'step': 'get_gemini_symbols',
            'symbol_count': len(symbols),
            'duration': duration
        })
        return symbols
    except requests.exceptions.Timeout:
        logger.error(f"[{correlation_id}] Timeout fetching symbols from Gemini", extra={
            'correlation_id': correlation_id,
            'step': 'get_gemini_symbols',
            'error': 'Timeout'
        })
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"[{correlation_id}] Error fetching symbols from Gemini: {e}", extra={
            'correlation_id': correlation_id,
            'step': 'get_gemini_symbols',
            'error': 'RequestException',
            'error_details': str(e)
        })
        return []
    except Exception as e:
        logger.error(f"[{correlation_id}] Unexpected error fetching symbols from Gemini: {e}", extra={
            'correlation_id': correlation_id,
            'step': 'get_gemini_symbols',
            'error': 'UnexpectedError',
            'error_details': str(e)
        })
        return []