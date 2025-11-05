# api/main.py
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import create_engine, text
from api.routes.crypto_routes import router as crypto_router
import os
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Simple in-memory rate limiting
class RateLimiter:
    """
    A simple in-memory rate limiter implementing a sliding window algorithm.
    
    This class tracks requests by IP address and enforces rate limits based on
    a configurable maximum number of requests within a time window.
    """
    
    def __init__(self):
        """
        Initialize the rate limiter with an empty requests dictionary.
        """
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str, max_requests: int, window_seconds: int) -> bool:
        """
        Check if a request from the given IP is allowed based on rate limits.
        
        Args:
            client_ip (str): The client's IP address
            max_requests (int): Maximum number of requests allowed in the window
            window_seconds (int): Time window in seconds
            
        Returns:
            bool: True if the request is allowed, False if rate limit exceeded
        """
        now = time.time()
        # Remove old requests outside the window
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if now - req_time < window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_ip]) < max_requests:
            self.requests[client_ip].append(now)
            return True
        return False

rate_limiter = RateLimiter()

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "default_api_key_for_testing")

def verify_api_key(x_api_key: str = Header(...)):
    """
    Verify API key from request header.
    
    This dependency function checks that the request includes a valid API key
    in the X-API-Key header. If the key is missing or invalid, it raises
    an HTTP 401 Unauthorized exception.
    
    Args:
        x_api_key (str): The API key from the X-API-Key header
        
    Returns:
        str: The verified API key
        
    Raises:
        HTTPException: If the API key is missing or invalid (401)
    """
    if x_api_key != API_KEY:
        logger.warning("Invalid or missing API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    logger.info("API key verified successfully")
    return x_api_key

def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """
    Rate limiting dependency factory.
    
    Creates a dependency function that enforces rate limits based on client IP.
    If the rate limit is exceeded, it raises an HTTP 429 Too Many Requests exception.
    
    Args:
        max_requests (int): Maximum number of requests allowed in the window (default: 10)
        window_seconds (int): Time window in seconds (default: 60)
        
    Returns:
        function: A FastAPI dependency function for rate limiting
    """
    def rate_limit_dependency(request: Request):
        """
        FastAPI dependency function that enforces rate limits.
        
        Args:
            request (Request): The incoming HTTP request
            
        Raises:
            HTTPException: If rate limit is exceeded (429)
        """
        # Get client IP, fallback to a default if not available
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip, max_requests, window_seconds):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return rate_limit_dependency

def check_database_health():
    """
    Check the health of the database connection.
    
    Returns:
        dict: Database health status with connection details
    """
    database_url = os.getenv('DATABASE_URL', 'sqlite:///crypto_data.db')
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            # Execute a simple query to test the connection
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        return {
            "status": "healthy",
            "database": "connected",
            "url": database_url.split("@")[-1] if "@" in database_url else database_url
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "url": database_url.split("@")[-1] if "@" in database_url else database_url
        }

app = FastAPI(
    title='Crypto ETL API', 
    description='Real-time cryptocurrency price aggregator API with Gemini data. Provides endpoints for fetching latest crypto prices, metrics, and symbol information.',
    version='1.0.0',
    contact={
        "name": "Crypto API Support",
        "email": "support@cryptoapi.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add trusted host middleware for security
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

@app.get("/")
def read_root():
    """
    Root endpoint that provides information about the API.
    
    This endpoint returns a welcome message and a link to the API documentation.
    It does not require authentication and is intended for initial API discovery.
    
    Returns:
        dict: A welcome message and link to the API documentation.
    """
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Real-Time Crypto Price Aggregator API", "docs": "/docs"}

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API is running.
    
    This endpoint returns the health status of the API and can be used for
    monitoring and uptime checks. It does not require authentication.
    
    Returns:
        dict: Comprehensive health status of the API with version, database, and system information.
    """
    logger.info("Health check endpoint accessed")
    
    # Check database health
    db_health = check_database_health()
    
    # Get system information
    import platform
    system_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "hostname": platform.node()
    }
    
    return {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database": db_health,
        "system": system_info
    }

# Include routes with API key authentication and rate limiting
app.include_router(crypto_router, prefix='/api/crypto', dependencies=[Depends(verify_api_key)])