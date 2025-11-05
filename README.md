# Crypto ETL API

## Project Overview

This is a Real-Time Crypto Price Aggregator that implements a complete ETL (Extract, Transform, Load) pipeline for cryptocurrency data. The system extracts historical data from CSV files and real-time data from the Gemini API, transforms the data to calculate metrics like price changes and volatility, loads the processed data into a database, and serves it through a REST API.

## Architecture

- **Data Sources**: CSV files (historical data) + Gemini API (real-time data)
- **ETL Pipeline**: Python with pandas and requests
- **Data Storage**: SQLite database (local development) / PostgreSQL (production)
- **API**: FastAPI REST API with security features
- **Scheduling**: Prefect for workflow orchestration
- **Containerization**: Docker

## Quickstart

1. Copy `.env.example` to `.env` and customize.
2. Place Kaggle CSV into `data/raw/`.
3. Install dependencies and run:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run ETL pipeline
PYTHONPATH=. python etl/pipeline.py

# Start API server
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints:
- `http://localhost:8000/` - API root
- `http://localhost:8000/health` - Health check endpoint
- `http://localhost:8000/api/crypto/latest` - Latest crypto prices with bid/ask and volume data
- `http://localhost:8000/api/crypto/metrics` - Aggregated metrics
- `http://localhost:8000/api/crypto/symbols` - List of available cryptocurrency symbols
- `http://localhost:8000/api/crypto/symbol/{symbol}` - Detailed data for a specific symbol
- `http://localhost:8000/api/crypto/gemini/symbols` - List of available symbols on Gemini exchange

## Security Features

The API includes several security features:

1. **API Key Authentication**: All endpoints require a valid API key in the `X-API-Key` header
2. **Rate Limiting**: Prevents API abuse with per-endpoint rate limits
3. **SQL Injection Prevention**: All database queries use parameterized statements
4. **Input Validation**: All inputs are validated and sanitized
5. **Data Validation**: Data is validated before database insertion
6. **Trusted Host Middleware**: Prevents HTTP Host header attacks

To use the API, include your API key in the `X-API-Key` header:
```bash
curl -H "X-API-Key: your_secure_api_key_here" http://localhost:8000/api/crypto/latest
```

## Enhanced Features

### Advanced ETL Pipeline
- **Retry Logic**: Exponential backoff for API calls
- **Circuit Breaker**: Protection against cascading failures
- **Caching**: In-memory LRU cache for API responses
- **Structured Logging**: Correlation IDs for end-to-end tracing
- **Data Quality Metrics**: Completeness, consistency, and freshness scoring

### Monitoring & Observability
- **Health Check Endpoint**: Comprehensive system status information
- **Detailed Logging**: Structured logs with timing and error information
- **Performance Metrics**: Request timing and success rates

### Security Enhancements
- **Secrets Management**: Secure API key handling
- **Data Masking**: Sensitive data protection in logs
- **Input Sanitization**: Protection against malicious inputs

## Development notes

* ETL runs once in `etl-worker` container; change command to run on a cron or use Prefect to schedule.
* Your DB url inside containers uses service name `postgres`.
* For local development, SQLite is used instead of PostgreSQL.
* The system currently supports Bitcoin, Ethereum, and Solana (Cardano is not available on Gemini).

## Project Structure

```
├── api/                 # FastAPI application
│   ├── main.py          # API entry point
│   └── routes/          # API routes
├── data/                # Data files
│   └── raw/             # Raw CSV data
├── etl/                 # ETL pipeline
│   ├── extract.py       # Data extraction
│   ├── transform.py     # Data transformation
│   ├── load.py          # Data loading
│   └── pipeline.py      # ETL pipeline orchestration
├── scheduler/           # Prefect workflow
├── tests/               # Unit tests
├── venv/                # Virtual environment
├── .env                 # Environment variables
├── .env.example         # Example environment variables
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile.api       # API Dockerfile
├── Dockerfile.etl       # ETL Dockerfile
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Deployment Options

### Local Development
```bash
# Using Docker Compose (recommended for local development)
docker-compose up -d

# Or run directly with Python
source venv/bin/activate
PYTHONPATH=. python etl/pipeline.py
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Production Deployment

#### Heroku (Simplest)
```bash
heroku create your-crypto-app
heroku addons:create heroku-postgresql:hobby-dev
git push heroku main
heroku config:set API_KEY=your_production_api_key_here
```

#### Docker Deployment
```bash
# Build and run with Docker
docker build -t crypto-etl-api -f Dockerfile.api .
docker run -p 8000:8000 crypto-etl-api
```

#### Cloud Platforms (AWS, GCP, Azure)
1. Containerize your application (already done with Dockerfiles)
2. Push to container registry (ECR, GCR, ACR)
3. Deploy to container service (ECS, GKE, AKS)

#### Kubernetes Deployment
```bash
kubectl apply -f k8s-deployment.yaml
```

## Environment Variables

Key environment variables to configure:

```bash
# Database configuration
POSTGRES_USER=crypto_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=crypto_db
DATABASE_URL=postgresql://crypto_user:your_secure_password@your-db-host:5432/crypto_db

# API Security
API_KEY=your_production_api_key
GEMINI_API_KEY=your_gemini_api_key  # Optional, for production use

# Coins to track
COIN_IDS=bitcoin,ethereum,solana

# Other
PREFECT_API_URL=http://prefect:4200
KAGGLE_DATA_PATH=data/raw/kaggle_crypto_prices.csv
```

## Testing

Run tests with:
```bash
PYTHONPATH=. pytest -v
```

## Monitoring and Maintenance

### Health Checks
- `/health` endpoint provides system status
- Database connectivity verification
- System platform information

### Performance Monitoring
- Structured logging with timing information
- Request duration tracking
- Error rate monitoring

### Data Quality Monitoring
- Completeness metrics (percentage of non-null values)
- Consistency checks (negative price detection)
- Freshness scoring (data recency)