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

## Testing

Run tests with:
```bash
PYTHONPATH=. pytest -v
```# Documents-Real-Time-Crypto-Price-Aggregator
