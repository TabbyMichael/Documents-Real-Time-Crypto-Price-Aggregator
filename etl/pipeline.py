# etl/pipeline.py
import os
import logging
from dotenv import load_dotenv
from etl.extract import extract_kaggle_data, extract_live_data, extract_gemini_candlestick_data
from etl.transform import transform_data
from etl.load import load_to_db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def run_pipeline():
    """
    Run the complete ETL pipeline.
    
    This function orchestrates the extraction, transformation, and loading of cryptocurrency data.
    """
    logger.info("🚀 Starting ETL pipeline with Gemini API")
    
    # Define kaggle_path early to avoid linter issues
    kaggle_path = os.getenv('KAGGLE_DATA_PATH', 'data/raw/kaggle_crypto_prices.csv')
    
    try:
        logger.info(f"Extracting Kaggle data from {kaggle_path}")
        kaggle_df = extract_kaggle_data(kaggle_path)
        logger.info(f"Successfully extracted {len(kaggle_df)} rows of historical data")
    except FileNotFoundError:
        logger.error(f"Kaggle data file not found at {kaggle_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting Kaggle data: {e}")
        raise

    try:
        symbols = os.getenv('COIN_IDS', 'bitcoin,ethereum').split(',')
        logger.info(f"Extracting live data for symbols: {symbols}")
        live_json = extract_live_data(symbols)
        logger.info(f"Successfully extracted live data for {len(live_json)} symbols")
    except Exception as e:
        logger.error(f"Error extracting live data: {e}")
        raise

    try:
        # Also get detailed data from Gemini
        logger.info("Extracting detailed Gemini data")
        detailed_data = extract_gemini_candlestick_data(symbols)
        logger.info(f"Successfully extracted detailed data for {len(detailed_data)} symbols")
        
        # Merge detailed data with live data
        for symbol in detailed_data:
            if symbol in live_json:
                live_json[symbol].update(detailed_data[symbol])
            else:
                live_json[symbol] = detailed_data[symbol]
        logger.info("Merged detailed data with live data")
    except Exception as e:
        logger.error(f"Error extracting or merging detailed data: {e}")
        raise

    try:
        logger.info("Transforming data")
        logger.info(f"Live data: {live_json}")
        df = transform_data(kaggle_df, live_json)
        logger.info(f"Successfully transformed data into {len(df)} rows")
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        raise

    try:
        logger.info("Loading data to database")
        load_to_db(df, os.getenv('DATABASE_URL'))
        logger.info("✅ ETL finished successfully")
    except Exception as e:
        logger.error(f"Error loading data to database: {e}")
        raise

if __name__ == '__main__':
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        exit(1)