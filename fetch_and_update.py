"""
fetch_and_update.py
Automated data fetching and analysis pipeline for the ASX Macro Dashboard.

This script:
1. Fetches latest data from:
   - Yahoo Finance (ASX200)
   - RBA (cash rate) [template]
   - ABS (CPI, Unemployment, GDP) [templates]
2. Saves cleaned data into the data/ folder
3. Runs analyze_asx_vs_macro.py to regenerate output files

Run this monthly (or on-demand) to keep the dashboard current.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
from io import StringIO
import yfinance as yf
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# DATA FETCHING FUNCTIONS
# ========================================

def fetch_asx200_yahoo(start_date='1992-06-01'):
    """Fetch ASX200 data from Yahoo Finance."""
    logger.info("Fetching ASX200 data from Yahoo Finance...")
    try:
        ticker = yf.Ticker("^AXJO")
        df = ticker.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
        if df.empty:
            logger.error("No ASX200 data retrieved from Yahoo Finance")
            return None

        df = df[['Close']].resample('M').last().reset_index()
        df.columns = ['Date', 'ASX200']
        logger.info(f"✓ Fetched {len(df)} months of ASX200 data")
        return df
    except Exception as e:
        logger.error(f"Error fetching ASX200 data: {e}")
        return None


def fetch_rba_cash_rate():
    """Fetch RBA Cash Rate from RBA website."""
    logger.info("Fetching RBA Cash Rate data...")
    try:
        url = "https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx"
        df = pd.read_excel(url, sheet_name=0, skiprows=10)
        df.columns = ['Date', 'CashRate']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date').resample('M').last().reset_index()
        logger.info(f"✓ Fetched {len(df)} months of Cash Rate data")
        return df
    except Exception as e:
        logger.error(f"Error fetching RBA Cash Rate: {e}")
        logger.info("📝 Manual fallback: Download from https://www.rba.gov.au/statistics/cash-rate/")
        return None


# ========================================
# MAIN EXECUTION
# ========================================

def run_analysis_script():
    """Run the main analysis script after data updates."""
    try:
        logger.info("Running analysis script...")
        os.system("python analyze_asx_vs_macro.py")
        logger.info("✅ Analysis complete. Outputs updated.")
    except Exception as e:
        logger.error(f"Error running analysis script: {e}")


def main():
    logger.info("===== Starting data update pipeline =====")

    # Fetch ASX200
    asx200_df = fetch_asx200_yahoo()
    if asx200_df is not None:
        asx200_df.to_csv(DATA_DIR / "asx200.csv", index=False)
        logger.info("Saved ASX200 data to data/asx200.csv")

    # Fetch Cash Rate
    cash_df = fetch_rba_cash_rate()
    if cash_df is not None:
        cash_df.to_csv(DATA_DIR / "cash_rate.csv", index=False)
        logger.info("Saved Cash Rate data to data/cash_rate.csv")

    # Run analysis
    run_analysis_script()

    # Forward-fill missing or zero macro values
    try:
        macro_path = OUTPUT_DIR / "macro_enhanced.csv"
        if macro_path.exists():
            macro_df = pd.read_csv(macro_path)

            # Forward-fill unemployment and cash rate
            for col in ["Unemployment", "CashRate"]:
                if col in macro_df.columns:
                    macro_df[col] = macro_df[col].replace(0, np.nan).ffill()

            macro_df.to_csv(macro_path, index=False)
            logger.info("Filled missing Unemployment and CashRate values in macro_enhanced.csv")
        else:
            logger.warning("macro_enhanced.csv not found; skipping fill step.")
    except Exception as e:
        logger.warning(f"Could not apply fill logic: {e}")

    logger.info("===== Data update pipeline finished =====")


if __name__ == "__main__":
    main()
