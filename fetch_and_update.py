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

Note:
- Some URLs / formats for ABS & RBA may change.
- Functions are written to fail gracefully with clear logs.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ASX_TICKER = "^AXJO"  # ASX 200 index from Yahoo Finance


# ========================================
# 1. FETCH FUNCTIONS
# ========================================

def fetch_asx200_yahoo(start_date: str = "1992-01-01") -> pd.DataFrame | None:
    """
    Fetch ASX200 (^AXJO) from Yahoo Finance, resample to month-end.
    """
    logger.info("Fetching ASX200 (^AXJO) from Yahoo Finance...")
    try:
        df = yf.download(
            ASX_TICKER,
            start=start_date,
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
        )

        if df.empty:
            logger.error("No ASX200 data returned from Yahoo.")
            return None

        df = df[["Close"]].resample("M").last().reset_index()
        df.columns = ["Date", "ASX200"]
        logger.info(f"✓ Fetched {len(df)} months of ASX200 data.")
        return df

    except Exception as e:
        logger.error(f"Error fetching ASX200: {e}")
        return None


def fetch_rba_cash_rate() -> pd.DataFrame | None:
    """
    Template: Fetch RBA Cash Rate.

    Uses RBA Table F1D as an example. RBA may change URLs/format.
    If this fails, you'll see a log + can manually update data/cashrate.csv.
    """
    logger.info("Fetching RBA Cash Rate (template)...")
    url = "https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx"

    try:
        df = pd.read_excel(url)
    except Exception as e:
        logger.error(f"Error downloading RBA cash rate file: {e}")
        logger.info("📝 Manual fallback: download from https://www.rba.gov.au/statistics/cash-rate/ and save as data/cashrate.csv")
        return None

    # This part is approximate; adjust columns based on actual sheet format.
    try:
        # Try to find date & rate columns
        df = df.iloc[10:]  # often metadata in first rows
        df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "CashRate"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df[["Date", "CashRate"]].set_index("Date").resample("M").last().reset_index()
        logger.info(f"✓ Parsed {len(df)} months of Cash Rate data.")
        return df
    except Exception as e:
        logger.error(f"Error parsing RBA cash rate format: {e}")
        logger.info("📝 Please inspect the Excel and create a clean data/cashrate.csv with Date, CashRate.")
        return None


def fetch_abs_cpi() -> pd.DataFrame | None:
    """
    Template: Fetch CPI All Groups (Australia).
    In practice, you may:
      - Use ABS API, or
      - Manually export a CSV and keep it in data/cpi.csv.

    Here we try to load local file if present; otherwise log instructions.
    """
    local = DATA_DIR / "cpi.csv"
    if local.exists():
        logger.info("Loading CPI from local data/cpi.csv")
        df = pd.read_csv(local)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    logger.info("No CPI automation configured. Please export CPI to data/cpi.csv with columns: Date, CPI.")
    return None


def fetch_abs_unemployment() -> pd.DataFrame | None:
    """
    Template: Fetch Unemployment Rate (seasonally adjusted).
    Tries local file first.
    """
    local = DATA_DIR / "unemployment.csv"
    if local.exists():
        logger.info("Loading Unemployment from data/unemployment.csv")
        df = pd.read_csv(local)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    logger.info("No Unemployment automation configured. Please export ABS unemployment to data/unemployment.csv with Date, Unemployment.")
    return None


def fetch_abs_gdp() -> pd.DataFrame | None:
    """
    Template: Fetch Real GDP (chain volume).
    Tries local file first.
    """
    local = DATA_DIR / "gdp.csv"
    if local.exists():
        logger.info("Loading GDP from data/gdp.csv")
        df = pd.read_csv(local)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    logger.info("No GDP automation configured. Please export ABS GDP to data/gdp.csv with Date, GDP_real.")
    return None


# ========================================
# 2. SAVE HELPERS
# ========================================

def save_csv(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        logger.warning(f"⚠️ Skipping save for {name}: no data.")
        return
    path = DATA_DIR / name
    df.to_csv(path, index=False)
    logger.info(f"💾 Saved {name} -> {path}")


# ========================================
# 3. RUN ANALYSIS
# ========================================

def run_analysis_script():
    """
    Run analyze_asx_vs_macro.py to regenerate output/*.csv and charts.
    """
    script = BASE_DIR / "analyze_asx_vs_macro.py"
    if not script.exists():
        logger.error("analyze_asx_vs_macro.py not found. Cannot run analysis.")
        return

    logger.info("▶️ Running analyze_asx_vs_macro.py ...")
    exit_code = os.system(f"{sys.executable} {script}")
    if exit_code != 0:
        logger.error(f"Analysis script exited with code {exit_code}")
    else:
        logger.info("✅ Analysis complete. Outputs updated.")


# ========================================
# 4. MAIN
# ========================================

def main():
    logger.info("===== Starting data update pipeline =====")

    # 1. ASX200
    asx_df = fetch_asx200_yahoo()
    save_csv(asx_df, "asx200.csv")

    # 2. Cash Rate (template / partial)
    cash_df = fetch_rba_cash_rate()
    if cash_df is not None:
        save_csv(cash_df, "cashrate.csv")

    # 3. CPI, Unemployment, GDP via local/manual (for now)
    cpi_df = fetch_abs_cpi()
    if cpi_df is not None:
        save_csv(cpi_df, "cpi_clean.csv")

    unemp_df = fetch_abs_unemployment()
    if unemp_df is not None:
        save_csv(unemp_df, "unemployment_clean.csv")

    gdp_df = fetch_abs_gdp()
    if gdp_df is not None:
        save_csv(gdp_df, "gdp_clean.csv")

    # 4. Run your existing analysis pipeline
    run_analysis_script()

    logger.info("===== Data update pipeline finished =====")


if __name__ == "__main__":
    main()
