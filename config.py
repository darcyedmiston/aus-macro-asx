from pathlib import Path

# Directories
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Analysis parameters
LOOKBACK_MONTHS = 12      # Months to use when comparing macro conditions
TOP_N_SIMILAR = 10        # How many similar periods to display

# Regime thresholds
REGIME_THRESHOLDS = {
    "recession": {"gdp": 0.0},
    "late_cycle": {"inflation": 4.0, "cash_rate": 3.5},
    "expansion": {"inflation": 3.0, "gdp": 2.0, "unemployment": 5.5},
    "recovery": {"unemployment": 6.0, "gdp": 0.0},
    "stagflation": {"inflation": 4.0, "gdp": 2.0},
}

# Similarity scoring weights (sum to 1.0)
SIMILARITY_WEIGHTS = {
    "CPI_YoY": 0.30,
    "CashRate": 0.25,
    "GDP_YoY": 0.25,
    "Unemployment": 0.20,
}

# Data sources metadata (for reference / future automation)
DATA_SOURCES = {
    "asx200": {
        "type": "yahoo",
        "ticker": "^AXJO",
        "start_date": "1992-06-01",
    },
    "cash_rate": {
        "type": "rba",
        "url": "https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx",
        "manual": "https://www.rba.gov.au/statistics/cash-rate/",
    },
    "cpi": {
        "type": "abs",
        "manual": "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation",
    },
    "unemployment": {
        "type": "abs",
        "manual": "https://www.abs.gov.au/statistics/labour/employment-and-unemployment",
    },
    "gdp": {
        "type": "abs",
        "manual": "https://www.abs.gov.au/statistics/economy/national-accounts",
    },
}

# Plot formatting
CHART_STYLE = "seaborn-v0_8-whitegrid"
COLOR_PALETTE = "husl"
FIGURE_DPI = 200

# Dashboard text
DASHBOARD_TITLE = "Australian Economy & ASX Macro Dashboard"
DASHBOARD_ICON = "🇦🇺"

# Formatting
DATE_FORMAT = "%Y-%m-%d"
DECIMAL_PLACES = {
    "percentage": 2,
    "index": 0,
    "correlation": 2,
}
