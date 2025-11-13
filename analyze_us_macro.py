"""
analyze_us_macro.py
Basic US macro analysis engine

- Reads:  data/us_macro_data.xlsx  (from fetch_us_data.py)
- Computes: YoY changes, spreads, simple regime classification
- Writes:  output/us_macro_enhanced.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# US REGIME CLASSIFICATION
# ===============================

def identify_us_regime(row):
    """
    Simple US regime classification using thresholds we discussed.
    Uses GDP_YoY, CPI_YoY, Unemployment, Fed_Funds_Rate, Real_Yield, Yield_Spread.
    """

    # Required basics
    needed = ["CPI_YoY", "GDP_YoY", "Unemployment", "Fed_Funds_Rate"]
    if any(pd.isna(row.get(col)) for col in needed):
        return "Unknown"

    inflation = row["CPI_YoY"]
    gdp = row["GDP_YoY"]
    unemp = row["Unemployment"]
    rate = row["Fed_Funds_Rate"]

    real_yield = row.get("Real_Yield", np.nan)
    spread = row.get("Yield_Spread", np.nan)

    # US thresholds from config_us idea
    # Recession
    if gdp < 0:
        return "Recession"

    # Late Cycle: high inflation + high rates (+ optional flat/inverted curve)
    if inflation > 3.5 and rate > 4.5:
        if not pd.isna(spread) and spread < 0.5:
            return "Late Cycle (Curve Flat)"
        return "Late Cycle"

    # Expansion: good growth, low inflation, low unemployment
    if inflation < 2.5 and gdp > 2.0 and unemp < 5.0:
        if not pd.isna(real_yield) and real_yield > 0:
            return "Expansion (Healthy)"
        return "Expansion"

    # Recovery: high-ish unemployment but GDP positive
    if unemp > 7.0 and gdp > 0:
        return "Recovery"

    # Stagflation: high inflation + weak-ish growth
    if inflation > 4.0 and gdp < 1.5:
        return "Stagflation"

    # Everything else
    return "Mid Cycle"


# ===============================
# MAIN ANALYSIS
# ===============================

def main():
    print("=" * 70)
    print("US Macro Analysis – S&P 500 vs US Macro (Stage 1)")
    print("=" * 70)

    # 1. Load data from Excel
    us_path = DATA_DIR / "us_macro_data.xlsx"
    if not us_path.exists():
        raise FileNotFoundError(f"Could not find {us_path}. Run fetch_us_data.py first.")

    df = pd.read_excel(us_path)

    # 2. Basic cleaning
    # Ensure Date is datetime, set as index
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Forward-fill some macro series (they often come quarterly / lagged)
    for col in ["10Y_Treasury", "Fed_Funds_Rate", "CPI", "Unemployment", "GDP_Real"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
            df[col] = df[col].ffill()

    # 3. Derived indicators
    # YoY changes (in %)
    df["SP500_YoY"] = df["SP500"].pct_change(12) * 100
    df["CPI_YoY"] = df["CPI"].pct_change(12) * 100
    df["GDP_YoY"] = df["GDP_Real"].pct_change(12) * 100

    # Unemployment YoY change (in percentage points)
    df["Unemp_Change_YoY"] = df["Unemployment"].diff(12)

    # Monthly change in Fed Funds (in percentage points)
    df["Rate_Change_MoM"] = df["Fed_Funds_Rate"].diff()

    # Bond / rate relationships
    if "10Y_Treasury" in df.columns:
        df["Yield_Spread"] = df["10Y_Treasury"] - df["Fed_Funds_Rate"]
        df["Real_Yield"] = df["10Y_Treasury"] - df["CPI_YoY"]

    # Market drawdown (like AU version)
    df["SP500_Peak"] = df["SP500"].expanding().max()
    df["Drawdown"] = (df["SP500"] / df["SP500_Peak"] - 1.0) * 100

    # 4. Regime classification
    df["Regime"] = df.apply(identify_us_regime, axis=1)

    # 5. Save to CSV
    out_path = OUT_DIR / "us_macro_enhanced.csv"
    df_reset = df.reset_index()  # put Date back as a column
    df_reset.to_csv(out_path, index=False)

    # 6. Basic summary
    print(f"\nSaved extended US dataset to: {out_path}")
    print(f"Rows: {len(df_reset)}  From: {df_reset['Date'].min().date()}  To: {df_reset['Date'].max().date()}")
    print("\nLatest row:")
    print(df_reset.tail(1).T)

    print("=" * 70)
    print("US macro analysis done (Stage 1).")
    print("=" * 70)


if __name__ == "__main__":
    main()
