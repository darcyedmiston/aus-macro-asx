# enhanced_macro_analyzer.py
# Enhanced version with pattern matching, regime detection, and lag analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, zscore
from datetime import datetime
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== Helper Functions ==========

def _find_col(cols, *keywords):
    """Return first column whose name contains any of keywords."""
    cols_lower = {c.lower(): c for c in cols}
    for c_lower, orig in cols_lower.items():
        for kw in keywords:
            if kw in c_lower:
                return orig
    return None

def read_asx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    date_col = _find_col(df.columns, "date")
    price_col = _find_col(df.columns, "close", "price", "last", "asx")
    if not date_col or not price_col:
        raise ValueError("ASX file must have Date and Close/Price column.")
    df = df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "ASX200"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.resample("M").last()
    return df

def read_cpi(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    date_col = _find_col(df.columns, "date", "quarter")
    cpi_col = _find_col(df.columns, "cpi") or df.select_dtypes("number").columns[0]
    if not date_col or not cpi_col:
        raise ValueError("CPI file must have a date/quarter column and a CPI value column.")
    df = df[[date_col, cpi_col]].rename(columns={date_col: "Date", cpi_col: "CPI"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.resample("M").ffill()
    return df

def read_rba_cash_rate(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    date_col = _find_col(df.columns, "date")
    rate_col = _find_col(df.columns, "cash", "target", "rate")
    if not date_col or not rate_col:
        raise ValueError("RBA cash-rate file must have Date and Rate column.")
    df = df[[date_col, rate_col]].rename(columns={date_col: "Date", rate_col: "CashRate"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.resample("M").ffill()
    return df

def read_unemployment(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    date_col = _find_col(df.columns, "date")
    u_col = _find_col(df.columns, "unemploy")
    if not date_col or not u_col:
        raise ValueError("Unemployment file must have Date and unemployment column.")
    df = df[[date_col, u_col]].rename(columns={date_col: "Date", u_col: "Unemployment"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.resample("M").ffill()
    return df

def read_gdp(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    date_col = _find_col(df.columns, "date", "quarter")
    gdp_col = _find_col(df.columns, "gdp") or df.select_dtypes("number").columns[0]
    if not date_col or not gdp_col:
        raise ValueError("GDP file must have Date/Quarter and GDP value.")
    df = df[[date_col, gdp_col]].rename(columns={date_col: "Date", gdp_col: "GDP_real"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.resample("M").ffill()
    return df

# ========== NEW: Enhanced Analysis Functions ==========

def add_derived_indicators(df):
    """Add rates of change and derived metrics"""
    # Year-over-year changes
    df['ASX_YoY'] = df['ASX200'].pct_change(12) * 100
    df['CPI_YoY'] = df['CPI'].pct_change(12) * 100  # Inflation rate
    df['GDP_YoY'] = df['GDP_real'].pct_change(12) * 100
    
    # Changes in unemployment and rates
    df['Unemp_Change_YoY'] = df['Unemployment'].diff(12)
    df['Rate_Change_MoM'] = df['CashRate'].diff()
    df['Rate_Change_YoY'] = df['CashRate'].diff(12)
    
    # Market drawdown from peak
    df['ASX_Peak'] = df['ASX200'].expanding().max()
    df['Drawdown'] = (df['ASX200'] / df['ASX_Peak'] - 1) * 100
    
    # Recession indicator (simplified: 2 consecutive quarters negative GDP)
    df['GDP_Rolling_6m'] = df['GDP_YoY'].rolling(6).mean()
    df['Recession'] = (df['GDP_Rolling_6m'] < 0).astype(int)
    
    # Bear market indicator (drawdown > 20%)
    df['Bear_Market'] = (df['Drawdown'] < -20).astype(int)
    
    return df

def analyze_lags(df, var1, var2, max_lag=24):
    """Analyze correlation between two variables at different lags"""
    results = []
    
    for lag in range(0, max_lag + 1, 3):
        try:
            # Shift var1 by lag months
            series1 = df[var1].shift(lag).dropna()
            series2 = df[var2].dropna()
            
            # Align the series
            aligned = pd.concat([series1, series2], axis=1).dropna()
            
            if len(aligned) > 30:  # Need sufficient data
                corr, pval = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                results.append({
                    'lag_months': lag,
                    'correlation': corr,
                    'p_value': pval
                })
        except:
            pass
    
    return pd.DataFrame(results)

def identify_regime(row):
    """Classify economic regime based on current conditions"""
    # Handle missing values
    if pd.isna(row['CPI_YoY']) or pd.isna(row['GDP_YoY']) or pd.isna(row['Unemployment']):
        return 'Unknown'
    
    inflation = row['CPI_YoY']
    gdp = row['GDP_YoY']
    unemp = row['Unemployment']
    rate = row['CashRate'] if not pd.isna(row['CashRate']) else 0
    
    # Regime logic
    if gdp < 0:
        return 'Recession'
    elif inflation > 4 and rate > 3.5:
        return 'Late Cycle'
    elif inflation < 3 and gdp > 2 and unemp < 5.5:
        return 'Expansion'
    elif unemp > 6 and gdp > 0:
        return 'Recovery'
    elif inflation > 4 and gdp < 2:
        return 'Stagflation'
    else:
        return 'Mid Cycle'

def find_similar_periods(df, reference_date=None, lookback=12, top_n=10):
    """Find historical periods most similar to a given reference period"""
    
    if reference_date is None:
        reference_date = df.index[-1]
    
    # Get reference period averages
    ref_idx = df.index.get_loc(reference_date)
    if ref_idx < lookback:
        print(f"Not enough history before {reference_date}")
        return None
    
    ref_window = df.iloc[ref_idx-lookback:ref_idx]
    ref_avg = ref_window[['CPI_YoY', 'CashRate', 'GDP_YoY', 'Unemployment']].mean()
    
    # Calculate similarity for all historical periods
    similarities = []
    
    for i in range(lookback, len(df) - 12):  # Leave 12 months for forward returns
        hist_window = df.iloc[i-lookback:i]
        hist_avg = hist_window[['CPI_YoY', 'CashRate', 'GDP_YoY', 'Unemployment']].mean()
        
        # Skip if any values are NaN
        if hist_avg.isna().any() or ref_avg.isna().any():
            continue
        
        # Calculate normalized Euclidean distance
        distances = []
        for col in ['CPI_YoY', 'CashRate', 'GDP_YoY', 'Unemployment']:
            if ref_avg[col] != 0:
                norm_dist = ((hist_avg[col] - ref_avg[col]) / abs(ref_avg[col])) ** 2
                distances.append(norm_dist)
        
        distance = np.sqrt(np.mean(distances))
        similarity = 1 / (1 + distance)
        
        # Calculate forward returns
        future_date = df.index[i + 12] if i + 12 < len(df) else df.index[-1]
        forward_return = (df['ASX200'].iloc[i + 12] / df['ASX200'].iloc[i] - 1) * 100 if i + 12 < len(df) else np.nan
        
        similarities.append({
            'date': df.index[i],
            'similarity_score': similarity,
            'distance': distance,
            'next_12m_return': forward_return,
            'CPI_YoY': hist_avg['CPI_YoY'],
            'CashRate': hist_avg['CashRate'],
            'GDP_YoY': hist_avg['GDP_YoY'],
            'Unemployment': hist_avg['Unemployment'],
            'regime': identify_regime(hist_avg)
        })
    
    results = pd.DataFrame(similarities).sort_values('similarity_score', ascending=False)
    return results.head(top_n)

def create_regime_summary(df):
    """Summarize market performance by regime"""
    df['Regime'] = df.apply(identify_regime, axis=1)
    
    regime_stats = df.groupby('Regime').agg({
        'ASX_YoY': ['mean', 'std', 'min', 'max'],
        'Drawdown': 'min',
        'CPI_YoY': 'mean',
        'GDP_YoY': 'mean',
        'Unemployment': 'mean'
    }).round(2)
    
    return regime_stats

# ========== Visualization Functions ==========

def plot_regime_timeline(df, output_path):
    """Plot timeline with regime shading"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot ASX
    ax.plot(df.index, df['ASX200'], linewidth=2, label='ASX 200', color='navy')
    
    # Shade recession periods
    recession_periods = df[df['Recession'] == 1]
    if len(recession_periods) > 0:
        for idx in recession_periods.index:
            ax.axvspan(idx, idx + pd.DateOffset(months=1), alpha=0.3, color='red', label='Recession' if idx == recession_periods.index[0] else '')
    
    ax.set_ylabel('ASX 200 Level', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('ASX 200 with Recession Periods', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_correlation_heatmap(df, output_path):
    """Plot correlation matrix of key indicators"""
    cols = ['ASX_YoY', 'CPI_YoY', 'CashRate', 'GDP_YoY', 'Unemployment']
    corr_matrix = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix: Key Indicators', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_lag_analysis(lag_results, var1, var2, output_path):
    """Plot correlation vs lag"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(lag_results['lag_months'], lag_results['correlation'], marker='o', linewidth=2, markersize=8)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (months)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(f'Correlation: {var1} vs {var2} at Different Lags', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Highlight max correlation
    max_idx = lag_results['correlation'].abs().idxmax()
    max_lag = lag_results.loc[max_idx, 'lag_months']
    max_corr = lag_results.loc[max_idx, 'correlation']
    ax.axvline(max_lag, color='green', linestyle=':', alpha=0.7)
    ax.text(max_lag, max_corr, f'  Max: {max_corr:.2f} at {max_lag}m', fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_current_vs_history(df, current_date, output_path):
    """Plot current conditions vs historical percentiles"""
    current = df.loc[current_date]
    
    indicators = ['CPI_YoY', 'CashRate', 'GDP_YoY', 'Unemployment']
    percentiles = []
    
    for ind in indicators:
        pct = (df[ind] < current[ind]).sum() / df[ind].notna().sum() * 100
        percentiles.append(pct)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red' if p > 75 or p < 25 else 'orange' if p > 60 or p < 40 else 'green' for p in percentiles]
    bars = ax.barh(indicators, percentiles, color=colors, alpha=0.7)
    
    ax.axvline(50, color='black', linestyle='--', alpha=0.5, label='Median')
    ax.axvline(25, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(75, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Percentile (%)', fontsize=12)
    ax.set_title(f'Current Conditions vs History (as of {current_date.strftime("%Y-%m")})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend()
    
    # Add value labels
    for i, (ind, pct) in enumerate(zip(indicators, percentiles)):
        val = current[ind]
        ax.text(pct + 2, i, f'{val:.1f} ({pct:.0f}th)', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

# ========== Main ==========

def main():
    print("=" * 60)
    print("Enhanced ASX Macro Analysis with Pattern Matching")
    print("=" * 60)
    
    # Read data
    asx_file = DATA_DIR / "asx200.xlsx"
    cpi_file = DATA_DIR / "cpi.xlsx"
    rba_file = DATA_DIR / "rba_cash_rate.xlsx"
    unemp_file = DATA_DIR / "unemployment.SA.xlsx"
    gdp_file = DATA_DIR / "gdp_real.xlsx"
    
    print("\n📊 Loading data...")
    asx = read_asx(asx_file)
    cpi = read_cpi(cpi_file)
    rba = read_rba_cash_rate(rba_file)
    unemp = read_unemployment(unemp_file)
    gdp = read_gdp(gdp_file)
    
    # Merge
    merged = asx.join([cpi, rba, unemp, gdp], how='outer').sort_index().dropna(how='all')
    
    # Add derived indicators
    print("🔧 Calculating derived indicators...")
    merged = add_derived_indicators(merged)
    
    # Save enhanced dataset
    merged.to_csv(OUT_DIR / "macro_enhanced.csv", index_label="Date")
    print(f"✅ Saved enhanced dataset: {OUT_DIR / 'macro_enhanced.csv'}")
    
    # Regime analysis
    print("\n📈 Analyzing regimes...")
    regime_summary = create_regime_summary(merged)
    regime_summary.to_csv(OUT_DIR / "regime_summary.csv")
    print("\nRegime Performance Summary:")
    print(regime_summary)
    
    # Lag analysis
    print("\n🕐 Analyzing lags...")
    lag_rate_asx = analyze_lags(merged, 'CashRate', 'ASX_YoY', max_lag=24)
    lag_rate_asx.to_csv(OUT_DIR / "lag_analysis_rate_asx.csv", index=False)
    print("\nCash Rate vs ASX Returns (lag analysis):")
    print(lag_rate_asx)
    
    # Find similar periods to current
    current_date = merged.index[-1]
    print(f"\n🔍 Finding periods similar to {current_date.strftime('%Y-%m')}...")
    similar = find_similar_periods(merged, reference_date=current_date, lookback=12, top_n=10)
    if similar is not None:
        similar.to_csv(OUT_DIR / "similar_periods.csv", index=False)
        print("\nTop 10 Most Similar Historical Periods:")
        print(similar[['date', 'similarity_score', 'next_12m_return', 'regime']].to_string())
    
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    plot_regime_timeline(merged, OUT_DIR / "asx_with_recessions.png")
    plot_correlation_heatmap(merged, OUT_DIR / "correlation_heatmap.png")
    plot_lag_analysis(lag_rate_asx, 'CashRate', 'ASX_YoY', OUT_DIR / "lag_cashrate_asx.png")
    plot_current_vs_history(merged, current_date, OUT_DIR / "current_vs_history.png")
    
    print("\n✅ All outputs saved to 'output/' folder:")
    print("   - macro_enhanced.csv (full dataset with derived indicators)")
    print("   - regime_summary.csv (performance by regime)")
    print("   - lag_analysis_rate_asx.csv (correlation at different lags)")
    print("   - similar_periods.csv (historical periods most like today)")
    print("   - asx_with_recessions.png")
    print("   - correlation_heatmap.png")
    print("   - lag_cashrate_asx.png")
    print("   - current_vs_history.png")
    
    print("\n" + "=" * 60)
    print("Analysis complete! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()
