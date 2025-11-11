import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Australian Economy & ASX Macro Dashboard",
    layout="wide"
)

# ===============================
# HEADER
# ===============================
st.title("🇦🇺 Australian Economy & ASX Macro Dashboard")
st.caption(
    "Built using Ray Dalio's principles – analysing cycles, rates, inflation, "
    "and market patterns to identify where we are in the economic machine."
)

# ===============================
# GLOSSARY
# ===============================
with st.expander("📘 Glossary & How to Read This Dashboard"):
    st.markdown("""
    ### Core Indicators
    - **ASX200** – Benchmark index of the 200 largest ASX companies.
    - **CPI (Consumer Price Index)** – Measures inflation; higher CPI = faster price growth.
    - **Cash Rate** – The RBA's policy rate; influences borrowing costs and valuations.
    - **Unemployment** – % of the labour force without a job.
    - **Real GDP** – Economic output adjusted for inflation.

    ### Derived Metrics
    - **YoY** – Year-over-year % change (12-month growth).
    - **MoM** – Month-over-month % change (short-term momentum).
    - **Rate_Change_MoM** – Monthly change in the cash rate.
    - **Unemp_Change_YoY** – Annual change in unemployment rate.

    ### Macro Regimes
    - **Expansion** – Strong growth, low unemployment, moderate inflation
    - **Late Cycle** – High inflation, rising rates, slowing growth
    - **Stagflation** – High inflation with weak growth
    - **Recession** – Negative GDP growth
    - **Recovery** – Rising unemployment but positive growth
    - **Mid Cycle** – Balanced conditions between regimes

    ### Analytical Outputs
    - **Correlation Matrix** – How closely two indicators move together.
    - **Regime Summary** – Average ASX performance in each macro phase.
    - **Lag Analysis** – How rate/inflation changes affect ASX after a delay.
    - **Similar Periods** – Past periods that looked most like today.
    """)

# ===============================
# LOAD DATA WITH ERROR HANDLING
# ===============================
@st.cache_data
def load_main_data():
    """Load the main macro enhanced dataset"""
    try:
        df = pd.read_csv("output/macro_enhanced.csv", parse_dates=["Date"])
        return df, None
    except FileNotFoundError:
        return None, "⚠️ Could not find 'output/macro_enhanced.csv'. Please run the analysis script first."
    except Exception as e:
        return None, f"⚠️ Error loading macro data: {str(e)}"

@st.cache_data
def load_similar_periods():
    """Load similar periods analysis"""
    try:
        df = pd.read_csv("output/similar_periods.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df, None
    except FileNotFoundError:
        return None, "⚠️ Similar periods file not found. Please run the analysis script."
    except Exception as e:
        return None, f"⚠️ Error loading similar periods: {str(e)}"

@st.cache_data
def load_regime_summary():
    """Load regime performance summary"""
    try:
        df = pd.read_csv("output/regime_summary.csv", index_col=0)
        return df, None
    except FileNotFoundError:
        return None, "⚠️ Regime summary not found. Please run the analysis script."
    except Exception as e:
        return None, f"⚠️ Error loading regime summary: {str(e)}"

# Load all data
df, error = load_main_data()
if error:
    st.error(error)
    st.stop()

similar_periods, similar_error = load_similar_periods()
regime_summary, regime_error = load_regime_summary()

# ===============================
# TODAY'S SNAPSHOT
# ===============================
st.header("📊 Today's Macro Snapshot")
st.caption("Current state of key economic indicators – your starting point for understanding the cycle.")

latest = df.iloc[-1]
cols = st.columns(5)
cols[0].metric("ASX200", f"{latest['ASX200']:.0f}")
cols[1].metric("CPI YoY", f"{latest['CPI_YoY']:.2f}%")
cols[2].metric("Cash Rate", f"{latest['CashRate']:.2f}%")
cols[3].metric("Unemployment", f"{latest['Unemployment']:.2f}%")

# Add regime classification for today
if 'Regime' in df.columns:
    current_regime = df.iloc[-1]['Regime'] if pd.notna(df.iloc[-1]['Regime']) else 'Unknown'
    cols[4].metric("Current Regime", current_regime)

st.divider()

# ===============================
# TODAY VS SIMILAR PERIODS (NEW SECTION)
# ===============================
st.header("🔍 Today vs Similar Historical Periods")
st.caption(
    "Pattern matching: Which past periods had similar macro conditions to today? "
    "This helps understand potential outcomes based on historical precedent."
)

if similar_error:
    st.warning(similar_error)
else:
    # Display similar periods table
    display_cols = ['date', 'similarity_score', 'regime', 'CPI_YoY', 'CashRate', 
                    'GDP_YoY', 'Unemployment', 'next_12m_return']
    
    similar_display = similar_periods[display_cols].copy()
    similar_display['date'] = similar_display['date'].dt.strftime('%Y-%m')
    similar_display['similarity_score'] = (similar_display['similarity_score'] * 100).round(1)
    similar_display['next_12m_return'] = similar_display['next_12m_return'].round(2)
    
    # Rename columns for better presentation
    similar_display.columns = ['Period', 'Similarity (%)', 'Regime', 'CPI YoY', 
                                'Cash Rate', 'GDP YoY', 'Unemployment', 'Next 12m ASX Return (%)']
    
    st.dataframe(similar_display, use_container_width=True, hide_index=True)
    
    # Generate natural language summary
    st.subheader("📈 Historical Context")
    
    valid_returns = similar_periods['next_12m_return'].dropna()
    if len(valid_returns) > 0:
        avg_return = valid_returns.mean()
        min_return = valid_returns.min()
        max_return = valid_returns.max()
        median_return = valid_returns.median()
        
        # Count positive vs negative outcomes
        positive_count = (valid_returns > 0).sum()
        total_count = len(valid_returns)
        
        st.markdown(f"""
        **Based on the {total_count} most similar historical periods:**
        
        - **Average next 12-month ASX return:** {avg_return:.2f}%
        - **Median return:** {median_return:.2f}%
        - **Range:** {min_return:.2f}% to {max_return:.2f}%
        - **Positive outcomes:** {positive_count} out of {total_count} periods ({positive_count/total_count*100:.0f}%)
        
        ⚠️ **Important:** This is *descriptive analysis*, not prediction. Past patterns inform context 
        but don't guarantee future outcomes. Use this to understand risk/reward distributions and prepare 
        for different scenarios.
        """)
        
        # Show regime distribution
        regime_counts = similar_periods['regime'].value_counts()
        st.markdown(f"**Regime distribution in similar periods:** {', '.join([f'{r} ({c})' for r, c in regime_counts.items()])}")
    else:
        st.info("Insufficient data for historical return analysis.")

st.divider()

# ===============================
# INDICATOR COMPARISON
# ===============================
st.header("📉 ASX vs Macro Indicators Over Time")
st.caption(
    "How have markets responded to changes in inflation, rates, unemployment, and growth? "
    "Look for patterns: do markets lead or lag the economy?"
)

indicator = st.selectbox(
    "Select indicator to compare with ASX:",
    ["CPI", "CashRate", "Unemployment", "GDP_real"]
)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df["Date"], df["ASX200"], label="ASX200", color="tab:blue", linewidth=2)
ax2 = ax1.twinx()
ax2.plot(df["Date"], df[indicator], label=indicator, color="tab:orange", linewidth=2)

ax1.set_xlabel("Date", fontsize=11)
ax1.set_ylabel("ASX200 Index", color="tab:blue", fontsize=11)
ax2.set_ylabel(indicator, color="tab:orange", fontsize=11)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.grid(alpha=0.3)
plt.title(f"{indicator} vs ASX200", fontsize=13, fontweight='bold')

st.pyplot(fig)

# Add contextual interpretation
if indicator == "CashRate":
    st.caption("""
    **💡 How to interpret:** In Dalio's framework, cash rates (the 'price of money') drive discount rates 
    for equities. Typically: rising rates compress valuations (especially growth stocks), while falling 
    rates support higher multiples. But timing matters—markets often anticipate rate changes 6-12 months ahead.
    """)
elif indicator == "CPI":
    st.caption("""
    **💡 How to interpret:** Moderate inflation (2-3%) is healthy. High inflation erodes real returns and 
    forces central banks to tighten. Low/negative inflation signals weak demand. Watch the *rate of change* 
    in CPI—accelerating inflation is often worse for stocks than stable high inflation.
    """)
elif indicator == "Unemployment":
    st.caption("""
    **💡 How to interpret:** Rising unemployment typically signals economic weakness, reducing consumer spending 
    and corporate profits. However, markets often bottom *before* unemployment peaks (as they're forward-looking). 
    Watch for the rate of change—accelerating job losses indicate worsening conditions.
    """)
else:
    st.caption("""
    **💡 How to interpret:** GDP measures economic output. Strong GDP growth supports corporate earnings, 
    but markets often anticipate changes in growth 6-9 months in advance. Watch for inflection points—
    when GDP growth accelerates or decelerates—as these often mark regime shifts.
    """)

st.divider()

# ===============================
# CORRELATION MATRIX
# ===============================
st.header("🔗 Correlation Matrix: Key Indicators")
st.caption(
    "Which indicators move together? Positive correlations mean they move in the same direction, "
    "negative means opposite. Helps identify leading/lagging relationships."
)

numeric_cols = ["ASX_YoY", "CPI_YoY", "CashRate", "GDP_YoY", "Unemployment"]

try:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, 
                fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix: Key Indicators", fontsize=13, fontweight='bold', pad=15)
    st.pyplot(fig)
    
    st.caption("""
    **💡 How to interpret:** Strong correlations (>0.5 or <-0.5) suggest reliable relationships. 
    For example, if CashRate and ASX_YoY have a negative correlation, it means rising rates 
    historically coincide with lower equity returns. But correlation ≠ causation—use these as 
    patterns to investigate, not certainties.
    """)
except Exception as e:
    st.warning(f"Correlation matrix could not be generated: {e}")

st.divider()

# ===============================
# REGIME PERFORMANCE
# ===============================
st.header("🎯 Economic Regime Performance Summary")
st.caption(
    "The macro 'machine' moves through phases. Each regime has typical characteristics. "
    "Knowing the regime helps set expectations and prepare portfolios."
)

if regime_error:
    st.warning(regime_error)
else:
    # Format the regime summary for better display
    regime_display = regime_summary.copy()
    
    # Round values for cleaner display
    for col in regime_display.columns:
        if regime_display[col].dtype in ['float64', 'float32']:
            regime_display[col] = regime_display[col].round(2)
    
    st.dataframe(regime_display, use_container_width=True)
    
    st.caption("""
    **💡 How to interpret:** Each regime has a typical ASX return profile:
    - **Expansion:** Best time for equities—growth is strong, rates moderate
    - **Late Cycle:** Returns moderate as rates rise to combat inflation
    - **Stagflation:** Challenging—high inflation + weak growth = poor stock performance
    - **Recession:** Volatile—markets fall first but often recover before GDP turns
    - **Recovery:** Mixed—growth returns but unemployment lags (watch for inflection)
    
    Use this table to calibrate expectations: if we're in Late Cycle, double-digit returns 
    are less likely than in Expansion. Risk management matters more.
    """)

st.divider()

# ===============================
# FOOTER & DATA INFO
# ===============================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Data Coverage")
    st.write(f"**Start Date:** {df['Date'].min().strftime('%Y-%m-%d')}")
    st.write(f"**End Date:** {df['Date'].max().strftime('%Y-%m-%d')}")
    st.write(f"**Total Observations:** {len(df)}")

with col2:
    st.markdown("### ℹ️ About This Dashboard")
    st.write("""
    This dashboard implements Ray Dalio's template framework: understanding 
    where we are in the cycle by studying historical patterns. It's educational, 
    not predictive—use it to build intuition and scenario-plan.
    """)

st.caption("💡 **Next Steps:** Re-run `analyze_asx_vs_macro.py` monthly to refresh data, then reload this dashboard.")
