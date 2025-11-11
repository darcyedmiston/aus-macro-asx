![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/darcyedmiston/aus-macro-asx/update_data.yml?label=Auto%20Data%20Update&logo=github)
[![Render Status](https://img.shields.io/badge/Render-Live%20Dashboard-brightgreen?logo=render)](https://aus-macro-asx.onrender.com)ASX Macro Dashboard - Ray Dalio Style Pattern Analysis
An educational macro analysis dashboard inspired by Ray Dalio's principles: "Everything happens over and over again." This tool helps understand where we are in the economic cycle by comparing current conditions to historical patterns.
🎯 What This Dashboard Does

Merges key Australian macro data (ASX200, CPI, Cash Rate, GDP, Unemployment)
Identifies macro regimes (Expansion, Late Cycle, Stagflation, Recession, Recovery)
Finds similar historical periods to today using pattern matching
Shows what typically happened next in those similar periods
Provides educational context for each chart and relationship

Important: This is a decision-support and learning tool, not a black-box trading signal.

📁 Project Structure
asx-macro-dashboard/
├── data/                          # Raw input data (Excel files)
│   ├── asx200.xlsx
│   ├── cpi.xlsx
│   ├── rba_cash_rate.xlsx
│   ├── unemployment.SA.xlsx
│   └── gdp_real.xlsx
├── output/                        # Generated analysis outputs
│   ├── macro_enhanced.csv
│   ├── similar_periods.csv
│   ├── regime_summary.csv
│   ├── lag_analysis_rate_asx.csv
│   └── *.png (charts)
├── analyze_asx_vs_macro.py       # Main analysis script
├── streamlit_app.py               # Interactive dashboard
├── fetch_and_update.py            # Automated data fetcher
├── .github/workflows/
│   └── update_data.yml            # GitHub Actions automation
└── README.md                      # This file

🚀 Quick Start
1. Install Dependencies
bashpip install pandas numpy matplotlib seaborn scipy streamlit yfinance openpyxl
2. Run Analysis (First Time)
bashpython analyze_asx_vs_macro.py
This processes your data files and generates:

output/macro_enhanced.csv - Full merged dataset
output/similar_periods.csv - Periods most like today
output/regime_summary.csv - Performance by regime
Various charts and correlation analyses

3. Launch Dashboard
bashstreamlit run streamlit_app.py
The dashboard will open in your browser at http://localhost:8501

📊 Key Features
Today vs Similar Periods

Shows the top 10 historical periods most similar to current conditions
Includes similarity score, regime classification, and subsequent 12-month returns
Provides statistical summary (mean, median, range) of historical outcomes
Educational focus: Emphasizes this is descriptive, not predictive

Macro Regime Classification

Expansion: Strong growth, moderate inflation
Late Cycle: High inflation, rising rates
Stagflation: High inflation + weak growth
Recession: Negative GDP growth
Recovery: Growth returns, unemployment lags
Mid Cycle: Balanced conditions

Interactive Analysis

Compare ASX performance against any macro indicator
Correlation heatmaps showing relationships
Regime performance summaries
Contextual explanations for each chart


🔄 Automated Updates
Option 1: Manual Updates (Current)

Download latest data from:

RBA: https://www.rba.gov.au/statistics/cash-rate/
ABS: https://www.abs.gov.au/statistics/
Yahoo Finance: Use fetch_and_update.py or manual download


Run analysis:

bash   python analyze_asx_vs_macro.py

Refresh dashboard (it will auto-reload)

Option 2: Automated Updates (GitHub Actions)

Push your code to GitHub
The workflow in .github/workflows/update_data.yml will run monthly
Updates are automatically committed back to the repo
If connected to Streamlit Cloud, dashboard auto-deploys

Setup:
bashgit add .
git commit -m "Initial setup"
git push origin main
The workflow runs on the 1st of each month or can be triggered manually from the GitHub Actions tab.
Option 3: Deploy to Streamlit Cloud

Push code to GitHub
Go to https://streamlit.io/cloud
Connect your GitHub repo
Deploy!

Streamlit Cloud will:

Auto-deploy on every push
Run your app 24/7
Handle dependencies from requirements.txt

Create requirements.txt:
txtpandas
numpy
matplotlib
seaborn
scipy
streamlit
yfinance
openpyxl

🛣️ Roadmap & Extensions
Phase 1: Core Enhancements (Next 1-3 months)

 Improve similarity scoring

Add feature scaling (z-scores)
Weight indicators by predictive power
Include momentum/volatility measures


 Add bond yields

10-year Australian government bonds
Yield curve (10Y - 2Y spread)
Real yields (nominal - inflation)


 Housing data

CoreLogic home prices
Housing credit growth
Add to regime classification



Phase 2: Global Context (3-6 months)

 US macro integration

Fed Funds Rate vs RBA Cash Rate
US CPI and unemployment
S&P 500 correlation with ASX200


 Currency analysis

AUD/USD exchange rate
Impact on commodity exporters


 Global growth proxies

China PMI (Australia's largest trading partner)
Global commodity prices



Phase 3: Advanced Analytics (6-12 months)

 Regime transition probabilities

Build Markov chain model
Show probability of moving to next regime


 Sector rotation analysis

Which ASX sectors perform in each regime?
Financials vs Materials vs Defensives


 Playbook section

"If X regime + Y conditions → Consider Z positioning"
Plain-language scenario guides


 Monte Carlo simulations

Given current conditions, simulate range of outcomes
Stress testing portfolio assumptions



Phase 4: Community & Data Quality

 Data validation dashboard

Show data freshness and quality checks
Alert on missing/suspicious values


 Backtesting framework

Test: "If we used this model historically, what would we have done?"
Learn from successes and failures


 Open data sources

Fully automated FRED, RBA, ABS APIs
No manual downloads needed




🧪 Technical Details
Similarity Algorithm
Current implementation uses Euclidean distance with normalization:
python# For each historical period, calculate distance to current conditions
distance = sqrt(mean([(hist_val - curr_val) / curr_val]^2))
similarity_score = 1 / (1 + distance)
Indicators used: CPI YoY, Cash Rate, GDP YoY, Unemployment (12-month averages)
Future: Consider Mahalanobis distance, dynamic time warping, or machine learning embeddings.
Regime Classification Logic
Simple rule-based system:

Recession: GDP < 0
Late Cycle: Inflation > 4% AND Rate > 3.5%
Expansion: Inflation < 3% AND GDP > 2% AND Unemployment < 5.5%
Recovery: Unemployment > 6% AND GDP > 0
Stagflation: Inflation > 4% AND GDP < 2%
Mid Cycle: Everything else

Future: Use clustering (K-means, HMM) to discover regimes data-driven.

🤝 Contributing
Contributions welcome! Areas where help is needed:

Data pipelines: Improve automated fetching from RBA/ABS APIs
UI/UX: Better visualizations, interactivity
Documentation: More explanations, video walkthroughs
Testing: Edge cases, data validation
Extensions: Implement roadmap features


⚠️ Disclaimer
This tool is for educational and research purposes only. It is NOT:

Investment advice
A trading system
A guarantee of future performance

Past patterns provide context but don't predict the future. Always:

Do your own research
Consult licensed professionals
Understand the risks
Diversify appropriately

The creators assume no liability for financial decisions made using this tool.

📚 Resources & Inspiration

Ray Dalio: Principles for Navigating Big Debt Crises
RBA: https://www.rba.gov.au/publications/
ABS: https://www.abs.gov.au/
Bridgewater: Economic research papers


📧 Contact & Support
Questions? Ideas? Found a bug?

GitHub Issues: Create an issue
Discussions: Share ideas in GitHub Discussions
Email: your.email@example.com


📄 License
MIT License - feel free to fork, modify, and use for your own learning!

Last Updated: 2025-11-10
Version: 1.0.0
