#!/bin/bash

echo "Setting up ASX Macro Dashboard project structure..."

mkdir -p data
mkdir -p output
mkdir -p .streamlit
mkdir -p .github/workflows

touch data/.gitkeep
touch output/.gitkeep

cat > data/README.md << EOF
# Data Directory

If not using automation, place your data files here:

- asx200.xlsx
- cpi.xlsx
- rba_cash_rate.xlsx
- unemployment.SA.xlsx
- gdp_real.xlsx
EOF

echo "✓ Directory structure created"
echo "Next:"
echo "1. pip install -r requirements.txt"
echo "2. Run: python analyze_asx_vs_macro.py"
echo "3. Run: streamlit run streamlit_app.py"
