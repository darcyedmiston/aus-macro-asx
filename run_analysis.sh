#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
python3 analyze_asx_vs_macro.py
