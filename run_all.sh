#!/bin/bash
set -e

# Step 1: Ensure IWSLT14 test set is present
bash prepare_iwslt14.sh

# Step 2: Run the comprehensive evaluation script for OpenNMT v3 EN-DE model
python3 evaluate_all_models.py
