#!/bin/bash
set -e

# Step 1: Ensure IWSLT14 test set is present
bash prepare_iwslt14.sh

# Step 2: Run the main Python script
python3 translate_sentences.py
