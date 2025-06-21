# config.py
# Edit these variables to match your file locations

import os

# List of model download URLs (fill these with your actual links)
MODEL_URLS = [
    "https://example.com/model1.pt",
    "https://example.com/model2.pt",
    "https://example.com/model3.pt",
    "https://example.com/model4.pt"
]

# Dynamically assign model paths based on URLs
MODEL_PATHS = [os.path.join("models", os.path.basename(url)) for url in MODEL_URLS]

# Path to OpenNMT-py translate.py script
TRANSLATE_PY = "/path/to/OpenNMT-py/translate.py"

# Paths to IWSLT15 test set source and reference files
SRC_FILE = "/path/to/IWSLT15.test.de"  # German source
REF_FILE = "/path/to/IWSLT15.test.en"  # English reference

# Output files for each model (auto-generated)
OUTPUT_FILES = [f"output_model{i+1}.txt" for i in range(4)]
