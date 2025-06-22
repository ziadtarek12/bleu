# config.py
# Edit these variables to match your file locations

import os

# List of model download URLs (fill these with your actual links)
MODEL_URLS = [
    "https://s3.amazonaws.com/opennmt-models/v3-py/ende/ende-large-withoutBT.pt"
]

# Dynamically assign model paths based on URLs
MODEL_PATHS = [os.path.join("models", os.path.basename(url)) for url in MODEL_URLS]

# Path to OpenNMT-py translate.py script
TRANSLATE_PY = "/path/to/OpenNMT-py/translate.py"

# Paths to IWSLT15 test set source and reference files
SRC_FILE = "data/de-en/test.de"  # German source
REF_FILE = "data/de-en/test.en"  # English reference

# Output files for each model (auto-generated)
OUTPUT_FILES = [f"output_model{i}.txt" for i in range(len(MODEL_URLS))]
