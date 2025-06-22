#!/bin/bash
set -e

# Step 1: Ensure IWSLT14 test set is present
bash prepare_iwslt14.sh

# Step 2: Download CTranslate2 model from HuggingFace Hub if not present
if [ ! -d "nllb-200-distilled-1.3B-ct2-int8" ]; then
    echo "Downloading CTranslate2 model from HuggingFace Hub..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='OpenNMT/nllb-200-distilled-1.3B-ct2-int8', local_dir='nllb-200-distilled-1.3B-ct2-int8')"
    echo "Model downloaded."
else
    echo "CTranslate2 model already exists."
fi

# Step 3: Run the main Python script
python3 translate_sentences.py
