# config.py
# Configuration for multiple NLLB models evaluation

import os

# List of NLLB CTranslate2 models to evaluate (pre-converted CTranslate2 models)
NLLB_MODELS = [
    {
        "name": "NLLB-200-3.3B",
        "repo_id": "entai2965/nllb-200-3.3B-ctranslate2",
        "local_dir": "nllb-200-3.3B-ctranslate2"
    },
    {
        "name": "NLLB-200-1.3B-distilled",
        "repo_id": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
        "local_dir": "nllb-200-distilled-1.3B-ctranslate2"
    },
    {
        "name": "NLLB-200-600M-distilled",
        "repo_id": "entai2965/nllb-200-distilled-600M-ctranslate2",
        "local_dir": "nllb-200-distilled-600M-ctranslate2"
    }
]

# SentencePiece model
SPM_URL = "https://s3.amazonaws.com/opennmt-models/nllb-200/flores200_sacrebleu_tokenizer_spm.model"
SPM_PATH = "flores200_sacrebleu_tokenizer_spm.model"

# IWSLT14 Test Set Paths
IWSLT_TEST_SRC = "data/de-en/test.de"
IWSLT_TEST_REF = "data/de-en/test.en"

# Translation settings
SOURCE_LANGUAGE = "deu_Latn"
TARGET_LANGUAGE = "eng_Latn"
