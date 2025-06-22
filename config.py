# OpenNMT v3 models to evaluate
OPENNMT_MODELS = [
    {
        "name": "OpenNMT-v3-EN-DE-Large",
        "model_url": "https://s3.amazonaws.com/opennmt-models/v3-py/ende/ende-large-withoutBT.pt",
        "local_model_path": "ende-large-withoutBT.pt",
        "bpe_url": "https://s3.amazonaws.com/opennmt-models/v3-py/ende/subwords.en_de.bpe",
        "bpe_path": "subwords.en_de.bpe",
        "ct2_model_path": "ende-large-ct2"  # CTranslate2 converted model path
    }
]

# IWSLT14 Test Set Paths
IWSLT_TEST_SRC = "data/de-en/test.en"
IWSLT_TEST_REF = "data/de-en/test.de"

# Translation settings for OpenNMT (different from NLLB language codes)
SOURCE_LANGUAGE = "en"  # English source
TARGET_LANGUAGE = "de"  # German target
