import subprocess
import tempfile
import os
from config import MODEL_PATHS
import sentencepiece as spm
import urllib.request

def download_sentencepiece_model(spm_url, spm_path):
    if not os.path.isfile(spm_path):
        print(f"Downloading SentencePiece model from {spm_url}...")
        urllib.request.urlretrieve(spm_url, spm_path)
        print("SentencePiece model downloaded.")
    else:
        print("SentencePiece model already exists.")

def preprocess_with_sentencepiece(sentences, spm_path, lang_tag=None):
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    processed = []
    for s in sentences:
        pieces = sp.encode(s, out_type=str)
        if lang_tag:
            pieces = [lang_tag] + pieces
        processed.append(' '.join(pieces))
    return processed

def translate_sentences(sentences, model_path, spm_path, lang_tag=None, gpu=0):
    # Preprocess sentences with SentencePiece
    sentences = preprocess_with_sentencepiece(sentences, spm_path, lang_tag)

    # Write sentences to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as src_file:
        for s in sentences:
            src_file.write(s.strip() + '\n')
        src_file_path = src_file.name

    # Prepare output file
    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as out_file:
        out_file_path = out_file.name

    # Build the onmt_translate command
    cmd = [
        "onmt_translate",
        "-model", model_path,
        "-src", src_file_path,
        "-output", out_file_path,
        "-replace_unk",
        "-verbose",
        "-gpu", str(gpu)
    ]

    print("Running translation...")
    subprocess.run(cmd, check=True)

    # Read and print translations
    with open(out_file_path, "r", encoding="utf-8") as f:
        translations = [line.strip() for line in f]

    # Clean up temp files
    os.remove(src_file_path)
    os.remove(out_file_path)

    return translations

if __name__ == "__main__":
    SPM_URL = "https://s3.amazonaws.com/opennmt-models/nllb-200/flores200_sacrebleu_tokenizer_spm.model"
    SPM_PATH = "flores200_sacrebleu_tokenizer_spm.model"
    download_sentencepiece_model(SPM_URL, SPM_PATH)
    # Example sentences to translate (German to English)
    sentences = [
        "Hallo, wie geht es dir?",
        "Das ist ein Test.",
        "Vielen Dank für Ihre Hilfe."
    ]
    # Use the first model in your MODEL_PATHS
    model_path = MODEL_PATHS[0]
    # For NLLB models, prepend the language tag (e.g., 'eng_Latn')
    lang_tag = "eng_Latn"
    translations = translate_sentences(sentences, model_path, SPM_PATH, lang_tag=lang_tag)
    print("\nTranslations:")
    for src, tgt in zip(sentences, translations):
        print(f"SRC: {src}\nTGT: {tgt}\n")
