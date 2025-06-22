import subprocess
import sys
import importlib.util
import os
import urllib.request
import sentencepiece as spm
from config import MODEL_PATHS, SRC_FILE, REF_FILE, OUTPUT_FILES
import matplotlib.pyplot as plt

# Helper to check and install a package if missing
def ensure_package(package_name):
    if importlib.util.find_spec(package_name) is None:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Ensure required packages are installed
ensure_package("sacrebleu")
ensure_package("OpenNMT-py")
from sacrebleu import corpus_bleu


def download_sentencepiece_model(spm_url, spm_path):
    if not os.path.isfile(spm_path):
        print(f"Downloading SentencePiece model from {spm_url}...")
        urllib.request.urlretrieve(spm_url, spm_path)
        print("SentencePiece model downloaded.")
    else:
        print("SentencePiece model already exists.")


def preprocess_with_sentencepiece_file(input_path, output_path, spm_path, lang_tag=None):
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            pieces = sp.encode(line.strip(), out_type=str)
            if lang_tag:
                pieces = [lang_tag] + pieces
            fout.write(' '.join(pieces) + '\n')


def translate_with_model(model_path, src_file, output_file, gpu=0):
    cmd = [
        "onmt_translate",
        "-model", model_path,
        "-src", src_file,
        "-output", output_file,
        "-replace_unk",
        "-verbose",
        "-gpu", str(gpu)
    ]
    print(f"Translating with model: {model_path}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("onmt_translate not found. Please ensure OpenNMT-py is installed and in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error during translation with model {model_path}: {e}")
        sys.exit(1)


def compute_bleu(hypothesis_file, reference_file):
    with open(reference_file, "r", encoding="utf-8") as ref_f:
        references = [line.strip() for line in ref_f]
    with open(hypothesis_file, "r", encoding="utf-8") as hyp_f:
        hypotheses = [line.strip() for line in hyp_f]
    bleu = corpus_bleu(hypotheses, [references])
    return bleu.score


def main():
    # Download SentencePiece model
    SPM_URL = "https://s3.amazonaws.com/opennmt-models/nllb-200/flores200_sacrebleu_tokenizer_spm.model"
    SPM_PATH = "flores200_sacrebleu_tokenizer_spm.model"
    download_sentencepiece_model(SPM_URL, SPM_PATH)

    # Preprocess the source file with SentencePiece and language tag
    preprocessed_src = "preprocessed_test.de"
    lang_tag = "eng_Latn"  # Set the target language tag for NLLB models
    preprocess_with_sentencepiece_file(SRC_FILE, preprocessed_src, SPM_PATH, lang_tag=lang_tag)

    # Translate with the model
    model_path = MODEL_PATHS[0]
    output_file = OUTPUT_FILES[0]
    translate_with_model(model_path, preprocessed_src, output_file, gpu=0)

    # Compute BLEU score
    score = compute_bleu(output_file, REF_FILE)
    print(f"\nModel BLEU score: {score:.2f}")

    # Plot BLEU score
    plt.figure(figsize=(5, 4))
    plt.bar(["Model"], [score], color='skyblue')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score of Translation Model')
    plt.ylim(0, max(score * 1.1, 1))
    plt.text(0, score + 0.5, f"{score:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
