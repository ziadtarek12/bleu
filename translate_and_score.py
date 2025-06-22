import subprocess
import sys
import importlib.util
import os
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


def translate_with_model(model_path, src_file, output_file):
    cmd = [
        "onmt_translate",
        "-model", model_path,
        "-src", src_file,
        "-output", output_file,
        "-replace_unk",
        "-verbose"
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


def check_files():
    for path in MODEL_PATHS:
        if not os.path.isfile(path):
            print(f"Model file not found: {path}")
            sys.exit(1)
    if not os.path.isfile(SRC_FILE):
        print(f"Source file not found: {SRC_FILE}")
        sys.exit(1)
    if not os.path.isfile(REF_FILE):
        print(f"Reference file not found: {REF_FILE}")
        sys.exit(1)


def prepare_iwslt_data():
    """Run the shell script to prepare IWSLT data if not already done."""
    de_file = "data/de-en/test.de"
    en_file = "data/de-en/test.en"
    if not (os.path.isfile(de_file) and os.path.isfile(en_file)):
        print("Preparing IWSLT data (this may take a while)...")
        script_content = '''
set -x
set -e
CURRENT_DIR=$(pwd)
RAW="$CURRENT_DIR/data"
TMP="$CURRENT_DIR/data/tmp"
MOSES_SCRIPTS="$CURRENT_DIR/mosesdecoder/scripts"
TOKENIZER="$MOSES_SCRIPTS/tokenizer/tokenizer.perl"
LC="$MOSES_SCRIPTS/tokenizer/lowercase.perl"
CLEAN="$MOSES_SCRIPTS/training/clean-corpus-n.perl"
URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ="de-en.tgz"
src=de
tgt=en
lang=de-en
prep=$RAW/de-en
orig=$TMP
if [ ! -d "$MOSES_SCRIPTS" ]; then
    echo "Moses scripts not found. Cloning Moses repository..."
    git clone https://github.com/moses-smt/mosesdecoder.git
fi
mkdir -p $orig $prep
cd $orig
if [ -f $GZ ]; then
    echo "$GZ already exists, skipping download"
else
    echo "Downloading data from ${URL}..."
    wget "$URL"
    if [ -f $GZ ]; then
        echo "Data successfully downloaded."
    else
        echo "Data not successfully downloaded."
        exit
    fi
    tar zxvf $GZ
fi
cd -
if [ -f $prep/train.en ] && [ -f $prep/train.de ] && \
    [ -f $prep/valid.en ] && [ -f $prep/valid.de ] && \
    [ -f $prep/test.en ] && [ -f $prep/test.de ]; then
    echo "iwslt dataset is already preprocessed, skip"
else
    echo "pre-processing train data..."
    for l in $src $tgt; do
        f=train.tags.$lang.$l
        tok=train.tags.$lang.tok.$l
        cat $orig/$lang/$f | \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        sed -e 's/<title>//g' | \
        sed -e 's/<\/title>//g' | \
        sed -e 's/<description>//g' | \
        sed -e 's/<\/description>//g' | \
        perl $TOKENIZER -threads 8 -l $l > $prep/$tok
        echo ""
    done
    perl $CLEAN -ratio 1.5 $prep/train.tags.$lang.tok $src $tgt $prep/train.tags.$lang.clean 1 175
    for l in $src $tgt; do
        perl $LC < $prep/train.tags.$lang.clean.$l > $prep/train.tags.$lang.$l
    done
    echo "pre-processing valid/test data..."
    for l in $src $tgt; do
        for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$prep/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -l $l | \
        perl $LC > $f
        echo ""
        done
    done
    echo "creating train, valid, test..."
    for l in $src $tgt; do
        awk '{if (NR%23 == 0)  print $0; }' $prep/train.tags.de-en.$l > $prep/valid.$l
        awk '{if (NR%23 != 0)  print $0; }' $prep/train.tags.de-en.$l > $prep/train.$l
        cat $prep/IWSLT14.TED.dev2010.de-en.$l \
            $prep/IWSLT14.TEDX.dev2012.de-en.$l \
            $prep/IWSLT14.TED.tst2010.de-en.$l \
            $prep/IWSLT14.TED.tst2011.de-en.$l \
            $prep/IWSLT14.TED.tst2012.de-en.$l \
            > $prep/test.$l
    done
fi
'''
        with open("prepare_iwslt14.sh", "w") as f:
            f.write(script_content)
        subprocess.run(["bash", "prepare_iwslt14.sh"], check=True)
        os.remove("prepare_iwslt14.sh")
        print("IWSLT data preparation complete.")
    else:
        print("IWSLT data already prepared.")


def download_models(model_urls, model_paths):
    """Download models from URLs if not already present."""
    os.makedirs("models", exist_ok=True)  # Ensure models directory exists
    for url, path in zip(model_urls, model_paths):
        if not os.path.isfile(path):
            print(f"Downloading model from {url} to {path}...")
            try:
                subprocess.run(["wget", "-O", path, url], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {url}: {e}")
                sys.exit(1)
        else:
            print(f"Model already exists: {path}")


def main():
    from config import MODEL_URLS
    download_models(MODEL_URLS, MODEL_PATHS)
    prepare_iwslt_data()
    check_files()
    # Translate with each model
    for i, model_path in enumerate(MODEL_PATHS):
        translate_with_model(model_path, SRC_FILE, OUTPUT_FILES[i])
    # Compute BLEU scores
    bleu_scores = []
    for output_file in OUTPUT_FILES:
        score = compute_bleu(output_file, REF_FILE)
        bleu_scores.append(score)
    # Print BLEU scores
    print("\nBLEU score comparison:")
    for i, score in enumerate(bleu_scores):
        print(f"Model {i+1} BLEU score: {score:.2f}")
    # Plot BLEU scores
    plt.figure(figsize=(8, 5))
    plt.bar([f"Model {i+1}" for i in range(len(bleu_scores))], bleu_scores, color='skyblue')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score Comparison of Translation Models')
    plt.ylim(0, max(bleu_scores) * 1.1 if bleu_scores else 1)
    for i, v in enumerate(bleu_scores):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
