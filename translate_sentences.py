import os
import urllib.request
import tarfile
import shutil
import sentencepiece as spm
import sacrebleu
import ctranslate2
import transformers
import torch
import matplotlib.pyplot as plt




# --- Download SentencePiece Model if Needed ---
SPM_URL = "https://s3.amazonaws.com/opennmt-models/nllb-200/flores200_sacrebleu_tokenizer_spm.model"
SPM_PATH = "flores200_sacrebleu_tokenizer_spm.model"
if not os.path.isfile(SPM_PATH):
    print(f"Downloading SentencePiece model from {SPM_URL}...")
    urllib.request.urlretrieve(SPM_URL, SPM_PATH)
    print("SentencePiece model downloaded.")
else:
    print("SentencePiece model already exists.")

# --- Download CTranslate2 Model if Needed ---
CT2_MODEL_URL = "https://huggingface.co/facebook/nllb-200-distilled-1.3B-ct2-int8/resolve/main/model.bin"
CT2_MODEL_DIR = "nllb_ct2_models/nllb-200-distilled-1.3B-ct2-int8"
CT2_MODEL_BIN = os.path.join(CT2_MODEL_DIR, "model.bin")

if not os.path.isfile(CT2_MODEL_BIN):
    print(f"Downloading CTranslate2 model from {CT2_MODEL_URL}...")
    os.makedirs(CT2_MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(CT2_MODEL_URL, CT2_MODEL_BIN)
    print("Model downloaded.")
else:
    print("CTranslate2 model already exists.")

# --- Load Model and Tokenizer ---
MODEL_DIR = CT2_MODEL_DIR
SOURCE_LANGUAGE = "deu_Latn"
TARGET_LANGUAGE = "eng_Latn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"CTranslate2 will attempt to use device: {DEVICE}")

print(f"Loading CTranslate2 model from: {MODEL_DIR}...")
try:
    if DEVICE == "cuda":
        compute_type_cuda_options = ["float16", "int8_float16", "float32"]
        selected_compute_type = None
        for c_type in compute_type_cuda_options:
            try:
                translator = ctranslate2.Translator(MODEL_DIR, device=DEVICE, compute_type=c_type)
                selected_compute_type = c_type
                print(f"Using compute_type='{selected_compute_type}' on GPU.")
                break
            except ValueError:
                print(f"Warning: {c_type} not supported on this GPU. Trying next option.")
        if selected_compute_type is None:
            raise RuntimeError("No supported compute type found for GPU.")
    else: # DEVICE == "cpu"
        try:
            translator = ctranslate2.Translator(MODEL_DIR, device=DEVICE, compute_type="int8")
            print("Using compute_type='int8' on CPU.")
        except ValueError:
            print("Warning: int8 not supported on this CPU. Falling back to float32.")
            translator = ctranslate2.Translator(MODEL_DIR, device=DEVICE, compute_type="float32")
            print("Using compute_type='float32' on CPU.")
except RuntimeError as e:
    print(f"FATAL ERROR during CTranslate2 model loading: {e}")
    print("Please ensure the MODEL_DIR is correct and the model files are complete and not corrupted.")
    exit()

print("Loading Hugging Face AutoTokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, sp_model_kwargs={"model_file": SPM_PATH})

# --- Preprocess IWSLT14 Test Set ---
print("Preprocessing IWSLT14 test set with SentencePiece and language tag...")
preprocessed_src = "preprocessed_test.de"
with open(IWSLT_TEST_SRC, 'r', encoding='utf-8') as fin, open(preprocessed_src, 'w', encoding='utf-8') as fout:
    for line in fin:
        tokenizer.src_lang = SOURCE_LANGUAGE
        token_ids = tokenizer.encode(line.strip(), add_special_tokens=True)
        tokens_as_strings = tokenizer.convert_ids_to_tokens(token_ids)
        # Prepend source language tag
        tokens_as_strings = [SOURCE_LANGUAGE] + tokens_as_strings
        fout.write(' '.join(tokens_as_strings) + '\n')

# --- Translate the Test Set ---
print("Translating the IWSLT14 test set...")
with open(preprocessed_src, 'r', encoding='utf-8') as f:
    processed_source_sentences = [line.strip().split() for line in f]

target_prefix = [[TARGET_LANGUAGE]] * len(processed_source_sentences)
translations = translator.translate_batch(
    processed_source_sentences,
    beam_size=5,
    target_prefix=target_prefix,
    max_decoding_length=256,
    repetition_penalty=1.0,
)

# --- Postprocess Output (Detokenization for BLEU) ---
translated_sentences_detokenized = []
SP_SPACE_CHAR = '\u2581'  # SentencePiece space char
for translation in translations:
    tgt_tokens = translation.hypotheses[0]
    if tgt_tokens and tgt_tokens[0] == TARGET_LANGUAGE:
        tgt_tokens = tgt_tokens[1:]
    if tgt_tokens and tgt_tokens[-1] == "</s>":
        tgt_tokens = tgt_tokens[:-1]
    detokenized_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tgt_tokens), skip_special_tokens=True)
    final_cleaned_text = detokenized_text.replace(SP_SPACE_CHAR, ' ').strip()
    translated_sentences_detokenized.append(final_cleaned_text)

# --- Save Translations to File ---
output_file = "nllb_ct2_translations.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for sentence in translated_sentences_detokenized:
        f.write(sentence + "\n")
print(f"Translations saved to: {output_file}")

# --- Prepare Reference Data for BLEU ---
reference_file = IWSLT_TEST_REF
with open(reference_file, "r", encoding="utf-8") as f:
    reference_sentences = [line.strip() for line in f]

# --- Compute BLEU Score ---
hypotheses = translated_sentences_detokenized
references = [reference_sentences]
bleu = sacrebleu.corpus_bleu(hypotheses, references, tokenize="flores200")
print(f"\nBLEU Score on IWSLT14 test set: {bleu.score:.2f}")

# --- Plot BLEU Score ---
plt.figure(figsize=(5, 4))
plt.bar(["Model"], [bleu.score], color='skyblue')
plt.ylabel('BLEU Score')
plt.title('BLEU Score on IWSLT14 Test Set')
plt.ylim(0, max(bleu.score * 1.1, 1))
plt.text(0, bleu.score + 0.5, f"{bleu.score:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.show()
