import os
import urllib.request
import subprocess
import ctranslate2
import transformers
import torch
import matplotlib.pyplot as plt
import sacrebleu
import shutil
from config import NLLB_MODELS, SPM_URL, SPM_PATH, IWSLT_TEST_SRC, IWSLT_TEST_REF, SOURCE_LANGUAGE, TARGET_LANGUAGE

def download_sentencepiece_model():
    if not os.path.isfile(SPM_PATH):
        print(f"Downloading SentencePiece model from {SPM_URL}...")
        urllib.request.urlretrieve(SPM_URL, SPM_PATH)
        print("SentencePiece model downloaded.")
    else:
        print("SentencePiece model already exists.")

def download_model(model_info):
    """Download a single model"""
    print(f"Downloading {model_info['name']} from HuggingFace Hub...")
    subprocess.run([
        "python", "-c",
        f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{model_info['repo_id']}', local_dir='{model_info['local_dir']}')"
    ], check=True)
    print(f"{model_info['name']} downloaded.")

def delete_model(model_info):
    """Delete a model directory to free up space"""
    if os.path.isdir(model_info["local_dir"]):
        print(f"Deleting {model_info['name']} to free up space...")
        shutil.rmtree(model_info["local_dir"])
        print(f"{model_info['name']} deleted.")

def translate_with_model(model_info, tokenizer, processed_sentences):
    print(f"\nTranslating with {model_info['name']}...")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load translator
    if DEVICE == "cuda":
        compute_type_cuda_options = ["int8_float16", "float16", "float32"]
        translator = None
        for c_type in compute_type_cuda_options:
            try:
                translator = ctranslate2.Translator(model_info["local_dir"], device=DEVICE, compute_type=c_type)
                print(f"Using compute_type='{c_type}' on GPU.")
                break
            except ValueError:
                print(f"Warning: {c_type} not supported on this GPU. Trying next option.")
        if translator is None:
            raise RuntimeError("No supported compute type found for GPU.")
    else:
        translator = ctranslate2.Translator(model_info["local_dir"], device=DEVICE, compute_type="int8")
    
    # Process in batches
    batch_size = 16  # Conservative batch size for multiple models
    translated_sentences = []
    SP_SPACE_CHAR = '\u2581'
    
    print(f"Processing {len(processed_sentences)} sentences in batches of {batch_size}...")
    
    for i in range(0, len(processed_sentences), batch_size):
        batch_sentences = processed_sentences[i:i+batch_size]
        target_prefix = [[TARGET_LANGUAGE]] * len(batch_sentences)
        
        print(f"Batch {i//batch_size + 1}/{(len(processed_sentences) + batch_size - 1)//batch_size}")
        
        translations = translator.translate_batch(
            batch_sentences,
            beam_size=2,
            target_prefix=target_prefix,
            max_decoding_length=128,
            repetition_penalty=1.0,
        )
        
        for translation in translations:
            tgt_tokens = translation.hypotheses[0]
            if tgt_tokens and tgt_tokens[0] == TARGET_LANGUAGE:
                tgt_tokens = tgt_tokens[1:]
            if tgt_tokens and tgt_tokens[-1] == "</s>":
                tgt_tokens = tgt_tokens[:-1]
            detokenized_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tgt_tokens), skip_special_tokens=True)
            final_cleaned_text = detokenized_text.replace(SP_SPACE_CHAR, ' ').strip()
            translated_sentences.append(final_cleaned_text)
    
    return translated_sentences

def main():
    # Ensure IWSLT14 test set is present
    subprocess.run(["bash", "prepare_iwslt14.sh"], check=True)
    
    # Download SentencePiece model
    download_sentencepiece_model()
    
    # Preprocess test set (do this once, before model loop)
    print("Preprocessing IWSLT14 test set...")
    
    # We need to load tokenizer first, so download the first model temporarily for tokenizer
    first_model = NLLB_MODELS[0]
    if not os.path.isdir(first_model["local_dir"]):
        download_model(first_model)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(first_model["local_dir"], sp_model_kwargs={"model_file": SPM_PATH})
    
    preprocessed_src = "preprocessed_test.de"
    with open(IWSLT_TEST_SRC, 'r', encoding='utf-8') as fin, open(preprocessed_src, 'w', encoding='utf-8') as fout:
        for line in fin:
            tokenizer.src_lang = SOURCE_LANGUAGE
            token_ids = tokenizer.encode(line.strip(), add_special_tokens=True)
            tokens_as_strings = tokenizer.convert_ids_to_tokens(token_ids)
            tokens_as_strings = [SOURCE_LANGUAGE] + tokens_as_strings
            fout.write(' '.join(tokens_as_strings) + '\n')
    
    with open(preprocessed_src, 'r', encoding='utf-8') as f:
        processed_sentences = [line.strip().split() for line in f]
    
    # Load reference sentences
    with open(IWSLT_TEST_REF, "r", encoding="utf-8") as f:
        reference_sentences = [line.strip() for line in f]
    
    # Evaluate each model one by one
    results = []
    for i, model_info in enumerate(NLLB_MODELS):
        try:
            # Download model if not already downloaded (first model is already downloaded)
            if i > 0 and not os.path.isdir(model_info["local_dir"]):
                download_model(model_info)
            
            # Translate and evaluate
            translations = translate_with_model(model_info, tokenizer, processed_sentences)
            
            # Save translations
            output_file = f"{model_info['name']}_translations.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for sentence in translations:
                    f.write(sentence + "\n")
            
            # Compute BLEU score
            bleu = sacrebleu.corpus_bleu(translations, [reference_sentences], tokenize="flores200")
            results.append({"name": model_info["name"], "bleu": bleu.score})
            print(f"{model_info['name']} BLEU Score: {bleu.score:.2f}")
            
            # Delete model to free up space (except for the last one, we'll delete it after plotting)
            delete_model(model_info)
            
        except Exception as e:
            print(f"Error with {model_info['name']}: {e}")
            results.append({"name": model_info["name"], "bleu": 0.0})
            # Still try to delete the model in case of error
            delete_model(model_info)
    
    # Plot comparison
    model_names = [r["name"] for r in results]
    bleu_scores = [r["bleu"] for r in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, bleu_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score Comparison: NLLB Models on IWSLT14 Test Set')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(bleu_scores) * 1.1 if bleu_scores else 1)
    
    for i, (bar, score) in enumerate(zip(bars, bleu_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{score:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\nFinal Results:")
    for result in results:
        print(f"{result['name']}: {result['bleu']:.2f}")
    
    # Clean up preprocessed file
    if os.path.exists(preprocessed_src):
        os.remove(preprocessed_src)

if __name__ == "__main__":
    main()