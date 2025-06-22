import os
import urllib.request
import subprocess
import torch
import matplotlib.pyplot as plt
import sacrebleu
import shutil
import sys
import ctranslate2
from config import OPENNMT_MODELS, IWSLT_TEST_SRC, IWSLT_TEST_REF, SOURCE_LANGUAGE, TARGET_LANGUAGE

def download_file(url, local_path):
    """Download a file from URL to local path"""
    if not os.path.isfile(local_path):
        print(f"Downloading {local_path} from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"{local_path} downloaded.")
    else:
        print(f"{local_path} already exists.")

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "OpenNMT-py",
        "ctranslate2", 
        "subword-nmt"
    ]
    
    for dep in dependencies:
        try:
            if dep == "OpenNMT-py":
                import onmt
            elif dep == "ctranslate2":
                import ctranslate2
            elif dep == "subword-nmt":
                import subword_nmt
            print(f"{dep} is already installed.")
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"{dep} installed successfully.")

def convert_to_ct2(model_info):
    """Convert OpenNMT model to CTranslate2 format"""
    if os.path.exists(model_info['ct2_model_path']):
        print(f"CTranslate2 model already exists at {model_info['ct2_model_path']}")
        return
    
    print(f"Converting {model_info['local_model_path']} to CTranslate2 format...")
    
    try:
        subprocess.run([
            "ct2-opennmt-py-converter",
            "--model_path", model_info['local_model_path'],
            "--output_dir", model_info['ct2_model_path'],
            "--quantization", "int8"
        ], check=True)
        print(f"Model converted to CTranslate2 format at {model_info['ct2_model_path']}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting model: {e}")
        raise

def apply_bpe_to_text(text_lines, bpe_model_path):
    """Apply BPE to a list of text lines"""
    # Create temporary input file
    temp_input = "temp_input.txt"
    temp_output = "temp_output.txt"
    
    with open(temp_input, 'w', encoding='utf-8') as f:
        for line in text_lines:
            f.write(line + '\n')
    
    # Apply BPE
    subprocess.run([
        "subword-nmt", "apply-bpe",
        "-c", bpe_model_path,
        "--input", temp_input,
        "--output", temp_output
    ], check=True)
    
    # Read BPE-encoded text
    with open(temp_output, 'r', encoding='utf-8') as f:
        bpe_lines = [line.strip().split() for line in f]
    
    # Clean up
    os.remove(temp_input)
    os.remove(temp_output)
    
    return bpe_lines

def translate_with_ct2(model_info, src_file):
    """Translate using CTranslate2"""
    print(f"\nTranslating with {model_info['name']} using CTranslate2...")
    
    # Load source sentences
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f]
    
    # Apply BPE to source sentences
    print("Applying BPE to source sentences...")
    bpe_sentences = apply_bpe_to_text(src_sentences, model_info['bpe_path'])
    
    # Initialize CTranslate2 translator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = ctranslate2.Translator(model_info['ct2_model_path'], device=device)
    
    # Translate in batches
    batch_size = 32
    all_translations = []
    
    print(f"Translating {len(bpe_sentences)} sentences in batches of {batch_size}...")
    
    for i in range(0, len(bpe_sentences), batch_size):
        batch = bpe_sentences[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(bpe_sentences) + batch_size - 1)//batch_size}")
        
        # Translate batch
        results = translator.translate_batch(
            batch,
            beam_size=4,
            max_decoding_length=200,
            length_penalty=0.6
        )
        
        # Extract translations and clean up BPE
        for result in results:
            translation_tokens = result.hypotheses[0]
            # Join tokens and clean up BPE markers
            translation = ' '.join(translation_tokens)
            # Remove BPE markers (@@)
            translation = translation.replace('@@ ', '').replace('@@', '')
            # Remove special markup tokens
            import re
            translation = re.sub(r'｟[^｠]*｠', '', translation)  # Remove markup tokens
            translation = re.sub(r'<unk>', '', translation)     # Remove <unk> tokens
            translation = re.sub(r'\s+', ' ', translation).strip()  # Clean up whitespace
            all_translations.append(translation)
    
    # Save translations
    output_file = f"{model_info['name']}_translations.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in all_translations:
            f.write(translation + '\n')
    
    return all_translations

def main():
    # Install dependencies
    install_dependencies()
    
    # Ensure IWSLT14 test set is present
    subprocess.run(["bash", "prepare_iwslt14.sh"], check=True)
    
    # Load reference sentences
    with open(IWSLT_TEST_REF, "r", encoding="utf-8") as f:
        reference_sentences = [line.strip() for line in f]
    
    # Evaluate each model
    results = []
    for model_info in OPENNMT_MODELS:
        try:
            # Download model and BPE files
            download_file(model_info['model_url'], model_info['local_model_path'])
            download_file(model_info['bpe_url'], model_info['bpe_path'])
            
            # Convert to CTranslate2 format
            convert_to_ct2(model_info)
            
            # Translate using CTranslate2
            translations = translate_with_ct2(model_info, IWSLT_TEST_SRC)
            
            # Compute BLEU score
            bleu = sacrebleu.corpus_bleu(translations, [reference_sentences])
            results.append({"name": model_info["name"], "bleu": bleu.score})
            print(f"{model_info['name']} BLEU Score: {bleu.score:.2f}")
            
            # Show sample translations
            print(f"\nSample translations from {model_info['name']}:")
            for i in range(min(3, len(translations))):
                with open(IWSLT_TEST_SRC, 'r', encoding='utf-8') as f:
                    src_lines = f.readlines()
                print(f"SRC: {src_lines[i].strip()}")
                print(f"REF: {reference_sentences[i]}")
                print(f"HYP: {translations[i]}")
                print("-" * 50)
            
        except Exception as e:
            print(f"Error with {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": model_info["name"], "bleu": 0.0})
    
    # Plot comparison
    if results:
        model_names = [r["name"] for r in results]
        bleu_scores = [r["bleu"] for r in results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, bleu_scores, color=['skyblue'])
        plt.ylabel('BLEU Score')
        plt.title('BLEU Score: OpenNMT v3 EN-DE Model on IWSLT14 Test Set (CTranslate2)')
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

if __name__ == "__main__":
    main()