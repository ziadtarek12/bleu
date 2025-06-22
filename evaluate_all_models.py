import os
import urllib.request
import subprocess
import torch
import matplotlib.pyplot as plt
import sacrebleu
import shutil
import sys
from config import OPENNMT_MODELS, IWSLT_TEST_SRC, IWSLT_TEST_REF, SOURCE_LANGUAGE, TARGET_LANGUAGE

def download_file(url, local_path):
    """Download a file from URL to local path"""
    if not os.path.isfile(local_path):
        print(f"Downloading {local_path} from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"{local_path} downloaded.")
    else:
        print(f"{local_path} already exists.")

def install_opennmt_py():
    """Install OpenNMT-py if not already installed"""
    try:
        import onmt
        print("OpenNMT-py is already installed.")
    except ImportError:
        print("Installing OpenNMT-py...")
        subprocess.run([sys.executable, "-m", "pip", "install", "OpenNMT-py"], check=True)
        print("OpenNMT-py installed successfully.")

def install_subword_nmt():
    """Install subword-nmt for BPE processing"""
    try:
        import subword_nmt
        print("subword-nmt is already installed.")
    except ImportError:
        print("Installing subword-nmt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "subword-nmt"], check=True)
        print("subword-nmt installed successfully.")

def apply_bpe(text_file, bpe_model, output_file):
    """Apply BPE encoding to text file"""
    print(f"Applying BPE to {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        subprocess.run([
            "subword-nmt", "apply-bpe", 
            "-c", bpe_model,
            "--input", text_file,
            "--output", output_file
        ], check=True)
    print(f"BPE applied, output saved to {output_file}")

def translate_with_opennmt(model_info, src_file):
    """Translate using OpenNMT-py"""
    print(f"\nTranslating with {model_info['name']}...")
    
    # Apply BPE to source
    bpe_src_file = "test_bpe.en"
    apply_bpe(src_file, model_info['bpe_path'], bpe_src_file)
    
    # Translate with OpenNMT-py
    output_file = f"{model_info['name']}_raw_translations.txt"
    
    translate_cmd = [
        "onmt_translate",
        "-model", model_info['local_model_path'],
        "-src", bpe_src_file,
        "-output", output_file,
        "-replace_unk",
        "-verbose"
    ]
    
    # Add GPU support if available
    if torch.cuda.is_available():
        translate_cmd.extend(["-gpu", "0"])
    
    print(f"Running translation command: {' '.join(translate_cmd)}")
    subprocess.run(translate_cmd, check=True)
    
    # Read translations and clean up BPE
    with open(output_file, 'r', encoding='utf-8') as f:
        translations = [line.strip() for line in f]
    
    # Remove BPE encoding from translations (replace @@ with empty string)
    cleaned_translations = []
    for translation in translations:
        cleaned = translation.replace("@@ ", "").replace("@@", "")
        cleaned_translations.append(cleaned)
    
    # Save cleaned translations
    clean_output_file = f"{model_info['name']}_translations.txt"
    with open(clean_output_file, 'w', encoding='utf-8') as f:
        for translation in cleaned_translations:
            f.write(translation + '\n')
    
    # Clean up temporary files
    if os.path.exists(bpe_src_file):
        os.remove(bpe_src_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    
    return cleaned_translations

def main():
    # Install dependencies
    install_opennmt_py()
    install_subword_nmt()
    
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
            
            # Translate
            translations = translate_with_opennmt(model_info, IWSLT_TEST_SRC)
            
            # Compute BLEU score
            bleu = sacrebleu.corpus_bleu(translations, [reference_sentences])
            results.append({"name": model_info["name"], "bleu": bleu.score})
            print(f"{model_info['name']} BLEU Score: {bleu.score:.2f}")
            
        except Exception as e:
            print(f"Error with {model_info['name']}: {e}")
            results.append({"name": model_info["name"], "bleu": 0.0})
    
    # Plot comparison
    if results:
        model_names = [r["name"] for r in results]
        bleu_scores = [r["bleu"] for r in results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, bleu_scores, color=['skyblue'])
        plt.ylabel('BLEU Score')
        plt.title('BLEU Score: OpenNMT v3 EN-DE Model on IWSLT14 Test Set')
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