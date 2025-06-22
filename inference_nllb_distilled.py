import os
import urllib.request
import subprocess
import ctranslate2
import transformers
import torch
import argparse

# Configuration for NLLB-200-1.3B-distilled
MODEL_INFO = {
    "name": "NLLB-200-1.3B-distilled",
    "repo_id": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
    "local_dir": "nllb-200-distilled-1.3B-ctranslate2"
}

SPM_URL = "https://s3.amazonaws.com/opennmt-models/nllb-200/flores200_sacrebleu_tokenizer_spm.model"
SPM_PATH = "flores200_sacrebleu_tokenizer_spm.model"

# Language settings (can be modified for different language pairs)
TRANSLATION_MODES = {
    "de2en": {
        "source": "deu_Latn",  # German
        "target": "eng_Latn",  # English
        "name": "German → English",
        "source_flag": "🇩🇪",
        "target_flag": "🇺🇸"
    },
    "en2de": {
        "source": "eng_Latn",  # English
        "target": "deu_Latn",  # German
        "name": "English → German",
        "source_flag": "🇺🇸",
        "target_flag": "🇩🇪"
    }
}

# Default mode
DEFAULT_MODE = "de2en"

def download_dependencies():
    """Download SentencePiece model if needed"""
    if not os.path.isfile(SPM_PATH):
        print("📥 Downloading SentencePiece tokenizer...")
        urllib.request.urlretrieve(SPM_URL, SPM_PATH)
        print("✅ SentencePiece tokenizer downloaded.")
    else:
        print("✅ SentencePiece tokenizer already exists.")

def download_model():
    """Download the NLLB CTranslate2 model if needed"""
    if not os.path.isdir(MODEL_INFO["local_dir"]):
        print(f"📥 Downloading {MODEL_INFO['name']} CTranslate2 model...")
        print("⏳ This may take a few minutes...")
        try:
            subprocess.run([
                "python", "-c",
                f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{MODEL_INFO['repo_id']}', local_dir='{MODEL_INFO['local_dir']}')"
            ], check=True)
            print(f"✅ {MODEL_INFO['name']} downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download model: {e}")
            print("Make sure you have huggingface_hub installed: pip install huggingface_hub")
            exit(1)
    else:
        print(f"✅ {MODEL_INFO['name']} already exists.")

def load_model_and_tokenizer():
    """Load the CTranslate2 model and tokenizer"""
    # Enhanced GPU detection and diagnostics
    print("🔍 GPU Diagnostics:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
        print(f"   Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.3f} GB")
    else:
        print("   No CUDA devices found")
    
    # Check CTranslate2 CUDA support
    try:
        import ctranslate2
        print(f"   CTranslate2 version: {ctranslate2.__version__}")
        print(f"   CTranslate2 CUDA: {ctranslate2.get_cuda_device_count() > 0}")
        if ctranslate2.get_cuda_device_count() > 0:
            print(f"   CTranslate2 CUDA devices: {ctranslate2.get_cuda_device_count()}")
    except Exception as e:
        print(f"   CTranslate2 error: {e}")
    
    DEVICE = "cuda" if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0 else "cpu"
    print(f"🔧 Selected device: {DEVICE.upper()}")
    
    # Load CTranslate2 translator
    if DEVICE == "cuda":
        print("🚀 Attempting to load on GPU...")
        compute_type_options = ["int8_float16", "float16", "float32"]
        translator = None
        for c_type in compute_type_options:
            try:
                print(f"   Trying {c_type}...", end="")
                translator = ctranslate2.Translator(MODEL_INFO["local_dir"], device=DEVICE, compute_type=c_type)
                print(f" ✅ SUCCESS")
                print(f"✅ GPU loading successful with compute_type='{c_type}'")
                break
            except Exception as e:
                print(f" ❌ FAILED: {str(e)}")
        
        if translator is None:
            print("⚠️  All GPU compute types failed, falling back to CPU...")
            translator = ctranslate2.Translator(MODEL_INFO["local_dir"], device="cpu", compute_type="int8")
            print("✅ CPU fallback successful")
    else:
        print("💻 Loading directly on CPU...")
        translator = ctranslate2.Translator(MODEL_INFO["local_dir"], device="cpu", compute_type="int8")
        print("✅ CPU loading successful")
    
    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_INFO["local_dir"], sp_model_kwargs={"model_file": SPM_PATH})
    print("✅ Tokenizer loaded successfully")
    
    return translator, tokenizer

def translate_sentence(sentence, translator, tokenizer, mode=DEFAULT_MODE):
    """Translate a single sentence"""
    # Preprocess sentence
    source_lang = TRANSLATION_MODES[mode]["source"]
    target_lang = TRANSLATION_MODES[mode]["target"]
    
    tokenizer.src_lang = source_lang
    token_ids = tokenizer.encode(sentence.strip(), add_special_tokens=True)
    tokens_as_strings = tokenizer.convert_ids_to_tokens(token_ids)
    tokens_as_strings = [source_lang] + tokens_as_strings
    
    # Translate
    target_prefix = [[target_lang]]
    translations = translator.translate_batch(
        [tokens_as_strings],
        beam_size=5,
        target_prefix=target_prefix,
        max_decoding_length=256,
        repetition_penalty=1.0,
    )
    
    # Postprocess translation
    SP_SPACE_CHAR = '\u2581'
    translation = translations[0]
    tgt_tokens = translation.hypotheses[0]
    
    if tgt_tokens and tgt_tokens[0] == target_lang:
        tgt_tokens = tgt_tokens[1:]
    if tgt_tokens and tgt_tokens[-1] == "</s>":
        tgt_tokens = tgt_tokens[:-1]
    
    detokenized_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tgt_tokens), skip_special_tokens=True)
    final_cleaned_text = detokenized_text.replace(SP_SPACE_CHAR, ' ').strip()
    
    return final_cleaned_text

def interactive_mode(translator, tokenizer, mode=DEFAULT_MODE):
    """Interactive command-line translation mode"""
    source_flag = TRANSLATION_MODES[mode]["source_flag"]
    target_flag = TRANSLATION_MODES[mode]["target_flag"]
    source_lang = TRANSLATION_MODES[mode]["source"]
    target_lang = TRANSLATION_MODES[mode]["target"]
    
    print("\n" + "="*60)
    print(f"🌍 NLLB-200-1.3B-distilled Interactive Translator ({TRANSLATION_MODES[mode]['name']})")
    print(f"{source_flag} {source_lang} → {target_flag} {target_lang}")
    print("="*60)
    print("💡 Tips:")
    print("   • Type sentences to get translations")
    print("   • Type 'quit', 'exit', or 'q' to exit")
    print("   • Press Ctrl+C to exit anytime")
    print("="*60)
    
    while True:
        try:
            print()
            sentence = input(f"{source_flag} {source_lang}: ").strip()
            
            if sentence.lower() in ['quit', 'exit', 'q', '']:
                print("👋 Goodbye!")
                break
                
            if sentence:
                print("⏳ Translating...", end="", flush=True)
                try:
                    translation = translate_sentence(sentence, translator, tokenizer, mode)
                    print(f"\r{target_flag} {target_lang}: {translation}")
                except Exception as e:
                    print(f"\r❌ Translation error: {e}")
            else:
                print("⚠️  Please enter a sentence to translate.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="NLLB-200-1.3B-distilled Interactive Translator")
    parser.add_argument("--sentence", "-s", type=str, help="Translate a single sentence and exit")
    parser.add_argument("--download-only", action="store_true", help="Only download the model and exit")
    parser.add_argument("--mode", "-m", type=str, choices=TRANSLATION_MODES.keys(), default=DEFAULT_MODE,
                        help="Translation mode: 'de2en' for German→English, 'en2de' for English→German")
    args = parser.parse_args()
    
    print("🚀 Starting NLLB-200-1.3B-distilled Translator...")
    print()
    
    # Download dependencies and model
    download_dependencies()
    download_model()
    
    if args.download_only:
        print("✅ Model download completed. Exiting.")
        return
    
    # Load model
    print("\n🔄 Initializing model...")
    translator, tokenizer = load_model_and_tokenizer()
    
    if args.sentence:
        # Single sentence mode
        print(f"\n🔤 Translating: '{args.sentence}'")
        try:
            translation = translate_sentence(args.sentence, translator, tokenizer, args.mode)
            print(f"✅ Translation: {translation}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Interactive mode
        interactive_mode(translator, tokenizer, args.mode)

if __name__ == "__main__":
    main()