import os
import torch
import subprocess
import sacrebleu
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, pipeline
from config import IWSLT_TEST_SRC, IWSLT_TEST_REF

# BERT model configuration
BERT_MODEL_NAME = "bert-base-multilingual-cased"

def download_dependencies():
    """Ensure IWSLT14 test set is present"""
    subprocess.run(["bash", "prepare_iwslt14.sh"], check=True)

def load_bert_translator():
    """Load BERT model for translation"""
    print(f"📥 Loading {BERT_MODEL_NAME}...")
    
    # Try to use translation pipeline if available
    try:
        # Some BERT models might have translation capabilities
        translator = pipeline("translation", model=BERT_MODEL_NAME, 
                            src_lang="de", tgt_lang="en", 
                            device=0 if torch.cuda.is_available() else -1)
        print("✅ Translation pipeline loaded successfully")
        return translator, "pipeline"
    except Exception as e:
        print(f"⚠️  Translation pipeline failed: {e}")
        
    # Fallback: Try text2text generation
    try:
        translator = pipeline("text2text-generation", model=BERT_MODEL_NAME,
                            device=0 if torch.cuda.is_available() else -1)
        print("✅ Text2text generation pipeline loaded")
        return translator, "text2text"
    except Exception as e:
        print(f"⚠️  Text2text generation failed: {e}")
        
    # Final fallback: Load as fill-mask (not ideal for translation)
    try:
        translator = pipeline("fill-mask", model=BERT_MODEL_NAME,
                            device=0 if torch.cuda.is_available() else -1)
        print("✅ Fill-mask pipeline loaded (limited translation capability)")
        return translator, "fill-mask"
    except Exception as e:
        print(f"❌ All pipeline methods failed: {e}")
        return None, None

def translate_with_bert(sentences, translator, method):
    """Translate sentences using BERT model"""
    translations = []
    
    if method == "pipeline":
        # Direct translation pipeline
        for sentence in sentences:
            try:
                result = translator(sentence)
                if isinstance(result, list) and len(result) > 0:
                    translations.append(result[0].get('translation_text', sentence))
                else:
                    translations.append(sentence)  # Fallback to original
            except Exception as e:
                print(f"⚠️  Translation failed for: {sentence[:50]}...")
                translations.append(sentence)  # Fallback to original
                
    elif method == "text2text":
        # Text2text generation with prompting
        for sentence in sentences:
            try:
                prompt = f"Translate German to English: {sentence}"
                result = translator(prompt, max_length=128, num_return_sequences=1)
                if isinstance(result, list) and len(result) > 0:
                    translations.append(result[0].get('generated_text', sentence).replace(prompt, "").strip())
                else:
                    translations.append(sentence)
            except Exception as e:
                print(f"⚠️  Translation failed for: {sentence[:50]}...")
                translations.append(sentence)
                
    elif method == "fill-mask":
        # Limited capability - mostly pass-through
        print("⚠️  Using fill-mask method - limited translation capability")
        translations = sentences  # BERT fill-mask can't really translate
        
    else:
        print("❌ No valid translation method available")
        translations = sentences  # Pass-through
        
    return translations

def evaluate_bert_translation():
    """Evaluate BERT model on IWSLT14 test set"""
    print("🚀 Starting BERT-base-multilingual-cased evaluation...")
    
    # Ensure data is ready
    download_dependencies()
    
    # Load BERT translator
    translator, method = load_bert_translator()
    if translator is None:
        print("❌ Failed to load BERT translator")
        return None
    
    # Load test data
    print("📚 Loading IWSLT14 test data...")
    with open(IWSLT_TEST_SRC, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f]
    
    with open(IWSLT_TEST_REF, 'r', encoding='utf-8') as f:
        reference_sentences = [line.strip() for line in f]
    
    print(f"📊 Loaded {len(source_sentences)} sentence pairs")
    
    # Translate (process in smaller batches to avoid memory issues)
    print("🔄 Translating with BERT...")
    batch_size = 50  # Small batches for memory efficiency
    all_translations = []
    
    for i in range(0, len(source_sentences), batch_size):
        batch = source_sentences[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(source_sentences) + batch_size - 1)//batch_size}")
        
        batch_translations = translate_with_bert(batch, translator, method)
        all_translations.extend(batch_translations)
    
    # Save translations
    output_file = "bert_multilingual_translations.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for translation in all_translations:
            f.write(translation + "\n")
    print(f"💾 Translations saved to: {output_file}")
    
    # Compute BLEU score
    print("📈 Computing BLEU score...")
    try:
        bleu = sacrebleu.corpus_bleu(all_translations, [reference_sentences])
        bleu_score = bleu.score
        print(f"📊 BERT BLEU Score: {bleu_score:.2f}")
        
        # Plot results
        plt.figure(figsize=(8, 5))
        plt.bar(["BERT-base-multilingual-cased"], [bleu_score], color='lightblue')
        plt.ylabel('BLEU Score')
        plt.title('BERT-base-multilingual-cased BLEU Score on IWSLT14')
        plt.ylim(0, max(bleu_score * 1.1, 1))
        plt.text(0, bleu_score + 0.5, f"{bleu_score:.2f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        
        return bleu_score
        
    except Exception as e:
        print(f"❌ BLEU computation failed: {e}")
        return None

def main():
    print("🎯 BERT-base-multilingual-cased BLEU Evaluation")
    print("="*50)
    print("⚠️  Note: BERT is primarily designed for understanding tasks,")
    print("   not generation. Translation performance may be limited.")
    print("="*50)
    
    bleu_score = evaluate_bert_translation()
    
    if bleu_score is not None:
        print(f"\n✅ Final BLEU Score: {bleu_score:.2f}")
        
        # Add context about the score
        if bleu_score < 5:
            print("💡 Very low BLEU score - BERT is not designed for translation")
        elif bleu_score < 15:
            print("💡 Low BLEU score - limited translation capability")
        else:
            print("💡 Unexpected good performance for a non-translation model!")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main()