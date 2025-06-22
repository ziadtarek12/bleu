import os
import torch
import subprocess
import sacrebleu
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from config import IWSLT_TEST_SRC, IWSLT_TEST_REF

# BERT CMLM model configuration
MODEL_NAME = "pythn/gomaa_grad"
MODEL_DISPLAY_NAME = "BERT CMLM (pythn/gomaa_grad)"

def download_dependencies():
    """Ensure IWSLT14 test set is present"""
    subprocess.run(["bash", "prepare_iwslt14.sh"], check=True)

def load_bert_cmlm_model():
    """Load BERT CMLM model for translation"""
    print(f"📥 Loading {MODEL_DISPLAY_NAME}...")
    
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        # Try as translation pipeline first
        translator = pipeline("translation", 
                            model=MODEL_NAME,
                            src_lang="de", 
                            tgt_lang="en",
                            device=device)
        print("✅ Translation pipeline loaded successfully")
        return translator, "translation"
    except Exception as e:
        print(f"⚠️  Translation pipeline failed: {e}")
        
    try:
        # Try as text2text generation
        translator = pipeline("text2text-generation",
                            model=MODEL_NAME,
                            device=device)
        print("✅ Text2text generation pipeline loaded")
        return translator, "text2text"
    except Exception as e:
        print(f"⚠️  Text2text generation failed: {e}")
        
    try:
        # Try as fill-mask (CMLM approach)
        translator = pipeline("fill-mask",
                            model=MODEL_NAME,
                            device=device)
        print("✅ Fill-mask pipeline loaded (CMLM)")
        return translator, "fill-mask"
    except Exception as e:
        print(f"❌ All pipeline methods failed: {e}")
        return None, None

def translate_with_cmlm(sentences, translator, method):
    """Translate sentences using BERT CMLM model"""
    translations = []
    
    print(f"🔄 Translating {len(sentences)} sentences...")
    
    if method == "translation":
        # Direct translation pipeline
        for i, sentence in enumerate(sentences):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(sentences)}")
            try:
                result = translator(sentence, max_length=128)
                if isinstance(result, list) and len(result) > 0:
                    translations.append(result[0]['translation_text'])
                else:
                    translations.append("")
            except Exception as e:
                print(f"⚠️  Translation failed for sentence {i + 1}")
                translations.append("")
                
    elif method == "text2text":
        # Text2text generation with prompting
        for i, sentence in enumerate(sentences):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(sentences)}")
            try:
                prompt = f"Translate German to English: {sentence}"
                result = translator(prompt, max_length=128, num_return_sequences=1)
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0]['generated_text']
                    # Remove the prompt from the output
                    if prompt in generated:
                        generated = generated.replace(prompt, "").strip()
                    translations.append(generated)
                else:
                    translations.append("")
            except Exception as e:
                print(f"⚠️  Translation failed for sentence {i + 1}")
                translations.append("")
                
    elif method == "fill-mask":
        # CMLM approach using masking
        for i, sentence in enumerate(sentences):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(sentences)}")
            try:
                # For CMLM, we might need a different approach
                # Try to create a template for translation
                masked_template = f"German: {sentence} English: [MASK]"
                result = translator(masked_template)
                
                if isinstance(result, list) and len(result) > 0:
                    # Take the top prediction
                    prediction = result[0]['token_str'] if 'token_str' in result[0] else ""
                    translations.append(prediction)
                else:
                    translations.append("")
            except Exception as e:
                print(f"⚠️  Translation failed for sentence {i + 1}")
                translations.append("")
    
    return translations

def evaluate_bert_cmlm():
    """Evaluate BERT CMLM model on IWSLT14 test set"""
    print("🚀 Starting BERT CMLM Translation Evaluation...")
    
    # Ensure data is ready
    download_dependencies()
    
    # Load model
    translator, method = load_bert_cmlm_model()
    if translator is None:
        print("❌ Failed to load BERT CMLM model")
        return None
    
    # Load test data
    print("📚 Loading IWSLT14 test data...")
    with open(IWSLT_TEST_SRC, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f]
    
    with open(IWSLT_TEST_REF, 'r', encoding='utf-8') as f:
        reference_sentences = [line.strip() for line in f]
    
    print(f"📊 Loaded {len(source_sentences)} sentence pairs")
    
    # Limit to first 1000 sentences for faster evaluation
    if len(source_sentences) > 1000:
        print("⚡ Using first 1000 sentences for faster evaluation...")
        source_sentences = source_sentences[:1000]
        reference_sentences = reference_sentences[:1000]
    
    # Translate
    print(f"🔧 Using method: {method}")
    translations = translate_with_cmlm(source_sentences, translator, method)
    
    # Save translations
    output_file = f"bert_cmlm_gomaa_grad_translations.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for translation in translations:
            f.write(translation + "\n")
    print(f"💾 Translations saved to: {output_file}")
    
    # Show some examples
    print("\n📝 Sample translations:")
    for i in range(min(5, len(source_sentences))):
        print(f"DE: {source_sentences[i]}")
        print(f"EN: {translations[i]}")
        print(f"REF: {reference_sentences[i]}")
        print("-" * 50)
    
    # Compute BLEU score
    print("📈 Computing BLEU score...")
    try:
        bleu = sacrebleu.corpus_bleu(translations, [reference_sentences])
        bleu_score = bleu.score
        print(f"📊 BERT CMLM BLEU Score: {bleu_score:.2f}")
        
        # Plot results
        plt.figure(figsize=(8, 5))
        plt.bar([MODEL_DISPLAY_NAME], [bleu_score], color='lightgreen')
        plt.ylabel('BLEU Score')
        plt.title('BERT CMLM (pythn/gomaa_grad) BLEU Score on IWSLT14')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(bleu_score * 1.1, 1))
        plt.text(0, bleu_score + 0.5, f"{bleu_score:.2f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        
        return bleu_score
        
    except Exception as e:
        print(f"❌ BLEU computation failed: {e}")
        return None

def main():
    print("🎯 BERT CMLM Translation Model Evaluation")
    print("="*60)
    print(f"📝 Evaluating: {MODEL_DISPLAY_NAME}")
    print("📝 This is a BERT CMLM model fine-tuned for translation")
    print("="*60)
    
    bleu_score = evaluate_bert_cmlm()
    
    if bleu_score is not None:
        print(f"\n✅ Final BLEU Score: {bleu_score:.2f}")
        
        # Add context about the score
        if bleu_score < 5:
            print("💡 Low BLEU score - model may need different prompting approach")
        elif bleu_score < 15:
            print("💡 Moderate BLEU score - reasonable for a BERT-based translation model")
        elif bleu_score < 25:
            print("💡 Good BLEU score - BERT CMLM performing well for translation")
        else:
            print("💡 Excellent BLEU score - very good performance for BERT CMLM!")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main()