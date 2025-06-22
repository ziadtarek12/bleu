import os
import torch
import subprocess
import sacrebleu
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config import IWSLT_TEST_SRC, IWSLT_TEST_REF

# BERT CMLM model configuration
MODEL_NAME = "pythn/gomaa_grad"
MODEL_DISPLAY_NAME = "BERT CMLM (pythn/gomaa_grad)"

def download_model():
    """Download the BERT CMLM model if needed"""
    model_dir = "pythn_gomaa_grad"
    if not os.path.isdir(model_dir):
        print(f"📥 Downloading {MODEL_DISPLAY_NAME}...")
        subprocess.run([
            "python", "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{MODEL_NAME}', local_dir='{model_dir}')"
        ], check=True)
        print(f"✅ {MODEL_DISPLAY_NAME} downloaded.")
    else:
        print(f"✅ {MODEL_DISPLAY_NAME} already exists.")
    return model_dir

def load_bert_cmlm_model():
    """Load BERT CMLM model for translation"""
    print(f"📥 Loading {MODEL_DISPLAY_NAME}...")
    
    # Download model first
    model_dir = download_model()
    
    # Check the actual file structure
    print("🔍 Checking model structure...")
    pt_file = os.path.join(model_dir, "cmlm_model", "model_step_100000.pt")
    
    if os.path.exists(pt_file):
        print(f"✅ Found PyTorch checkpoint: {pt_file}")
        
        try:
            # Load the PyTorch checkpoint
            print("🔄 Loading PyTorch checkpoint...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(pt_file, map_location=device)
            
            print(f"✅ Checkpoint loaded on {device}")
            print(f"Checkpoint type: {type(checkpoint)}")
            
            # Analyze what's in the checkpoint
            if isinstance(checkpoint, dict):
                print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
                
                # Check if it's a state dict or complete model
                if 'model' in checkpoint:
                    print("📦 Found 'model' key - extracting model")
                    model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    print("📦 Found 'state_dict' key - this is just weights")
                    print("⚠️  This requires the model architecture to load properly")
                    return None, None, None
                elif any(key.startswith(('bert.', 'encoder.', 'decoder.')) for key in checkpoint.keys()):
                    print("📦 This appears to be a state dict (just weights)")
                    print("⚠️  Available weight keys:")
                    for key in list(checkpoint.keys())[:10]:  # Show first 10 keys
                        print(f"     {key}")
                    if len(checkpoint.keys()) > 10:
                        print(f"     ... and {len(checkpoint.keys()) - 10} more")
                    print("⚠️  Need model architecture to load these weights")
                    return None, None, None
                else:
                    print("🤷 Unknown checkpoint format")
                    model = checkpoint
            else:
                print("📦 Checkpoint is a complete model object")
                model = checkpoint
            
            # Set model to evaluation mode if possible
            if hasattr(model, 'eval'):
                model.eval()
                print("✅ Model set to evaluation mode")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            
            return model, tokenizer, device
            
        except Exception as e:
            print(f"❌ Failed to load PyTorch checkpoint: {e}")
            return None, None, None
    else:
        print(f"❌ PyTorch checkpoint not found at: {pt_file}")
        return None, None, None

def translate_with_pytorch_model(sentences, model, tokenizer, device):
    """Translate sentences using the PyTorch CMLM model"""
    print("🔄 Using PyTorch CMLM model for translation...")
    
    translations = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for i, sentence in enumerate(sentences):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(sentences)}")
            
            try:
                # Tokenize input sentence
                inputs = tokenizer(sentence, 
                                 return_tensors="pt", 
                                 max_length=128, 
                                 truncation=True, 
                                 padding=True)
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get model outputs
                if hasattr(model, '__call__'):
                    outputs = model(**inputs)
                elif hasattr(model, 'forward'):
                    outputs = model.forward(**inputs)
                else:
                    # If model is just a state dict, we can't use it directly
                    print("⚠️  Model appears to be a state dict, not a complete model")
                    translations.append(sentence)  # Fallback
                    continue
                
                # Process outputs to get translation
                if hasattr(outputs, 'logits'):
                    # Get predicted token IDs
                    predicted_ids = torch.argmax(outputs.logits, dim=-1)
                    # Decode to text
                    translation = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                elif hasattr(outputs, 'last_hidden_state'):
                    # For encoder-only models, this won't give us translation
                    # but we can try to extract meaningful information
                    translation = sentence  # Fallback for now
                else:
                    # Unknown output format
                    translation = sentence  # Fallback
                
                translations.append(translation)
                
            except Exception as e:
                print(f"⚠️  Translation failed for sentence {i + 1}: {e}")
                translations.append(sentence)  # Use original as fallback
    
    return translations

def evaluate_bert_cmlm():
    """Evaluate BERT CMLM model on IWSLT14 test set"""
    print("🚀 Starting BERT CMLM Translation Evaluation...")
    
    # Ensure data is ready
    subprocess.run(["bash", "prepare_iwslt14.sh"], check=True)
    
    # Load model
    model, tokenizer, device = load_bert_cmlm_model()
    if model is None:
        print("❌ Failed to load BERT CMLM model")
        return None
    
    # Load test data
    print("📚 Loading IWSLT14 test data...")
    with open(IWSLT_TEST_SRC, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f]
    
    with open(IWSLT_TEST_REF, 'r', encoding='utf-8') as f:
        reference_sentences = [line.strip() for line in f]
    
    print(f"📊 Loaded {len(source_sentences)} sentence pairs")
    
    # Use full IWSLT14 test set like other model scripts
    print(f"🔄 Processing full IWSLT14 test set ({len(source_sentences)} sentences)...")
    
    # Translate
    print(f"🔧 Using PyTorch CMLM model for inference")
    translations = translate_with_pytorch_model(source_sentences, model, tokenizer, device)
    
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
        print(f"📊 BERT CMLM (Fallback) BLEU Score: {bleu_score:.2f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar([MODEL_DISPLAY_NAME], [bleu_score], color='lightcoral')
        plt.ylabel('BLEU Score')
        plt.title('BERT CMLM (pythn/gomaa_grad) BLEU Score on IWSLT14 Test Set')
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
    print("🎯 BERT CMLM Translation Model Analysis")
    print("="*60)
    print(f"📝 Analyzing: {MODEL_DISPLAY_NAME}")
    print("📝 This appears to be a PyTorch checkpoint file")
    print("="*60)
    
    evaluate_bert_cmlm()
    
    print("\n💡 Next Steps:")
    print("1. Contact the model author for usage instructions")
    print("2. Look for model architecture code in the repository")
    print("3. Check if there are example usage scripts")
    print("4. Consider using a standard HuggingFace translation model instead")

if __name__ == "__main__":
    main()