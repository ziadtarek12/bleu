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
            checkpoint = torch.load(pt_file, map_location='cpu')
            print(f"✅ Checkpoint loaded. Keys: {list(checkpoint.keys())}")
            
            # Try to load with base BERT tokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            
            # For now, we'll use a simple approach
            return checkpoint, tokenizer, "pytorch_checkpoint"
            
        except Exception as e:
            print(f"❌ Failed to load PyTorch checkpoint: {e}")
            return None, None, None
    else:
        print(f"❌ PyTorch checkpoint not found at: {pt_file}")
        print("📂 Available files:")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                print(f"   {os.path.join(root, file)}")
        return None, None, None

def translate_with_checkpoint(sentences, checkpoint, tokenizer):
    """Translate sentences using the PyTorch checkpoint"""
    print("⚠️  Note: This is a PyTorch checkpoint, not a standard HF model.")
    print("⚠️  Translation capability depends on the specific model architecture.")
    
    translations = []
    
    # Since we can't easily use the checkpoint without knowing the exact architecture,
    # we'll implement a simple fallback approach
    print("🔄 Using fallback translation approach...")
    
    for i, sentence in enumerate(sentences):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{len(sentences)}")
        
        try:
            # For now, we'll use a simple tokenization approach
            # In a real implementation, you'd need to know the exact model architecture
            # and how it expects inputs for translation
            
            # Tokenize the input
            inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)
            
            # Since we don't have the model architecture, we'll use a placeholder
            # In practice, you'd need to:
            # 1. Load the model architecture that matches the checkpoint
            # 2. Apply the checkpoint weights
            # 3. Run inference
            
            # For demonstration, we'll just return the input (this won't be a real translation)
            translations.append(f"[Checkpoint-based translation of: {sentence[:50]}...]")
            
        except Exception as e:
            print(f"⚠️  Processing failed for sentence {i + 1}: {e}")
            translations.append("")
    
    return translations

def evaluate_bert_cmlm():
    """Evaluate BERT CMLM model on IWSLT14 test set"""
    print("🚀 Starting BERT CMLM Translation Evaluation...")
    
    # Ensure data is ready
    subprocess.run(["bash", "prepare_iwslt14.sh"], check=True)
    
    # Load model
    checkpoint, tokenizer, method = load_bert_cmlm_model()
    if checkpoint is None:
        print("❌ Failed to load BERT CMLM model")
        return None
    
    # Load test data
    print("📚 Loading IWSLT14 test data...")
    with open(IWSLT_TEST_SRC, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f]
    
    with open(IWSLT_TEST_REF, 'r', encoding='utf-8') as f:
        reference_sentences = [line.strip() for line in f]
    
    print(f"📊 Loaded {len(source_sentences)} sentence pairs")
    
    # Limit to first 100 sentences for this demo
    print("⚡ Using first 100 sentences for demonstration...")
    source_sentences = source_sentences[:100]
    reference_sentences = reference_sentences[:100]
    
    # Translate
    print(f"🔧 Using method: {method}")
    translations = translate_with_checkpoint(source_sentences, checkpoint, tokenizer)
    
    # Save translations
    output_file = f"bert_cmlm_gomaa_grad_checkpoint_translations.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for translation in translations:
            f.write(translation + "\n")
    print(f"💾 Translations saved to: {output_file}")
    
    # Show some examples
    print("\n📝 Sample outputs:")
    for i in range(min(5, len(source_sentences))):
        print(f"DE: {source_sentences[i]}")
        print(f"Output: {translations[i]}")
        print(f"REF: {reference_sentences[i]}")
        print("-" * 50)
    
    print("\n⚠️  Important Note:")
    print("This model is a PyTorch checkpoint that requires specific")
    print("architecture code to load and use properly for translation.")
    print("For actual translation, you would need:")
    print("1. The model architecture definition")
    print("2. Code to load the checkpoint into the architecture")
    print("3. Translation inference code")
    
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