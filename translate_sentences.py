import subprocess
import tempfile
import os
from config import MODEL_PATHS

def translate_sentences(sentences, model_path, gpu=0):
    # Write sentences to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as src_file:
        for s in sentences:
            src_file.write(s.strip() + '\n')
        src_file_path = src_file.name

    # Prepare output file
    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as out_file:
        out_file_path = out_file.name

    # Build the onmt_translate command
    cmd = [
        "onmt_translate",
        "-model", model_path,
        "-src", src_file_path,
        "-output", out_file_path,
        "-replace_unk",
        "-verbose",
        "-gpu", str(gpu)
    ]

    print("Running translation...")
    subprocess.run(cmd, check=True)

    # Read and print translations
    with open(out_file_path, "r", encoding="utf-8") as f:
        translations = [line.strip() for line in f]

    # Clean up temp files
    os.remove(src_file_path)
    os.remove(out_file_path)

    return translations

if __name__ == "__main__":
    # Example sentences to translate (German to English)
    sentences = [
        "Hallo, wie geht es dir?",
        "Das ist ein Test.",
        "Vielen Dank für Ihre Hilfe."
    ]
    # Use the first model in your MODEL_PATHS
    model_path = MODEL_PATHS[0]
    translations = translate_sentences(sentences, model_path)
    print("\nTranslations:")
    for src, tgt in zip(sentences, translations):
        print(f"SRC: {src}\nTGT: {tgt}\n")
