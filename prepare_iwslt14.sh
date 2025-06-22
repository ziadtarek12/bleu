#!/bin/bash
set -e

DATA_DIR="data/de-en"
TEST_DE="$DATA_DIR/test.de"
TEST_EN="$DATA_DIR/test.en"
URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
TGZ="de-en.tgz"

if [ -f "$TEST_DE" ] && [ -f "$TEST_EN" ]; then
    echo "IWSLT14 test set already present."
    exit 0
fi

echo "Downloading and extracting IWSLT14 de-en test set..."
mkdir -p "$DATA_DIR"
wget -O "$TGZ" "$URL"
tar -xzf "$TGZ" -C data
rm "$TGZ"

# Move test files to the right place if needed
for l in de en; do
    if [ ! -f "$DATA_DIR/test.$l" ]; then
        found=$(find data/de-en -name "test.$l" | head -n 1)
        if [ -n "$found" ]; then
            mv "$found" "$DATA_DIR/test.$l"
        fi
    fi

done

echo "IWSLT14 test set ready."
