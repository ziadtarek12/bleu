#!/usr/bin/env bash

set -x
set -e

# Use current directory structure
CURRENT_DIR=$(pwd)
RAW="$CURRENT_DIR/data"
TMP="$CURRENT_DIR/data/tmp"

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ="de-en.tgz"

src=de
tgt=en
lang=de-en

prep=$RAW/de-en
orig=$TMP

mkdir -p $orig $prep
cd $orig

if [ -f $GZ ]; then
    echo "$GZ already exists, skipping download"
else
    echo "Downloading data from ${URL}..."
    wget "$URL"
    if [ -f $GZ ]; then
        echo "Data successfully downloaded."
    else
        echo "Data not successfully downloaded."
        exit
    fi
    tar zxvf $GZ
fi
cd -

if [ -f $prep/test.en ] && [ -f $prep/test.de ]; then
    echo "IWSLT14 test dataset is already prepared, skip"
else
    echo "Extracting raw text from XML files..."
    for l in $src $tgt; do
        for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$prep/${fname%.*}
        echo "Processing $o -> $f"
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\'/\'/g" > $f
        echo ""
        done
    done

    echo "Creating test files from raw XML extracts..."
    for l in $src $tgt; do
        # Create test files by concatenating the official test sets
        cat $prep/IWSLT14.TED.dev2010.de-en.$l \
            $prep/IWSLT14.TEDX.dev2012.de-en.$l \
            $prep/IWSLT14.TED.tst2010.de-en.$l \
            $prep/IWSLT14.TED.tst2011.de-en.$l \
            $prep/IWSLT14.TED.tst2012.de-en.$l \
            > $prep/test.$l
    done
    
    echo "IWSLT14 test set ready."
fi
