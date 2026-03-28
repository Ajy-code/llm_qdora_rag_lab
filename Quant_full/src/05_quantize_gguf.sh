#!/bin/bash
set -e
#Quantization du modèle convertit ( au format gguf )

GGUF_FILE="../models/gguf_ref/Ministral-3B-BF16.gguf"
QUANT_DIR="../models/gguf_quantized"

# Vérification de l'existence du modèle au format gguf
if [ ! -f "$GGUF_FILE" ]; then
    echo "Erreur : Le fichier source $GGUF_FILE est introuvable."
    exit 1
fi

mkdir -p $QUANT_DIR

cd llama.cpp
#Création de l'executable en c++ qui va quantizé le fichier gguf
if [ ! -f "build/bin/llama-quantize" ]; then
    cmake -B build
    cmake --build build --config Release -j --target llama-quantize
fi
cd ..

#Déclaration des types de quantization
TYPES=("Q5_K_M" "Q4_K_M")

for TYPE in "${TYPES[@]}"; do
    echo "Traitement du type $TYPE"
    ./llama.cpp/build/bin/llama-quantize $GGUF_FILE $QUANT_DIR/Ministral-3B-$TYPE.gguf $TYPE
done
