#!/bin/bash

#Création de la pipeline Imatrix, méthode de quantification dynamique ( on quntifie fort les poids fortement sollicité et moins fort ceux qui sont peu sollicité)

GGUF_FILE="../models/gguf_ref/Ministral-3B-BF16.gguf"
QUANT_DIR="../models/gguf_quantized"
DAT_DIR="../models/imatrix"
CORPUS_TEXT="../prompts/calibration_chunks.txt"

#On vérifie l'existence de GGUF_FILE
if [ ! -f $GGUF_FILE ]; then
    echo "Le fichier source $GGUF_FILE est introuvable"
    exit 1
fi
#Création de QUANT_DIR et/ou DAT_DIR s'il n'existe pas
mkdir -p $QUANT_DIR
mkdir -p $DAT_DIR

#Vérification que llama-quantize est bien présent sinon on le télécharge
cd llama.cpp
if [ ! -f "build/bin/llama-quantize" ]; then
    cmake -B build
    cmake --build build --config Release -j --target llama-quantize
fi

#Vérification que llama-imatrix est bien présent sinon on le télécharge
if [ ! -f "build/bin/llama-imatrix" ]; then
    cmake -B build
    cmake --build build --config Release -j --target llama-imatrix
fi
cd ..

echo "Création du fichier imatrix.dat"
./llama.cpp/build/bin/llama-imatrix -m $GGUF_FILE -f $CORPUS_TEXT -o $DAT_DIR/imatrix.dat -c 512 -ngl 12 2>&1 | pv -l -s 100 > /dev/null
echo "imatrix.dat crée avec succée"

echo "Quantification dynamique du modèle en Q4_K_M avec la méthode imatrix"
./llama.cpp/build/bin/llama-quantize --imatrix $DAT_DIR/imatrix.dat $GGUF_FILE $QUANT_DIR/Ministral-3B-Q4_K_M-IMATRIX.gguf Q4_K_M

echo "Quantification du modèle sans la méthode imatrix en Q4_K_M"
./llama.cpp/build/bin/llama-quantize  $GGUF_FILE $QUANT_DIR/Ministral-3B-Q4_K_M.gguf Q4_K_M
