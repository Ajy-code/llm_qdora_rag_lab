#!/bin/bash

#Conversion du modèle du format HF au format GGUF
MODEL_ID="mistralai/Ministral-3-3B-Instruct-2512-BF16"
HF_DIR="../models/hf/ministral-3b"
GGUF_OUT="../models/gguf_ref/Ministral-3B-BF16.gguf"

echo "Conversion du modèle stocké dans $HF_DIR"

#Crée les dossiers s'ils ne sont pas crée
mkdir -p $HF_DIR
mkdir -p ../models/gguf_ref
mkdir -p ../outputs/gguf

#Clone le dossier llama.cp s'il n'est pas déja cloné (présent)
if [ ! -d "llama.cpp" ]; then
    echo "Clonage de Llama.cpp (depuis github)"
    git clone https://github.com/ggerganov/llama.cpp.git
    pip install -r llama.cpp/requirements.txt
fi

#Téléchargement du modèle au format HF, on le met dans HF_DIR
#Le premier --local-dir est là pour indiquer où faire le téléchargement, --local-dir-use-symlinks False est là pour forcer le programme à ne pas juste faire un raccourcis,
#il faut que les fichier soit physiquement en HF_DIR
huggingface-cli download $MODEL_ID --local-dir $HF_DIR --local-dir-use-symlinks False
#Conversion
python llama.cpp/convert_hf_to_gguf.py $HF_DIR --outtype bf16 --outfile $GGUF_OUT 2>&1 | tee ../outputs/gguf/conversion.log
