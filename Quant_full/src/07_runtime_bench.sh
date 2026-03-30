#!/bin/bash

#Test de l'influence de la variation du paramètre n_gpu_layers lors de l'inférence
#Il faudra faire de même avec n_ctx, n-batch, n_ubatch et calculé précisement la place prise dans la VRAM en fonction de la valeur du paramètre
#Mesure des performance du modèle avec llama-bench

MODEL_FILE="../models/gguf_quantized/Ministral-3B-Q4_K_M-IMATRIX.gguf"
BENCH_FILE="../outputs/test/bench_results.csv"

echo "n_gpu_layers,pp_speed,tg_speed" > $BENCH_FILE

cd llama.cpp
if [ ! -f "build/bin/llama-bench" ]; then
    echo "Construction de llama-bench"
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j 4 --target llama-bench
fi
cd ..

#PP_SPEDD: Vitesse de de lecture du prompt
#TG_SPEDD: Vitesse de géneration de token/s en phase decoding

for k in 0 8 16 24 32 ; do
    PP_SPEED=$(./llama.cpp/build/bin/llama-bench -m $MODEL_FILE -p 512 -n 0 -ngl $k | grep "pp512" | awk -F '|' '{print $8}' )
    TG_SPEED=$(./llama.cpp/build/bin/llama-bench -m $MODEL_FILE -p 0 -n 128 -ngl $k | grep "tg128" | awk -F '|' '{print $8}' )
    echo "$k,$PP_SPEED,$TG_SPEED" >> $BENCH_FILE
done
