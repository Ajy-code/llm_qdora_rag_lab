#!/bin/bash

MODEL_FILE="../models/gguf_quantized/Ministral-3B-Q4_K_M-IMATRIX.gguf"
BENCH_FILE="../outputs/test/bench_results.csv"

echo "n_gpu_layers,pp_speed,tg_speed" > $BENCH_FILE

cd llama.cpp
if [ ! -f "build/bin/llama-bench" ]; then
    echo "Construction de llama-bench"
    cmake -B build
    cmake --build build --config Release -j --target llama-bench
fi
cd ..


for k in 0 8 16 24 32; do
    PP_SPEED=$(./llama.cpp/build/bin/llama-bench -m $MODEL_FILE -p 512 -n 0 -ngl $k | grep "pp 512" | awk -F '|' '{print $7}' )
    TG_SPEED=$(./llama.cpp/build/bin/llama-bench -m $MODEL_FILE -p 0 -n 128 -ngl $k | grep "tg 128" | awk -F '|' '{print $7}' )
    echo "$k,$PP_SPEED,$TG_SPEED" >> $BENCH_FILE
done
