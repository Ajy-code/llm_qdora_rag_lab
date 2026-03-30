#!/bin/bash

MODEL_REF="../models/gguf_ref/Ministral-3B-BF16.gguf"
MODEL_QUANT="../models/gguf_quantized/Ministral-3B-Q4_K_M.gguf"
MODEL_IMATRIX="../models/gguf_quantized/Ministral-3B-Q4_K_M-IMATRIX.gguf"
BENCH_FILE="../outputs/test/bench_results_perplexity.csv"
CORPUS_TEXT="../prompts/corpus_text_ai_act.txt"

echo "model,perplexity" > $BENCH_FILE

cd llama.cpp
if [ ! -f "build/bin/llama-perplexity" ]; then
    echo "Construction de llama-perplexity"
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j 4 --target llama-perplexity
fi
cd ..

for model in $MODEL_REF $MODEL_QUANT $MODEL_IMATRIX; do
    pplx=$(./llama.cpp/build/bin/llama-perplexity -m "$model" -f "$CORPUS_TEXT" -ngl 32 2>&1 | grep "Final estimate: PPL =" | awk '{print $5}')
    echo "$(basename $model),$pplx" >> $BENCH_FILE
done