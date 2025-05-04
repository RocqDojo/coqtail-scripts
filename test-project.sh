#!/bin/bash

export LLM_MODEL=/workspace/ssd1/model/april/Qwen2.5-7B-Instruct-finetuning
export RETRIVER_MODEL=/workspace/ssd1/model/retriver/paraphrase-multilingual-MiniLM-L12-v2
export VECTOR_DB=/workspace/ssd1/coqfinetuning/vectorDB/coq
export TORCH_DEVICE=cuda:0
export RESULT_SUFFIX=$(basename "$LLM_MODEL")

echo "testing on $(find $1 -name '*.v' | wc -l) files ..."

find $1 -name '*.v' | split -n l/4 --numeric-suffixes=0 --suffix-length=1 - test_tasks_

for i in {0..3}; do
    part_file="test_tasks_$i"
    TORCH_DEVICE="cuda:$i" python ./test-driver.py $(cat "$part_file") &
done


