#!/bin/bash

# brief: run single step tactic hint pass rate test on all coq source code files in a directory
# usage: bash extract-many.sh /opt/coqgym/coq_projects
#
# Must first run proof steps extraction before testing
# The following bash script start 8 test process on 8 CUDA devices. Adjust the code if the machine has less/more than 8 GPUs available.
# The tasks assigned to GPU_i are stored in test_tasks_i


export BASE_MODEL=/workspace/ssd1/model/base/Meta-Llama-3.1-8B-Instruct
export LORA_MODEL=/workspace/ssd1/model/new/Meta-Llama-3.1-8B-Instruct-noRAG-true-lora
export RETRIVER_MODEL=/workspace/ssd1/model/retriver/paraphrase-multilingual-MiniLM-L12-v2
export VECTOR_DB=/workspace/ssd1/coqfinetuning/vectorDB/curated-prelude/
export TORCH_DEVICE=cuda:0
export RESULT_SUFFIX=$(basename "$LORA_MODEL")-redoSFT-RAG

echo "testing on $(find $1 -name '*.v' | wc -l) files ..."

find $1 -name '*.v' | shuf > v-tasks
split -n l/8 --numeric-suffixes=0 --suffix-length=2 v-tasks test_tasks_

for i in {0..7}; do
    # zero‑pad i, i+8 and i+16 to two digits
    idx0=$(printf "%02d" $i)

    # run the three shards on GPU $i
    TORCH_DEVICE="cuda:$i" python run-with-coqlib.py tester.py $(cat "test_tasks_$idx0") &
done
