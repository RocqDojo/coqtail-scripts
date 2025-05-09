#!/usr/bin/fish

set models \
  Qwen2.5-7B-Instruct-finetuning \
  Qwen2.5-7B-Instruct \
  Qwen2.5-7B-finetuning \
  Qwen2.5-7B \
  llama3.1-8B-instruct-finetuning \
  Meta-Llama-3.1-8B-Instruct \
  llama3.1-8B-finetuning \
  Meta-Llama-3.1-8B

set root ../coqfinetuning/test_data/sf-vol12/vol1-sol/

set -x MAX_ATTEMPTS 20

for model in $models
  echo -n $model :
  python count-pass.py (find $root -name "*.v.$model")
end

