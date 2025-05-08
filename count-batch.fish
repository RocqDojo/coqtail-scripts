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

for model in $models
  for k in (seq 1 10)
    set -x K $k
    echo -n $model :
    python count-correct.py (find $root -name "*.v.$model")
  end
end

