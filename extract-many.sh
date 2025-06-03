#!/bin/bash

# brief: run proof steps extraction on all coq source code files in a directory
# usage: bash extract-many.sh /opt/coqgym/coq_projects

echo "working on $(find $1 -name '*.v' | wc -l) files ..."

echo "wait 3 seconds before starting"
sleep 3

# run in parallel, with a limit of 20 tasks at one moment
find $1 -name '*.v' | xargs -P 20 -I {} python run-with-coqlib.py extract.py {} | tee "extract-$(date -Iminutes).log"


