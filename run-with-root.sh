#!/bin/bash

echo "working on $(find $1 -name '*.v' | wc -l) files ..."

echo "wait 3 seconds before starting"
sleep 3

find $1 -name '*.v' | xargs -P 20 -I {} python runner.py {} | tee "$(date -Iminutes).log"


