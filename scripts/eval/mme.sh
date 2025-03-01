#!/bin/bash

files=(
    "OCR"
    "artwork"
    "celebrity"
    "code_reasoning"
    "color"
    "commonsense_reasoning"
    "count"
    "existence"
    "landmark"
    "numerical_calculation"
    "position"
    "posters"
    "scene"
    "text_translation"
)

for file in "${files[@]}"
do
    echo "python ssr/eval/mme.py --task ${file}"
    python ssr/eval/mme.py --task "${file}"
    wait
done