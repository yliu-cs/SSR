#!/bin/bash

tasks=(
    "size"
    "positional"
    "reach"
    "existence"
    "counting"
)

for num_beams in 1 3 5 7 9
do
    for temperature in 0.2 0.5 0.8
    do
        for file in "${tasks[@]}"
        do
            echo "python ssr/eval/spatialbench.py --task ${file} --num_beams ${num_beams} --temperature ${temperature}"
            python ssr/eval/spatialbench.py --task "${file}" --num_beams ${num_beams} --temperature ${temperature}
            wait
        done
    done
done

for num_beams in 1 3 5 7 9
do
    for temperature in 0.2 0.5 0.8
    do
        for file in "${tasks[@]}"
        do
            echo "python ssr/eval/spatialbench.py --task ${file} --do_sample --num_beams ${num_beams} --temperature ${temperature}"
            python ssr/eval/spatialbench.py --task "${file}" --do_sample --num_beams ${num_beams} --temperature ${temperature}
            wait
        done
    done
done