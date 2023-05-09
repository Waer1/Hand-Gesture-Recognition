#!/bin/bash

# Loop through feature and model values
for f in $(seq 0 3); do
    for m in $(seq 0 3); do
        # Run Python script with feature and model parameters
        python FinalCode/main.py --feature "$f" --model "$m" > "./outputs/output_${f}_${m}.txt"
        echo "Finished feature $f and model $m"
    done
done
