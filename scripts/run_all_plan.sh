#!/bin/bash
# $1 -> plan dir
# $2 -> log dir

# Ensure the log directory exists
# mkdir -p "$2"

# Loop over all .plan files in the PLAN_DIR
for plan_file in "$1"/*.plan; do
    # Extract the base name without the extension
    base_name=$(basename "$plan_file" .plan)

    # # Define the output log file path
    # log_file="$2/${base_name}.log"

    # Execute the command with the current plan file and direct output to the corresponding
    python3 run_multiple_dnn_experiments.py --plan1=build/overhead_dla/alexnet_dla.plan --plan2="$plan_file"
    # exit

done

