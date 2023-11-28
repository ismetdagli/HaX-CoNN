#!/bin/bash
# $1 -> plan dir
# $2 -> log dir

# Ensure the log directory exists
mkdir -p "$2"

# Loop over all .plan files in the PLAN_DIR
for plan_file in "$1"/*.plan; do
    # Extract the base name without the extension
    base_name=$(basename "$plan_file" .plan)

    # Define the output log file path
    log_file="$2/${base_name}.log"

    # Execute the command with the current plan file and direct output to the corresponding log file
    /usr/src/tensorrt/bin/trtexec --iterations=100 --avgRuns=1 --warmUp=10000 --duration=0 --loadEngine="$plan_file" > "$log_file"
    echo "Saved log to $log_file"
done

