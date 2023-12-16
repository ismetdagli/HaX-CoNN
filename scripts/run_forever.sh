#!/bin/bash

# Alexnet on DLA can run infinitely, which we then decided to comment out for clarity.
# # Path to the .plan file
# PLAN_FILE=$1

# # Directory to save the log files
# LOG_DIR=$2

# # Extract the basename of the plan file
# BASENAME=$(basename "$PLAN_FILE" .plan)

# # Ensure the log directory exists
# mkdir -p "$LOG_DIR"

# # Infinite loop
# while true; do
#     # Get current timestamp
#     timestamp=$(date +"%Y%m%d_%H%M%S")

#     # Define the log file path with basename and timestamp
#     log_file="$LOG_DIR/${BASENAME}_${timestamp}.log"

#     # Execute the command and save the output to the log file
#     /usr/src/tensorrt/bin/trtexec --iterations=100 --avgRuns=1 --warmUp=10000 --duration=0 --loadEngine="$PLAN_FILE" > "$log_file"
#     echo "Saved log to $log_file"

#     # Optional: sleep for a certain period if needed between executions
#     # sleep 1
# done

while true; do
    #The output is useless, write into temp.txt
    python3 src/z3_solver_multi_dnn.py >> temp.txt
done