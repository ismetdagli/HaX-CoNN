#!/usr/bin/env bash

mkdir orin_baseline_engines_logs

printf "\n\nExperiment 1: GoogleNet and ResNet101 data profiling\n"
#GPU Baseline experiment
python3 orin_run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_only_gpu.plan --plan2=baseline_engines/resnet101_only_gpu.plan
printf '\n'
#GPU and DLA 
python3 orin_run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_only_gpu.plan --plan2=baseline_engines/resnet101_only_dla.plan
printf '\n'
#Herald
python3 orin_run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_dla_transition_at_24.plan --plan2=baseline_engines/resnet101_gpu_transition_at_101.plan
printf '\n'
#H2H
python3 orin_run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_dla_transition_at_95.plan --plan2=baseline_engines/resnet101_gpu_transition_at_415.plan
printf '\n'
#HaX-CoNN
python3 orin_run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_dla_transition_at_38.plan --plan2=baseline_engines/resnet101_gpu_transition_at_312.plan
printf '\n'

cd orin_baseline_engines_logs 
grep -r "                     Total " * > orin_mean_results_of_executions.txt

cd ..
printf '\n\n\n'

