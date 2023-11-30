#!/usr/bin/env bash

mkdir baseline_singlednn_engine_logs

printf "\n\n Experiment 1: GoogleNet and ResNet data profiling\n"
#GPU Baseline experiment
python3 run_single_dnn_experiments.py --plan1=baseline_engines/resnet152_gpu_only1.plan --plan2=baseline_engines/resnet152_gpu_only.plan

#GPU&DLA baseline, we run both GPU/DLA to pick the best.
python3 run_single_dnn_experiments.py --plan1=baseline_engines/resnet152_gpu_only.plan --plan2=baseline_engines/resnet152_dla_only.plan

#Mensa
python3 run_single_dnn_experiments.py --plan1=baseline_engines/resnet152_dla_transition_at_165.plan --plan2=baseline_engines/resnet152_gpu_transition_at_364.plan

#HaX-CoNN
python3 run_single_dnn_experiments.py --plan1=baseline_engines/resnet152_dla_transition_at_636.plan --plan2=baseline_engines/resnet152_gpu_transition_at_165.plan


cd baseline_singlednn_engine_logs 
grep -r " mean: " * | grep -v "end to end" > mean_results_of_executions.txt

cd ..
printf '\n\n\n'

printf "Final Step: Summary of experiments\n"


