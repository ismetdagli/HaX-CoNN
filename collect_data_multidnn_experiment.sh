#!/usr/bin/env bash

mkdir baseline_engine_logs

printf "\n\n Experiment 1: GoogleNet and ResNet data profiling\n"
#GPU Baseline experiment
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_gpu_only.plan --plan2=baseline_engines/resnet101_gpu_only.plan

#GPU&DLA baseline, we run both GPU/DLA to pick the best.
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_gpu_only.plan --plan2=baseline_engines/resnet101_dla_only.plan
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_dla_only.plan --plan2=baseline_engines/resnet101_gpu_only.plan

#Herald
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_dla_transition_at_10.plan --plan2=baseline_engines/resnet101_gpu_transition_at_4.plan

#H2H
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_dla_transition_at_39.plan --plan2=baseline_engines/resnet101_gpu_transition_at_101.plan

#HaX-CoNN
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/googlenet_gpu_transition_at_80.plan --plan2=baseline_engines/resnet101_dla_transition_at_3.plan



printf "\n\n\nExperiment 2: Inception and Resnet152  data profiling\n"
#GPU Baseline experiment
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/inception_gpu_only.plan --plan2=baseline_engines/resnet152_gpu_only.plan

#GPU&DLA baseline, we run both GPU/DLA to pick the best.
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/resnet152_gpu_only.plan --plan2=baseline_engines/inception_dla_only.plan
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/resnet152_dla_only.plan --plan2=baseline_engines/inception_gpu_only.plan

#Herald
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/inception_dla_transition_at_30.plan --plan2=baseline_engines/resnet152_gpu_transition_at_46.plan

#H2H
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/inception_dla_transition_at_95.plan --plan2=baseline_engines/resnet152_gpu_transition_at_101.plan

#HaX-CoNN
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/inception_gpu_transition_at_510.plan --plan2=baseline_engines/resnet152_dla_transition_at_636.plan




printf "\n\n\nExperiment 3:Alexnet Resnet101 data profiling\n"
#GPU Baseline experiment
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/alexnet_gpu_only.plan --plan2=baseline_engines/resnet101_gpu_only.plan

#GPU&DLA baseline, we run both GPU/DLA to pick the best.
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/alexnet_gpu_only.plan --plan2=baseline_engines/resnet101_dla_only.plan
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/alexnet_dla_only.plan --plan2=baseline_engines/resnet101_gpu_only.plan

#Herald
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/alexnet_gpu_transition_at_16.plan --plan2=baseline_engines/resnet101_dla_transition_at_58.plan

#H2H
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/alexnet_gpu_transition_at_14.plan --plan2=baseline_engines/resnet101_dla_transition_at_101.plan

#HaX-CoNN
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/alexnet_gpu_transition_at_16.plan --plan2=baseline_engines/resnet101_dla_transition_at_4.plan




printf "\n\n\nExperiment 4:VGG19 Resnet152 data profiling\n"
#GPU Baseline experiment
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/vgg19_gpu_only.plan --plan2=baseline_engines/resnet152_gpu_only.plan

#GPU&DLA baseline, we run both GPU/DLA to pick the best.
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/vgg19_gpu_only.plan --plan2=baseline_engines/resnet152_dla_only.plan
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/vgg19_dla_only.plan --plan2=baseline_engines/resnet152_gpu_only.plan

#Herald
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/vgg19_gpu_transition_at_9.plan --plan2=baseline_engines/resnet152_dla_transition_at_46.plan

#H2H
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/vgg19_gpu_transition_at_27.plan --plan2=baseline_engines/resnet152_dla_transition_at_286.plan

#HaX-CoNN
python3 run_multiple_dnn_experiments.py --plan1=baseline_engines/vgg19_dla_transition_at_9.plan --plan2=baseline_engines/resnet152_gpu_transition_at_165.plan


cd baseline_engine_logs 
grep -r " mean: " * | grep -v "end to end" > mean_results_of_executions.txt

cd ..

