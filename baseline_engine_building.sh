#!/usr/bin/env bash

mkdir baseline_engines

echo "\n\nStep 1: Building baseline engines\n"

echo "Experiment 1: GoogleNet and ResNet"
#GoogleNet and ResNet
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output baseline_engines/googlenet_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output starter_guide_logs/googlenet_dla_only.plan \
--start dla \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_only.plan \
--start dla \
--transition -1 \
--verbose


#Herald
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output starter_guide_logs/googlenet_dla_transition_at_10.plan \
--start dla \
--transition 10 \
--verbose

#Herald
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_gpu_transition_at_4.plan \
--start gpu \
--transition -1 \
--verbose

#H2H
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output starter_guide_logs/googlenet_dla_transition_at_39.plan \
--start dla \
--transition 39 \
--verbose

#H2H
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_gpu_transition_at_101.plan \
--start gpu \
--transition -1 \
--verbose

#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output starter_guide_logs/googlenet_gpu_transition_at_80.plan \
--start gpu \
--transition 80 \
--verbose

#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_transition_at_3.plan \
--start dla \
--transition 3 \
--verbose







echo "\n\n\n Experiment 2: Inception and Resnet152\n"
#Inception and Resnet152
python3 src/build_engine.py \
--prototxt prototxt_input_files/inception.prototxt \
--output baseline_engines/inception_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/inception.prototxt \
--output starter_guide_logs/inception_dla_only.plan \
--start dla \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_dla_only.plan \
--start dla \
--transition -1 \
--verbose











echo "\n\n\nExperiment 3: Alexnet Resnet101\n"
#Alexnet Resnet101
python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output starter_guide_logs/alexnet_dla_only.plan \
--start dla \
--transition -1 \
--verbose

#Note: Resnet101 gpu/dla plans previous built.

#Herald
python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_gpu_transition_at_16.plan \
--start gpu \
--transition 9 \
--verbose
#Herald
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_transition_at_58.plan \
--start dla \
--transition 58 \
--verbose

#H2H
python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_gpu_transition_at_14.plan \
--start gpu \
--transition 14 \
--verbose
#H2H
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_transition_at_101.plan \
--start dla \
--transition 101 \
--verbose

#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_gpu_transition_at_16.plan \
--start gpu \
--transition 16 \
--verbose
#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_transition_at_4.plan \
--start dla \
--transition 4 \
--verbose

#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_gpu_transition_at_16.plan \
--start gpu \
--transition 16 \
--verbose
#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_transition_at_4.plan \
--start dla \
--transition 4 \
--verbose




echo "\n\n\nExperiment 4: VGG-19 ResNet152 \n"
#VGG-19 ResNet152
python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output starter_guide_logs/alexnet_dla_only.plan \
--start dla \
--transition -1 \
--verbose

