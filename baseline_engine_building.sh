#!/usr/bin/env bash

mkdir baseline_engines

echo "\n\nStep 1: Building baseline engines\n"

echo "Experiment 1: GoogleNet and ResNet101"
##GoogleNet and ResNet GPU & DLA
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
--output baseline_engines/googlenet_dla_only.plan \
--start dla \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_only.plan \
--start dla \
--transition -1 \
--verbose


#Herald PLAN 1
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output baseline_engines/googlenet_dla_transition_at_10.plan \
--start dla \
--transition 10 \
--verbose

#Herald PLAN 2
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_gpu_transition_at_4.plan \
--start gpu \
--transition -1 \
--verbose

#H2H PLAN 1
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output baseline_engines/googlenet_dla_transition_at_39.plan \
--start dla \
--transition 39 \
--verbose

#H2H PLAN 2
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_gpu_transition_at_101.plan \
--start gpu \
--transition -1 \
--verbose

#HaX-CoNN PLAN 1
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output baseline_engines/googlenet_gpu_transition_at_80.plan \
--start gpu \
--transition 80 \
--verbose

#HaX-CoNN PLAN 2
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output baseline_engines/resnet101_dla_transition_at_3.plan \
--start dla \
--transition 3 \
--verbose







echo "\n\n\n Experiment 2: Inception and Resnet152\n"
#Inception and Resnet152 GPU & DLA
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
--output baseline_engines/inception_dla_only.plan \
--start dla \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_dla_only.plan \
--start dla \
--transition -1 \
--verbose


#inception_30_resnet152_46_1_results.log:[11/23/2023-10:39:59]
#Herald
python3 src/build_engine.py \
--prototxt prototxt_input_files/inception.prototxt \
--output baseline_engines/inception_dla_transition_at_30.plan \
--start dla \
--transition 30 \
--verbose
#Herald
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_gpu_transition_at_46.plan \
--start gpu \
--transition 46 \
--verbose

# inception_95_resnet152_101_1_results.log:[11/23/2023-11:11:27]
#H2H
python3 src/build_engine.py \
--prototxt prototxt_input_files/inception.prototxt \
--output baseline_engines/inception_dla_transition_at_95.plan \
--start dla \
--transition 95 \
--verbose
#H2H
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_gpu_transition_at_101.plan \
--start gpu \
--transition 101 \
--verbose

#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/inception.prototxt \
--output baseline_engines/inception_gpu_transition_at_510.plan \
--start gpu \
--transition 510 \
--verbose
#HaX-CoNN
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_dla_transition_at_636.plan \
--start dla \
--transition 636 \
--verbose







echo "\n\n\nExperiment 3: Alexnet Resnet101\n"
#Alexnet Resnet101  GPU & DLA
python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/alexnet.prototxt \
--output baseline_engines/alexnet_dla_only.plan \
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








echo "\n\n\nExperiment 4: VGG-19 ResNet152 \n"
#VGG-19 ResNet152  GPU & DLA
python3 src/build_engine.py \
--prototxt prototxt_input_files/vgg19.prototxt \
--output baseline_engines/vgg19_gpu_only.plan \
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
--prototxt prototxt_input_files/vgg19.prototxt \
--output baseline_engines/vgg19_dla_only.plan \
--start dla \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_dla_only.plan \
--start dla \
--transition -1 \
--verbose


#Herald
python3 src/build_engine.py \
--prototxt prototxt_input_files/vgg19.prototxt \
--output baseline_engines/vgg19_gpu_transition_at_9.plan \
--start gpu \
--transition 9 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_dla_transition_at_46.plan \
--start dla \
--transition 46 \
--verbose

#H2H
#resnet152_286_vgg19_modified_gpu_27_2_results.log
python3 src/build_engine.py \
--prototxt prototxt_input_files/vgg19.prototxt \
--output baseline_engines/vgg19_gpu_transition_at_27.plan \
--start gpu \
--transition 27 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_dla_transition_at_286.plan \
--start dla \
--transition 286 \
--verbose

##HaX-CoNN
#vgg19_modified_dla_9_resnet152_165_1_results.log
python3 src/build_engine.py \
--prototxt prototxt_input_files/vgg19.prototxt \
--output baseline_engines/vgg19_dla_transition_at_9.plan \
--start dla \
--transition 9 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_gpu_transition_at_165.plan \
--start gpu \
--transition 165 \
--verbose
