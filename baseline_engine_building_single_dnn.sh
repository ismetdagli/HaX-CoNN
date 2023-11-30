#!/usr/bin/env bash

mkdir baseline_engines

echo "\n\nStep 1: Building baseline engines\n"

echo "Experiment 1: ResNet152 Single DNN experiment"

#VGG-19 ResNet152  GPU & DLA
# python3 src/build_engine.py \
# --prototxt prototxt_input_files/resnet152.prototxt \
# --output baseline_engines/resnet152_gpu_only.plan \
# --start gpu \
# --transition -1 \
# --verbose

# #copied to use as a second engine for GPU use.
# cp baseline_engines/resnet152_gpu_only.plan baseline_engines/resnet152_gpu_only1.plan

# python3 src/build_engine.py \
# --prototxt prototxt_input_files/resnet152.prototxt \
# --output baseline_engines/resnet152_dla_only.plan \
# --start dla \
# --transition -1 \
# --verbose

# #resnet152_165_resnet152_364
# #Mensa
# python3 src/build_engine.py \
# --prototxt prototxt_input_files/resnet152.prototxt \
# --output baseline_engines/resnet152_dla_transition_at_165.plan \
# --start dla \
# --transition 165 \
# --verbose

# python3 src/build_engine.py \
# --prototxt prototxt_input_files/resnet152.prototxt \
# --output baseline_engines/resnet152_gpu_transition_at_364.plan \
# --start gpu \
# --transition 364 \
# --verbose



##HaX-CoNN
#resnet152_636_resnet152_165
python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_dla_transition_at_636.plan \
--start dla \
--transition 635 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet152.prototxt \
--output baseline_engines/resnet152_gpu_transition_at_165.plan \
--start gpu \
--transition 165 \
--verbose



