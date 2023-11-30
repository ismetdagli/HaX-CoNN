#!/usr/bin/env bash

mkdir starter_guide_logs


echo "Step 1: Building engines"

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output starter_guide_logs/resnet101_dla_transition_at_3.plan \
--start dla \
--transition 3 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output starter_guide_logs/googlenet_gpu_transition_at_80.plan \
--start gpu \
--transition 80 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output starter_guide_logs/googlenet_only_gpu.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output starter_guide_logs/googlenet_only_dla.plan \
--start dla \
--transition 999 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output starter_guide_logs/resnet101_only_gpu.plan \
--start gpu \
--transition -1 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/resnet101.prototxt \
--output starter_guide_logs/resnet101_only_dla.plan \
--start dla \
--transition 999 \
--verbose


printf '\n\n\n'
echo "Step 2: creating TensorRT binaries"

cp -r /usr/src/tensorrt/ ./tensorrt_sharedMem1/
cp -r /usr/src/tensorrt/ ./tensorrt_sharedMem2/

cp ./modified_tensorrts/sampleInference1.cpp ./tensorrt_sharedMem1/samples/common/sampleInference.cpp
cp ./modified_tensorrts/sampleInference2.cpp ./tensorrt_sharedMem2/samples/common/sampleInference.cpp

cd ./tensorrt_sharedMem1/samples/trtexec/
make -j4
cd ../../../

cd ./tensorrt_sharedMem2/samples/trtexec/
make -j4
cd ../../../

# mkdir temp


printf '\n\n\n'
echo "Step 3: Executions start"
python3 starter_guide_experiment.py
cd starter_guide_logs
grep -r " mean: " * | grep -v "end to end" >> mean_results_of_executions.txt
cd ..

printf '\n\n\n'
echo "Final Step: Summary of experiments"
python3 src/summarize_starter_guide_executions.py

cd ..

