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

python3 ./src/build_engine.py --prototxt ./prototxt_input_files/googlenet.prototxt --start gpu --output ./google_only_gpu.plan
python3 ./src/build_engine.py --prototxt ./prototxt_input_files/googlenet.prototxt --start dla --output ./google_only_dla.plan

mkdir ./multi_dnn_execution_logs/

#Run the python code to check TensorRT binaries working fine.
python3 ./run_multiple_dnn.py

cat multi_dnn_execution_logs/google_only_dla_google_only_gpu_2_results.log | grep  mean |
 grep -v "end to end"

cat multi_dnn_execution_logs/google_only_gpu_google_only_dla_1_results.log | grep  mean |
 grep -v "end to end"