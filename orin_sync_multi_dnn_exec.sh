cp -r /usr/src/tensorrt/ ./tensorrt_sharedMem1/
cp -r /usr/src/tensorrt/ ./tensorrt_sharedMem2/

cp ./modified_tensorrts/sampleInference1_orinagx.cpp ./tensorrt_sharedMem1/samples/common/sampleInference.cpp
cp ./modified_tensorrts/sampleInference2_orinagx.cpp ./tensorrt_sharedMem2/samples/common/sampleInference.cpp

cd ./tensorrt_sharedMem1/samples/trtexec/
make -j4
cd ../../../

cd ./tensorrt_sharedMem2/samples/trtexec/
make -j4
cd ../../../


mkdir baseline_engines
python3 orin_build_engine.py
chmod +x orin_collect_data_multidnn_experiment.sh
./orin_collect_data_multidnn_experiment.sh
python3 orin_summarize_multi_dnn_executions.py