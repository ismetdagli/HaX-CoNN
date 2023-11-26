# HaX-CoNN Artifact
This is artifact of HaX-Conn: Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips. This 

Artifact described here includes the source code for HaX-CoNN GPU and DLA runtimes and the sources for the applications used in our evaluation.

## Description

1. Checklist(meta information)
* Hardware: Jetson Xavier AGX 32 GB
* Software easy installation: Jetpack 4.5.1
* Software details of Jetpack 4.5.1 includes: #TODO_ISMET


# Experimental Setup

First and foremost, this is a empirical study. We are open sourcing all the details how we collected data. The data collected through profiling has been encoded to script.  

## Layer profiling: 
This creates a text file of a DNN. The line after " [I] GPU Compute" are our target data. We use mean data as the average of X number of iterations   #TODO_ISMET
```bash
python3 collect_data_single_layers.py
```
Note: `+` sign demonstrates the layers are merged. `||` demonstrates outputs of the layers will be concataned (as concatanation layer). `{}` demonstrates that DLA fuses the layers and profiling of all layers are treated as one layer(basically, this is a profiling limitation in DLA architectures).

## Transition time profiling: 
The easiest way to profile the layer's transition cost is to generate transition per layer engines. ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#abstract) refers to executable DNN files, we follow the same terms to prevent any confusion)
```bash
python3 build_transition_time_engines.py
```



## EMC utilization can be profiled running the command below. 
Figure 3 is calculated running the commands below.

DNNs are generated by running the script below. The script reads prototxt files from `convolution_characterization_prototxts/` and generates a TensorRT engine for each layer in `build/convolution_characterization_plans/`.

```bash
python3 scripts/emc_analysis/engine_build_convolution_characterization.py
```

To run the generated DNNs and profile them, run the command below.

```bash
python3 scripts/emc_analysis/emc_util_all.py
```

The output is visible in output/emc_results.yaml

```bash
cat output/emc_results.yaml
```
Optionally you would prefer to run a specific DNN engine you can use the command below
```bash
scripts/emc_single_run.sh <your-TensorRT-engine>
```
The script will provide you the maximum EMC usage during the engine's run.

## Synchronous multiple DNN execution

* create two distinct copies of the original Tensorrt directory to an empty directories
* *replace sampleInference.cpp with the corresponding directories
* build the directories & write 0 to a tmp shared file.
* built googlenet only gpu and dla engines
* run the multiple dnn
#TODO_ISMET UPDATE THE SHM FILE, CHANGE THE NAME, assign to 0 in run_multiple.py
```bash
cp -r /usr/src/tensorrt tensorrt_sharedMem1 && cp -r /usr/src/tensorrt tensorrt_sharedMem2
cp modified_tensorrts/sampleInference1.cpp tensorrt_sharedMem1/samples/common/sampleInference.cpp  && cp modified_tensorrts/sampleInference2.cpp tensorrt_sharedMem1/samples/common/sampleInference.cpp 
cd tensorrt_sharedMem1/samples/trtexec && make -j4 & cd ../../../tensorrt_sharedMem2/samples/trtexec && make -j4 
echo '0' | sudo tee /tmp/shared_mem.txt
python3 build_engine.py
mkdir multi_dnn_execution_logs && python3 run_multiple_dnn.py
```

#TODOS:
Citation file
