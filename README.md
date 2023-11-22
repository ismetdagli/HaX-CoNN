# HaX-CoNN Artifact
This is artifact of HaX-Conn: Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips. This 

Artifact described here includes the source code for HaX-CoNN GPU and DLA runtimes and the sources for the applications used in our evaluation.

## Description

1. Checklist(meta information)
* Hardware: Jetson Xavier AGX 32 GB
* Software easy installation: Jetpack 4.5
* Software details: #TODO_ISMET


# Experimental Setup

First and foremost, this is a empirical study. We are open sourcing all the details how we collected data. The data collected through profiling has been encoded to script.  

## Layer profiling: This creates a text file of a DNN. The line after " [I] GPU Compute" are our target data. We use mean data as the average of X number of iterations   #TODO_ISMET
```bash
python3 collect_data_single_layers.py
```
Note: `+` sign demonstrates the layers are merged. `||` demonstrates outputs of the layers will be concataned (as concatanation layer). `{}` demonstrates that DLA fuses the layers and profiling of all layers are treated as one layer(basically, this is a profiling limitation in DLA architectures).

## Transition time profiling: The easiest way to profile the layer's transition cost is to generate transition per layer engines. ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#abstract) refers to executable DNN files, we follow the same terms to prevent any confusion)
```bash
python3 build_transition_time_engines.py
```



## EMC utilization can be profiled running the command below. 
Figure 3 is calculated through running the script below for each DNN.
```bash
sudo tegrastats >> emc_utilization.tex
```
DNNs are generated as running the script below. The script reads to prototxt files from $convolution_characterization_prototxts and generates an TensorRT engine for each layer. 

```bash
python3 engine_build_convolution_characterization.py
```

## Synchronous multiple DNN execution

1/2-create two distinct copies of the original Tensorrt directory to an empty directories
3/4-replace sampleInference.cpp with the corresponding directories
5/6-build the directories
7-run the multiple dnn

```bash
cp /usr/src/tensorrt tensorrt_sharedMem1 && cp /usr/src/tensorrt tensorrt_sharedMem2
cp modified_tensorrts/sampleInference1.cpp tensorrt_sharedMem1/samples/common/sampleInference.cpp  && cp modified_tensorrts/sampleInference2.cpp tensorrt_sharedMem1/samples/common/sampleInference.cpp 
cd tensorrt_sharedMem1/samples/trtexec && make -j4 & cd tensorrt_sharedMem2/samples/trtexec && make -j4 
python3 run_multiple_dnn.py
```



```bash
sudo tegrastats >> emc_utilization.tex
```