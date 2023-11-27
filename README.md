# HaX-CoNN Artifact
This is artifact of HaX-Conn: Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips. This 

Artifact described here includes the source code for HaX-CoNN GPU and DLA runtimes and the sources for the applications used in our evaluation.

## Description

1. Check-list (artifact meta information)
* Hardware: NVIDIA Jetson Xavier AGX 32 GB and NVIDIA Jetson Orin AGX 32 GB
* Software easy installation: [Jetpack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Xavier AGX and [TODO-Jetpack Version](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Orin AGX
* Architecture: aarch64 
* Software details needed: Xavier AGX uses Python 3.6.9, TensorRT 7.1.3, CUDA 10.2.89  and Orin AGX uses Python 3.8, TensorRT 8.4.0, CUDA 11.2
* Binary: Binary files are large. So, generating binary files are neccesary by using scripts in this artifact.
* Output: Profiling data (execution time, transition time, memory use) for both layers and neural networks. The end results is the improved execution time/throughput. 
* Experiment workflow: Python and bash scripts

2. Hardware dependencies

We performed our experiments on an NVIDIA Jetson Xavier AGX 32 GB and NVIDIA Jetson AGX Orin  32 GB. While HaX-CoNN is compatible with any architectures using TensorRT with NVIDIA GPUs, we also use DLA which does only exist in Jetson Families. So, reproducibility of current status of the code requires Xavier AGX or AGX Orin.

3. Software dependencies

The easiest way to follow our dependencies is to use [Jetpack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Xavier AGX and [TODO-Jetpack Version](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Orin AGX.  We mainly use TensorRT as ML framework in our implementation since DLA can be programmed via only TensorRT. Xavier AGX has TensorRT 7.1.3  and Orin AGX uses TensorRT 8.4.0. It is important to note that manually installing TensorRT/Cuda etc. is not suggested.

4. Installation 

We assume installation through JetPack is followed. Upon it, run the script below to install python dependencies.

pip: sudo apt install -y python3-pip

The command below will install natsort, jetson-stats and z3-solver:
```bash
sudo -H pip3 install -r requirements.txt
```



## Experimental Setup

This is a empirical study. We are listing the details how we collected data. The data collected through profiling has been encoded to scripts. Run the makefile to built some of the necessary binaries to collect data

TODO_EYMEN: Eymen, you need to explain what has been built after make file ()
My understanding is this: (please modify/elaborate/update etc. to make this instruction clear and detailed)
1- Built googlenet 22 tensorrt binary file running only GPU and DLA. The first 11 binary uses only GPU and the next 11 binary uses DLA. (Line 17)
2- We collect iterate through binary files (.plan/.engine) to collect total execution time (line 23). (Refer to "Transition time profiling" section below for further details )

3- QUESTION_EYMEN:Do we run such things? We built 25 convolution layer engines varying input sizes and filter (kernel) sizes. We measure external memory controller (EMC) utilization while running these engines of convolution layers. 
4- QUESTION_EYMEN: Do we run EMC profiling here?



NOTE: Running make takes ~1/2 hours on Xavier AGX. 

```bash
cd HaX-CoNN/
export PYTHONPATH="$(pwd):$PYTHONPATH"
make
```

### Building Engines

The builder script `src/build_engine.py` can be used to serve TensorRT engines with varying configurations. Engines can be configured to be run only on gpu or dla or both by setting a transition layer.

```bash
> python3 src/build_engine.py -h
usage: build_engine.py [-h] --prototxt PROTOTXT --output OUTPUT
                       [--starts_gpu STARTS_GPU] [--transition TRANSITION]
                       [--verbose]

Build a TensorRT engine from a Caffe prototxt file.

optional arguments:
  -h, --help            show this help message and exit
  --prototxt PROTOTXT   Path to the input Caffe prototxt file
  --output OUTPUT       Output path to save the output engine
  --starts_gpu STARTS_GPU
                        Whether the network starts on GPU (True) or DLA
                        (False)
  --transition TRANSITION
                        Layer index where the transition occurs. Omit the
                        option if a single device will be used.
  --verbose             Enable verbose output
```

## Layer profiling: 
This creates a text file of a DNN. The line after " [I] GPU Compute" are our target data. We use *mean* data as the average of X number of iterations iteration is passed as argument to our trtexec binary file. We generally use 1000 iteration to mitigate if any unexpected noise occurs.


```bash
python3 collect_data_single_layers.py
```
Note: `+` sign demonstrates the layers are merged. `||` demonstrates outputs of the layers will be concataned (as concatanation layer). `{}` demonstrates that DLA fuses the layers and profiling of all layers are treated as one layer(basically, this is a profiling limitation in DLA architectures).

To generate filtered layer timing information in json:
```bash
python3 scripts/layer_analysis/layer_gpu_util.py --profile <profile-path>
```
.e.g.
```bash
python3 scripts/layer_analysis/layer_gpu_util.py --profile build/googlenet_transition_plans/profiles/googlenet_dla_transition_at_24.profile
```


#TODO_EYMEN: We need to add a script/command here showing that we are generating the layer's execution time. The command should generate an output file as this:

Layer group     GPU(ms)   DLA(ms)
0-9               x         y
10-24             x         y
25-38             x         y
39-52             x         y
.
.
.


## Transition time profiling: 
The easiest way to profile the layer's transition cost is to generate transition per layer engines. ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#abstract) refers to executable DNN files, we follow the same terms to prevent any confusion)
```bash
python3 build_transition_time_engines.py
```

Makefile generates all the engines in every transition layer. To create your own engine:
```bash
python3 src/build_engine.py --prototxt <prototxt-path> --starts_gpu True --output <output-path> --transition <transition> --verbose
/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile --exportProfile=<profile-path> --avgRuns=1 --warmUp=5000 --duration=0 --loadEngine=<engine-path> > <log-path>
```
e.g.
```bash
python3 src/build_engine.py --prototxt prototxt_input_files/googlenet.prototxt --starts_gpu False --output build/googlenet_transition_plans/googlenet_dla_transition_at_141.plan --transition 141 --verbose

/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile --exportProfile=build/googlenet_transition_plans/profiles/googlenet_gpu_transition_at_0.profile --avgRuns=1 --warmUp=5000 --duration=0 --loadEngine=build/googlenet_transition_plans/googlenet_gpu_transition_at_0.plan > build/googlenet_transition_plans/profile_logs/googlenet_gpu_transition_at_0.log
```

#TODO_EYMEN: Similar to above, We need to add a script/command here showing that we are generating the layer's transition cost time. The command should generate an output file as this:

Layer group     Transition from GPU to DLA
0-9                         x         
10-24                       x         

25-38                       x         
39-52                       x         
.
.
.

## EMC Analysis 

#### File Summary

Input Files:

 - Prototxt Files: Located in `PROTOTXT_DIR` (`convolution_characterization_prototxts` directory). These files describe the convolution layer configurations, including input sizes and filter (kernel) sizes.

Intermediate Files:

 - Engine Plan Files: Located in `EMC_PLANS_DIR` (`build/convolution_characterization_plans` directory). These are the TensorRT engine files (.plan) generated from the input Prototxt files. Each engine file represents a specific convolution layer configuration and is used to measure EMC utilization.
 - Time Text Files: Located in `EMC_TIMES_DIR` (`build/convolution_characterization_plans/times` directory). These are the EMC utilization percentage distributions saved from the start to the end of engine's run. Every engine has its own time distribution in text format.

Output File:

 -  EMC Utilization JSON: The file `output/emc_results.json` is the final output. It contains the EMC utilization data for each convolution layer configuration. This data is gathered by running the engine files and measuring EMC utilization.

#### Script Summary
Scripts which are specific to EMC analysis are summarised below:

- `scripts/emc_analysis/emc_single_run.sh <tensorrt-engine-path> <output-path>`
- `python3 scripts/emc_analysis/emc_util_all.py` 

### Process Overview:

 1.  Engine File Generation: For each Prototxt file in `PROTOTXT_DIR`, a corresponding engine (.plan) file is generated in `EMC_PLANS_DIR` using the script build_engine.py. This script configures and builds a TensorRT engine for each layer configuration described in the Prototxt files.


An example build for single engine:
 ```bash
python3 src/build_engine.py --prototxt convolution_characterization_prototxts/conv1_kernel1.prototxt --output build/convolution_characterization_plans/conv1_kernel1.plan --starts_gpu True
 ```

 2.  EMC Utilization Measurement: The script `emc_single_run.sh` is executed for each engine file. It runs the engine and measures the EMC utilization, storing the results in `EMC_TIMES_DIR` (`build/convolution_characterization_plans/times` directory).

 An example EMC utilization measurement from single engine:
 ```bash
scripts/emc_analysis/emc_single_run.sh build/convolution_characterization_plans/conv1_kernel1.plan build/convolution_characterization_plans/times/conv1_kernel1.txt
 ```
 The time distribution can be viewed:
 ```bash
> cat build/convolution_characterization_plans/times/conv1_kernel1.txt
8%
4%
18%
57%
75%
82%
86%
88%
88%
89%
89%
88%
 ```

 3.  Results Compilation: Finally, the Python script `emc_util_all.py` compiles all the EMC utilization measurements from `EMC_TIMES_DIR` into a single JSON file, `output/emc_results.json`, by finding the maximum in each time file.

 Running the script:
 ```bash
 python3 scripts/emc_util_all.py
 ```
 View the output: (TODO_ISMET: We will give a reference to figure 3 in the paper)
 ```bash
 > cat output/emc_results.json
{
    "conv1": {
        "kernel1": "89%",
        "kernel2": "78%",
        "kernel3": "70%",
        "kernel4": "56%",
        "kernel5": "48%"
    },
    "conv2": {
        "kernel1": "77%",
    ...
 ```

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
python3 src/build_engine.py --prototxt prototxt_input_files/googlenet.prototxt --starts_gpu True --output google_only_gpu.plan
python3 src/build_engine.py --prototxt prototxt_input_files/googlenet.prototxt --starts_gpu False --output google_only_dla.plan
mkdir multi_dnn_execution_logs && python3 run_multiple_dnn.py
```

#Todo_ismet:

Add z3 solver code.

Give a reference to the code with execution time, transition time and memory use



#TODOS:
Citation file
