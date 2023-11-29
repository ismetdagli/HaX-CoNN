# HaX-CoNN Artifact
This is the artifact of HaX-Conn: Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips. The artifact described here includes the source code for HaX-CoNN GPU and DLA runtimes and the sources for the applications used in our evaluation.

## Description

### If the reviewer has access to edge devices:

1. Check-list (artifact meta information)
* Hardware: NVIDIA Jetson Xavier AGX 32 GB and NVIDIA Jetson Orin AGX 32 GB
* Software for easy installation: [Jetpack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Xavier AGX and [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack-sdk-511) on Orin AGX
* Architecture: aarch64 
* Software details needed: Xavier AGX uses Python 3.6.9, TensorRT 7.1.3, CUDA 10.2.89  and Orin AGX uses Python 3.8.10, TensorRT 8.4.0, CUDA 11.2
* Binary: Binary files are large. So, generating them by using scripts in this artifact is necessary.
* Output: Profiling data (execution time, transition time, memory use) for both layers and neural networks. The end results is the improved execution time/throughput. 
* Experiment workflow: Python and bash scripts

2. Hardware dependencies

We performed our experiments on an NVIDIA Jetson Xavier AGX 32 GB and NVIDIA Jetson AGX Orin  32 GB. While HaX-CoNN is compatible with any architectures using TensorRT with NVIDIA GPUs, we also use DLA which does only exist in Jetson Families. So, reproducibility of current status of the code requires Xavier AGX or AGX Orin.

3. Software dependencies

The easiest way to follow our dependencies is to use [Jetpack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Xavier AGX and [Jetpack 5.1.1](https://developer.nvidia.com/embedded/jetpack-sdk-511) on Orin AGX.  We mainly use TensorRT as ML framework in our implementation since DLA can be programmed via only TensorRT. Xavier AGX has TensorRT 7.1.3  and Orin AGX uses TensorRT 8.4.0. It is important to note that manually installing TensorRT/Cuda etc. is not suggested.

4. Installation 

We assume installation through JetPack is followed. Upon it, run the script below to install python dependencies.

pip: sudo apt install -y python3-pip

The command below will install natsort, jetson-stats and z3-solver:
```bash
sudo -H pip3 install -r requirements.txt
```

If you are using a different Python 3 version than the default one that comes with JetPack, please modify the default version as 3.6.9 on Xavier AGX and 3.8.10 on Orin AGX by using [update-alternatives](https://hackersandslackers.com/multiple-python-versions-ubuntu-20-04/)

Note: Creating a docker or a VM is infeasible due to large size/access to the required hardware (DLA) etc. The authors provide remote access as explained below. 

5. Public availability

We maintain the most updated version of the code in this repository. Please refer to this repository for the most updated version.

As stated in requirements of green badge definition, this code is publicly available under zenodo [in this link](INSERT_LINK_ISMET_TODO) 

### If the reviewer opts to access remotely to our edge devices:

We target NVIDIA Jetson boards since it has DLA and GPU. Moreover, DLA can be only programmed via TensorRT. For this reason, this artifact requires access to NVIDIA Xavier AGX and AGX Orin boards. So, as an remote access, we request reviewers to use AnyDesk. We understand reviewer's busy schedule and the remote access will be open to the reviewer anytime during the review progress. If the reviewer may face difficulties of connecting to the device(not common but may occur for first timers), the authors are kindly requested to connect with authors.

AnyDesk id: TODO_ISMET
AnyDesk password: TODO_ISMET

Needed software for remote access: AnyDesk

Login information to Xavier AGX and AGX Orin are in the edge_account_info.txt in the desktop (visible after connection)


## Starter Guide:

Device power mode: We use these devices in [MAXN power mode](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-325/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html), which is default max power mode in these systems.

```bash
sudo nvpmodel -m 0
```

This is a starter guide of an example motivated in our paper. Basically, the system needs to run ResNet101 and GoogleNet. The system can either run both DNNs on GPU, or select the map among GPU and DLA. These mappings are done in DNN-level. What we propose is to distribute the layers among GPU and DLA in layer-level. There are four steps in this script:

Step 1: We build necessary engines to run Googlenet and Resnet on GPU and DLA.

Step 2: We build our modified TensorRT implementation.

Step 3: We collect profiling data for four different scenario explained above.

Step 4: We report the execution time upon baselines.

```bash
chmod +x starter_guide.sh
#This may take 10-20 minutes on Xavier AGX.
./starter_guide.sh

```
```bash
#Expected Output
Final Step: Summary of experiments
Average time of using only GPU: 12.6
Average time of Resnet101 on GPU and Googlenet on DLA: 12.4
Average time of Resnet101 on DLA and Googlenet on GPU: 14.3
Average time of the schedule found by HaX-CoNN: 7.9
Overall improvement over best-baseline: 57.3%


```


Note: The values might slightly be different since this is an empirical study. This artifact does not claim 100% match of the results with the reports. Minimal changes may occur (which are normally less than a couple of percentage). 
Note 2: While performing our experiments, we are trying to use the most updated JetPack versions. However, as the system get an update, the execution time may slightly affected. This execution time affects profiles/baselines/HaX-CoNN results. The authors did their best to give consistent experiment design and results.

## Experimental Setup in Detail (Step by step instructions)

This is an empirical study. We are listing the details on how we collected data. The data collected through profiling has been encoded to scripts. Run the makefile to build some of the necessary binaries to collect data.

NOTE: Even though collecting data per executions takes a couple of seconds, building an engine/plan takes a couple of minutes. This is because TensorRT builder checks and applies possible optimizations to run the kernels efficiently. Even though disabling some of them are partially provided by their APIs, this is not definitely suggested to comprehensively evaluate our work. So, building the binary files and running `make` below takes ~1 hours on Xavier AGX. 

```bash
cd HaX-CoNN/
export PYTHONPATH="$(pwd):$PYTHONPATH"
#After running make, it requests sudo in a couple of minutes and waits until reading the password. Please watch out!
make
```

### Step 1: Building Engines

The builder script `src/build_engine.py` can be used to serve TensorRT engines with varying configurations. Engines can be configured to be run only on gpu or dla or both by setting a transition layer.

```bash
> python3 src/build_engine.py -h
usage: build_engine.py [-h] --prototxt PROTOTXT --output OUTPUT --start
                       {gpu,dla} [--transition TRANSITION] [--verbose]

Build a TensorRT engine from a Caffe prototxt file.

optional arguments:
  -h, --help            show this help message and exit
  --prototxt PROTOTXT   Path to the input Caffe prototxt file
  --output OUTPUT       Output path to save the output engine
  --start {gpu,dla}     Specify whether to start on GPU or DLA
  --transition TRANSITION
                        Layer index where the transition occurs. Omit the
                        option if a single device will be used.
  --verbose             Enable verbose output
```

.e.g.
```bash
mkdir temp
#prototxt input files are given for each target DNN.
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output temp/googlenet_gpu_only.plan \
--start gpu \
--transition -1 \
--verbose
```

Note 2: Transition given at -1 represents that there will no transition for the engine and it will assign layers to the device given for --start. 

The output of file will be used as an input file to TensorRT (trtexec) binary file. While building engines/plans, TensorRT applies JIT optimizations that optimizes the kernel execution. Basically, engines/plans stores the optimized execution of target models + weight/layer topology etc. (anything related to DNN inference) 

### Step 2: Layer profiling: 

Input File:

 -  Prototxt File: Specified in `PROTOTXT` (`prototxt_input_files/googlenet.prototxt`). This file describes the architecture of the GoogleNet model.

Intermediate Files:

 -  GPU Engine Plan File: Located in `build/googlenet_gpu_plans/` directory. TensorRT engine file for the GoogleNet model running only on GPU:
 -  Profile Output File of GPU: In `TR_TIME_PROFILES_DIR` (inside `build/googlenet_transition_plans/profiles`). These files contain detailed execution profiles for each engine plan.
 -  DLA Engine Plan Files: Located in `TR_TIME_PLANS_DIR` (`build/googlenet_transition_plans` directory). These are the TensorRT engine files for the GoogleNet model with transitions at different layers:
        DLA Engine Plans (`PLANS_DLA`): For running the model initially on DLA. These are used for DLA layer analysis
 -  Profile Output Files: In `TR_TIME_PROFILES_DIR` (inside `build/googlenet_transition_plans/profiles`). These files contain detailed execution profiles for each engine plan.
 -  Filtered GPU Profile: In `build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json`, GPU's filtered profile data resides. This file is heavily filtered from the original profile giving information about the name, execution time and layer count in the fused layer. 
 -  DLA Profile Results: In `output/dla_compute_times.json`, every transition engine is run and their DLA layer execution information is summed and saved in this file.

Output Files:

 -  Layer Analysis Results JSON: The file `output/layer_results.json` is the final output. It contains the layer profiling for GPU and DLA. 

This creates a text file of a DNN. The line after " [I] GPU Compute" are our target data. We use *mean* data as the average of X number of iterations iteration is passed as argument to our trtexec binary file. We generally use 1000 iteration to mitigate if any unexpected noise occurs.


```bash
python3 src/collect_data_single_layers.py
```

This execution runs googlenet on GPU with 1000 iteration, 1000ms warmup. The output generated by the executions are the total/average/percentages of times in layer-level.

Note: `+` sign demonstrates the layers are merged. `||` demonstrates outputs of the layers will be concataned (as concatanation layer). `{}` demonstrates that DLA fuses the layers and profiling of all layers are treated as one layer(basically, this is a profiling limitation in DLA architectures). 


To generate filtered layer timing information in json:
```bash
python3 scripts/layer_analysis/layer_gpu_util.py --profile <profile-path>
```
e.g.
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


### Step 3: Transition time profiling: 

#### File Summary

Input File:

 -  Prototxt File: Specified in `PROTOTXT` (`prototxt_input_files/googlenet.prototxt`). This file describes the architecture of the GoogleNet model used for transition time analysis.

Intermediate Files:

 -  Engine Plan Files: Located in `TR_TIME_PLANS_DIR` (`build/googlenet_transition_plans` directory). These are the TensorRT engine files for the GoogleNet model with transitions at different layers, including:
        GPU Engine Plans (`PLANS_GPU`): For running the model initially on GPU.
        DLA Engine Plans (`PLANS_DLA`): For running the model initially on DLA.
 -  Profile Output Files: In `TR_TIME_PROFILES_DIR` (inside `build/googlenet_transition_plans/profiles`). These files contain detailed execution profiles for each engine plan.
 -  Profile Log Files: In `TR_TIME_PROF_LOGS_DIR` (inside `build/googlenet_transition_plans/profile_logs`). These logs include console outputs from the profiling process.

Output Files:

 -  Transition Cost JSON: The file `output/transition_results.json` is the final output. It contains the mean execution times and their transition costs for each convolution layer configuration. This data is gathered by running the engine files and profiling.

#### Script Summary
Scripts which are specific to Transition analysis are summarised below:

- `python3 scripts/transition_time_analysis/transition_util.py`

#### Process Overview:

To build all necessary engines, measure their transition costs and save the output run the following. You can view the final results in the `output/emc_results.json` file.

```bash
make emc
cat output/transition_results.json
```

 1. Engine File Generation:
 The build_engine.py script is used to generate engine files for both GPU and DLA executions based on the GoogleNet model defined in the Prototxt file.
 Two sets of engine files with different transition layers are generated at this step.

Example builds for single engine:
```bash
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output build/googlenet_transition_plans/googlenet_gpu_transition_at_24.plan \
--start gpu \
--transition 24 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output build/googlenet_transition_plans/googlenet_dla_transition_at_24.plan \
--start dla \
--transition 24 \
--verbose
```

 2. Profile and Log Generation:
    Using trtexec, the model is run with each engine file, and detailed performance profiles are collected.
    These profiles are saved as intermediate files in TR_TIME_PROFILES_DIR, accompanied by logs in TR_TIME_PROF_LOGS_DIR.

Makefile generates all the engines in every transition layer. An example is provided below:
```bash
python3 src/build_engine.py --prototxt prototxt_input_files/googlenet.prototxt \
--starts gpu --output build/googlenet_transition_plans/googlenet_gpu_transition_at_0.plan \
--transition 141 --verbose

/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile --exportProfile=build/googlenet_transition_plans/profiles/googlenet_gpu_transition_at_0.profile \
--avgRuns=1 --warmUp=5000 --duration=0 --loadEngine=build/googlenet_transition_plans/googlenet_gpu_transition_at_0.plan > build/googlenet_transition_plans/profile_logs/googlenet_gpu_transition_at_0.log
```
The transition analysis makes use of mean compute values. You can view the logs to see the mean values:
```bash
> cat build/googlenet_transition_plans/profile_logs/googlenet_gpu_transition_at_109.log | grep -C 4 mean
[11/26/2023-13:46:01] [I] Average on 1 runs - GPU latency: 2.62305 ms - Host latency: 2.68164 ms (end to end 2.69336 ms, enqueue 2.63281 ms)
[11/26/2023-13:46:01] [I] Host Latency
[11/26/2023-13:46:01] [I] min: 2.38086 ms (end to end 2.3877 ms)
[11/26/2023-13:46:01] [I] max: 7.18164 ms (end to end 7.19727 ms)
[11/26/2023-13:46:01] [I] mean: 2.56869 ms (end to end 2.57815 ms)
[11/26/2023-13:46:01] [I] median: 2.50391 ms (end to end 2.51367 ms)
[11/26/2023-13:46:01] [I] percentile: 3.17871 ms at 99% (end to end 3.19629 ms at 99%)
[11/26/2023-13:46:01] [I] throughput: 383.698 qps
[11/26/2023-13:46:01] [I] walltime: 26.0622 s
--
[11/26/2023-13:46:01] [I] median: 2.46094 ms
[11/26/2023-13:46:01] [I] GPU Compute
[11/26/2023-13:46:01] [I] min: 2.33887 ms
[11/26/2023-13:46:01] [I] max: 7.11914 ms
[11/26/2023-13:46:01] [I] mean: 2.51739 ms
[11/26/2023-13:46:01] [I] median: 2.45142 ms
[11/26/2023-13:46:01] [I] percentile: 3.10742 ms at 99%
[11/26/2023-13:46:01] [I] total compute time: 25.1739 s
[11/26/2023-13:46:01] [I] 
```

3. Results Compilation

The final python script parses the mean values, processes the difference between baseline value and compiles all of the data into a single json:

```bash
python3 scripts/transition_time_analysis/transition_util.py
```

You can view the transition cost analysis results in `output/transition_results.json`

```bash
> cat output/transition_results.json                         
{
    "googlenet_dla_transition_at_-1": {
        "mean_time": 1.9701,
        "transition_cost": 0.0
    },
    "googlenet_dla_transition_at_0": {
        "mean_time": 4.07455,
        "transition_cost": 2.10445
    },
    "googlenet_dla_transition_at_10": {
        "mean_time": 3.11484,
        "transition_cost": 1.14474
    },
    "googlenet_dla_transition_at_24": {
        "mean_time": 3.05384,
        "transition_cost": 1.08374
    },
    ...
```

### Step 4: EMC Analysis 

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

#### Process Overview:

To build all necessary engines, measure their EMC utilizations and save the output run the following. You can view the final results in the `output/emc_results.json` file.

```bash
make emc
cat output/emc_results.json
```

 1.  Engine File Generation: For each Prototxt file in `PROTOTXT_DIR`, a corresponding engine (.plan) file is generated in `EMC_PLANS_DIR` using the script build_engine.py. This script configures and builds a TensorRT engine for each layer configuration described in the Prototxt files.

An example build for a single engine:
 ```bash
python3 src/build_engine.py --prototxt convolution_characterization_prototxts/conv1_kernel1.prototxt --output build/convolution_characterization_plans/conv1_kernel1.plan --start gpu
 ```

 2.  EMC Utilization Measurement: The script `emc_single_run.sh` is executed for each engine file. It runs the engine and measures the EMC utilization, storing the results in `EMC_TIMES_DIR` (`build/convolution_characterization_plans/times` directory).

 An example EMC utilization measurement from single engine:
 ```bash
mkdir build/convolution_characterization_plans/times
scripts/emc_analysis/emc_single_run.sh build/convolution_characterization_plans/conv1_kernel1.plan build/convolution_characterization_plans/times/conv1_kernel1.txt
 ```

 The execution above will print %89 as the max of the emc utilization. The time distribution can be viewed:
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
 View the output: 
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

Reference from paper: The outputs in Json file demonstrates very similar pattern in the Figure 3. The effect of varying with input and filter size for a convolution layer is illustrated in Figure 3.)

### Step 5: Memory Throughput Profiling

This step targets to profile memory throughput.  This code profiles the memory throughput in kernel level(Nsight compute automatically converts layers to kernels).

```bash
mkdir nsight_compute_logs
#Note: This prompt requests sudo privilege. Takes a couple of minutes to run
python3 src/nsight_compute.py
```

This code outputs a nsight_compute_$DNN_.report. TensorRT has its own naming and output report structure that becomes very complicated. This requires a lot of effort to match the layers and their instructions. We leave this script here for a reference. 

For the ones who are interested in, we suggest to follow this strategy explained as a summary below. This gives the memory throughput of a layer. We use a recorded memory throughput data in z3 solvers below.

To calculate a memory throughput of a layer group:  (memory throughput of a layer) * (duration of a layer in the group) / (duration of all layers in the group).   

We can't profile memory throughput of DLA since Nsight Compute does not allow to use DLA(targets only GPU). So, we convert the profiling data we have obtained in EMC analysis by using each layer group's EMC GPU and DLA utilization. The formulation given below. EMC data is obtained as explained above and memory throughput is profiled via Nsight Compute.

DLA's memory throughput for a layer X: (EMC utilization of layer X on DLA / EMC utilization of layer X on GPU) * memory throughput of layer X on GPU


### Step 6: Synchronous multiple DNN execution

We assume that multiple DNNs starts at the same time. 

This session is also briefly explained in "Neural network synchronization" in Session 4.

* create two distinct copies of the original Tensorrt directory to an empty directories
* replace sampleInference.cpp with the corresponding directories
* build the directories & write 0 to a tmp shared file.
* built googlenet only gpu and dla engines
* run the multiple dnn
```bash
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
python3 ./starter_guide_experiment.py
```

### Step 7: Z3 Solver execution

Add z3 solver code.

Until now, we followed the steps on how to profile and execute DNNs. We profiled execution time of layers, transition time and memory throughput of layers. All these data are used as an input.

Note: slowdown values are found by [PCCS](https://dl.acm.org/doi/abs/10.1145/3466752.3480101) and [Source Code](https://github.com/processorcentricmodel/PCCS). If a user targets to use a different environment, the model needs to be reconstructed. We follow the steps defined by the authors.

```bash
python3 z3_solver_multi_dnn.py > output/schedule_summary.txt
```

The expected output from solver is the schedule of DNNs.

```bash
> cat output/schedule_summary.txt
GoogleNet starts on GPU
GoogleNet applies transition at 81
Resnet50 starts on DLA
Resnet50 applies transition at 39
```

For other DNNs execution scenarios, previous steps needs to be followed and the profile data needs to be given the corresponding files.


### Step 8: Multi DNN, HaX-CoNN results

Until here, we have been collecting profiling data for execution time, transition time and memory throughput in layer-level. Then, we used z3 solver to find the corresponding schedule. Now, to verify the performance of our model, we will execute the corresponding schedules.

For the baselines, we generate GPU executions, GPU&DLA executions, and [H2H](https://dl.acm.org/doi/10.1145/3489517.3530509) and [Herald](https://www.computer.org/csdl/proceedings-article/hpca/2021/223500a071/1t0HUXqfspW). We follow their implementation as given in [source code](https://github.com/xyzxinyizhang/H2H). For Herald, we stop the execution at step 2 (as H2H did in their methodology) and for H2H baseline, we follow the execution for all steps (including step 4). 

For the sake of simplicity of the artifact, we provide the schedules previously found as an input in baseline.txt. We build the engines, run them and compare the results. To run the experiment, run the script below:

```bash
chmod +x baseline_engine_building.sh
bash baseline_engine_building.sh
```


#### Reusability on Orin AGX
For a demonstrationg of reusability of our setup, we target to run the same DNNs. We have to use TensorRT to be able to use DLA. However, TensorRT version is 8.5 in Orin AGX (unlike Xavier AGX which has 7.1.3).
Moreover, we use INT8 setting unlike FP16 in Xavier AGX settings. Even though these changes seems easy enough, using a new devices with many different chan For the sake of simplicity, profiling data has been preprocessed and the functions are adapted for new TensorRT version. (There has been changes in the function name in TensorRT's API). Plus, we have integrated the calibration for FP16 to INT8. 

To run the experiments, please login Orin AGX(credentials given in desktop)

Follow the steps to build tensorrt binary files. Since the version is different, sampleInference is given in a different file (sampleInference$Number_orinagx.cpp) Also, building engine file is different too(src/build_engine_orin.py). For the sake of easiness of the reviewer, we add the commands to run below for Orin AGX 

```bash
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

python3 ./src/build_engine_orin.py --prototxt ./prototxt_input_files/googlenet.prototxt --start gpu --output ./google_only_gpu.plan
python3 ./src/build_engine_orin.py --prototxt ./prototxt_input_files/googlenet.prototxt --start dla --output ./google_only_dla.plan

mkdir ./multi_dnn_execution_logs/

#Run the python code to check TensorRT binaries working fine.
TODO_ISMET script_to_collect_data
```


### Step 9: Single DNN, HaX-CoNN results #TODO_ISMET

Until here, we have been collecting profiling data for execution time, transition time and memory throughput in layer-level. Then, we used z3 solver to find the corresponding schedule. Now, to verify the performance of our model, we will execute the corresponding schedules. For the sake of simplicity of the artifact, we provide the schedules found as an input to run  

Run z3 to find the single dnn schedule:

Z3 should be like this
input: Run the command line with profiling data
output: schedule to run for those DNNs




### Step 10: Overhead Analysis

#TODO\_Ismet-OR-Eymen

Before starting make sure:
```bash
chmod +x scripts/run_all_plan.sh scripts/run_forever.sh
mkdir -p build/overhead_gpu/logs build/overhead/alexnet_dla
```
- Build AlexNet DLA engine.

```bash
python3 src/build_engine.py --prototxt prototxt_input_files/alexnet.prototxt --output build/overhead/alexnet_dla.plan --start dla
```

- Build GPU engines for DenseNet GoogleNet Inc-res-v2 Inception MobileNet ResNet18 ResNet50 ResNet101 ResNet152 VGG16 VGG19

```bash
python3 src/build_engine.py --prototxt prototxt_input_files/ --output build/overhead_gpu --start gpu
```

- Run AlexNet DLA with each GPU engines. Collect data for each execution. it should include the execution of 121 DNNs Example:AlexnetDLA-DenseNetGPU, AlexnetDLA-GoogleNetGPU, AlexNet-InceptionGPU etc.

```bash
./scripts/run_forever.sh build/overhead/alexnet_dla.plan build/overhead/alexnet_dla/ &
./scripts/run_all_plan.sh build/overhead_gpu build/overhead_gpu/logs

pkill run_forever.sh
```

- Then, run z3 solver(src/dummy_z3_solver) in an infinite loop and run the same executions of AlexnetDLA + GPU(any network). Note: while(True) at Line 128 on z3_solver helps to run in an infinite loop. You can define a automated stopping mechanism to stop the code after all executions done. (not necessary but optional. If we leave as it is, it should not be an issue)

```bash

```

Compare the results of executions(each DNN on GPUs) with z3 and without z3. Comparison example: 
average exec time of Inception on GPU (when alexnet on DLA + z3 running ) / average exec time of Inception on GPU (when alexnet on DLA + no z3)


