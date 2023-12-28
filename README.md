# HaX-CoNN Artifact
This is the artifact of HaX-Conn: Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips. The artifact described here includes the source code for HaX-CoNN GPU and DLA runtimes and the sources for the applications used in our evaluation.

## Description

1. Check-list (artifact meta information)
* Hardware: NVIDIA Jetson Xavier AGX 32 GB and NVIDIA Jetson Orin AGX 32 GB
* Software for easy installation: [Jetpack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Xavier AGX and [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack-sdk-511) on Orin AGX
* Architecture: aarch64 
* Software details needed: Xavier AGX uses Python 3.6.9, TensorRT 7.1.3, CUDA 10.2.89  and Orin AGX uses Python 3.8.10, TensorRT 8.4.0, CUDA 11.2
* Binary: Binary files are large, so generating them by using scripts in this artifact is necessary.
* Output: Profiling data (execution time, transition time, memory use) for both layers and neural networks. The end result is the improved total execution time/throughput. 
* Experiment workflow: Python and bash scripts

2. Hardware dependencies

We performed our experiments on an NVIDIA Jetson Xavier AGX 32 GB and NVIDIA Jetson AGX Orin  32 GB. While HaX-CoNN is compatible with any architectures using TensorRT with NVIDIA GPUs, we also use DLA which does only exist in NVIDIA Jetson Families. So, reproducibility of the code requires Xavier AGX or AGX Orin whereas the methodology can be applied to other heterogeneous shared-memory SoCs (i.e., Qualcomm 865 Development Kit).

3. Software dependencies

The easiest way to follow our dependencies is to use [Jetpack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive) on Xavier AGX and [Jetpack 5.1.1](https://developer.nvidia.com/embedded/jetpack-sdk-511) on Orin AGX.  We mainly use TensorRT as ML framework in our implementation, as DLA can only be programmed via TensorRT. Xavier AGX has TensorRT 7.1.3  and Orin AGX uses TensorRT 8.4.0. It is important to note that manually installing TensorRT/CUDA on Jetson boards etc. is not suggested.

4. Installation 

We assume installation through JetPack is followed. Upon it, run the script below to install python dependencies.

pip: sudo apt install -y python3-pip

The command below will install natsort, jetson-stats and z3-solver:
```bash
sudo -H pip3 install -r requirements.txt
```

If you are using a different Python 3 version than the default one that comes with JetPack, please modify the default version as 3.6.9 on Xavier AGX and 3.8.10 on Orin AGX by using [update-alternatives](https://hackersandslackers.com/multiple-python-versions-ubuntu-20-04/)

> Creating a docker or a VM is infeasible due to large size/access to the required hardware (DLA) etc. The authors can provide remote access as explained below. 

5. Public Availability

We maintain the most updated version of the code in this repository. Please refer to this repository for the most updated version.

As stated in requirements of green badge definition, this code is publicly available under [zenodo](https://zenodo.org/uploads/10225025)


## Starter Guide:

Device power mode: We use these devices in [MAXN power mode](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-325/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html), which is default max power mode in these systems.

```bash
sudo nvpmodel -m 0
```

This is a starter guide of an example motivated in our paper. Basically, the system needs to run ResNet101 and GoogleNet. The system can either run both DNNs on GPU, or select the map among GPU and DLA. These mappings are done at the DNN-level. What we propose is to distribute the layers among GPU and DLA at the layer-level. There are four steps in this script:

Step 1: We build the necessary engines to run Googlenet and Resnet on GPU and DLA.

Step 2: We build our modified TensorRT implementation.

Step 3: We collect profiling data for four different scenarios explained above.

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
Average time of the schedule found by HaX-CoNN: 10.3
Overall improvement over best-baseline: 20.6%
```


> The values might slightly be different since this is an empirical study. This artifact does not claim 100% match of the results with the reports. Minimal changes may occur (which are normally less than a couple of percentage). 

> While performing our experiments, we are trying to use the most updated JetPack versions. However, as the system get an update, the execution time may slightly affected. This execution time affects profiles/baselines/HaX-CoNN results. The authors did their best to give consistent experiment design and results.

## Experimental Setup in Detail (Step-by-step instructions)

This is an empirical study. We are listing the details on how we collected data. The data collected through profiling has been encoded into scripts. Run the makefile to build some of the necessary binaries to collect data.

> Even though collecting data per executions takes a couple of seconds, building an engine/plan takes a couple of minutes. This is because TensorRT builder checks and applies possible optimizations to run the kernels efficiently. Even though disabling some of them are partially provided by their APIs, this is not definitely suggested to comprehensively evaluate our work. So, building the binary files and running `make` below takes ~1 hours on Xavier AGX. 

```bash
cd HaX-CoNN/
export PYTHONPATH="$(pwd):$PYTHONPATH"
#After running make, it requests sudo in a couple of minutes and waits until reading the password. Please watch out!
make
```

### Step 1: Building Engines

#### Summary

This step explains how building engine scripts work. No instructions are needed for evaluation.

#### Details

The builder script `src/build_engine.py` can be used to serve TensorRT engines with varying configurations. Engines can be configured to be run only on gpu or dla or both by setting a transition layer.

```bash
> python3 src/build_engine.py -h
usage: build_engine.py [-h] --prototxt PROTOTXT --output OUTPUT --start
                       {gpu,dla} [--transition TRANSITION] [--mark MARK]
                       [--verbose]

Build a TensorRT engine from a Caffe prototxt file.

optional arguments:
  -h, --help            show this help message and exit
  --prototxt PROTOTXT   Path to the input Caffe prototxt file or a directory
                        for bulk build
  --output OUTPUT       Output path to save the output engine or directory for
                        bulk build
  --start {gpu,dla}     Specify whether to start on GPU or DLA
  --transition TRANSITION
                        Layer index where the transition occurs. Omit the
                        option if a single device will be used.
  --mark MARK           Layer index which is marked for output. Omit the
                        option for all analyses except for transition cost
                        analysis.
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

> Not giving any transition value or transition given at -1 represents that there will be no transition for the engine and it will assign layers to the device given for `--start`. 

The output of file will be used as an input file to TensorRT binary file (trtexec). While building engines/plans, TensorRT applies JIT optimizations that improves the kernel execution performance. Basically, engines/plans store the optimized execution of target models + weight/layer topology etc. (anything related to DNN inference) 

Until step 6, we settle our examples on GoogleNet. Each DNN has different details and settings that we need to go over. To keep the story flowing and consistent, we target on GoogleNet in this README.

### Step 2: Layer profiling:

#### Summary

This step explains how layers are profiled. Summary of comprehensive profiling results can be obtained by running the command below. 

```bash
mkdir output
make layer
#Pythonpath and mkdir output might be ignored if alreayd modified.
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 scripts/layer_analysis/layer_all_util.py --gpu_json build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json --dla_json output/dla_compute_times.json

# Summary of last lines given below:
# {
# ...
#     "124-140": {
#         "gpu": {
#             "total_gpu_time_ms": 0.21411756,
#             "layer_count": 17
#         }
#     }
# }

#Expected output given below in detail for further comparison. (This file contains pre-collected data by authors)
cat output_expected/layer_results.json
```




#### Details


Input File:

 -  Prototxt File: Specified in `PROTOTXT` (`prototxt_input_files/googlenet.prototxt`). This file describes the architecture of the GoogleNet model.

Intermediate Files:

 -  GPU Engine Plan File: Located in `build/googlenet_transition_plans/googlenet_gpu_transition_-1.plan`. TensorRT engine file for the GoogleNet model running only on GPU.
 -  Profile Output File of GPU: In `TR_TIME_PROFILES_DIR` (inside `build/googlenet_transition_plans/profiles`). 
 -  GPU Engine Plan Files: Located in `TR_TIME_PLANS_DIR` (`build/googlenet_transition_plans` directory).:
        GPU Engine Plans (`TR_PLANS_GPU`): For running the model initially on GPU and continuing on DLA. These are used for DLA layer analysis.
	DLA Engine Plans (`MARK_PLANS_DLA`): For running the model until a marked output. Analyzed deeply in transition cost analysis.
 -  Profile Output Files: In `TR_TIME_PROFILES_DIR` (inside `build/googlenet_transition_plans/profiles`). These files contain detailed execution profiles for each engine plan.
 -  Filtered GPU Profile: In `build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json`, GPU's filtered profile data resides. This file is heavily filtered from the original profile giving information about the name, execution time and layer count in the fused layer. 

Output Files:

 -  DLA Profile Results: In `output/dla_compute_times.json`, every transition engine is run and their DLA layer execution information is summed and saved in this file.
 -  Layer Analysis Results JSON: The file `output/layer_results.json` is the final output. It contains the layer profiling for GPU and the extracted info from DLA result file. 

To generate all necessary files:
```bash
make layer
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 scripts/layer_analysis/layer_all_util.py --gpu_json build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json --dla_json output/dla_compute_times.json
```

1. Engine File Generation

The build can be done conveniently by the build_engine.py script. One thing to note is we need two types of engines. One with transitions and the other with a certain layer marked as output.

An example of engine building which starts on GPU and transitions to DLA on layer 38:
```bash
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--start gpu \
--output build/googlenet_transition_plans/googlenet_gpu_transition_at_38.plan \
--transition 38 \
--verbose
```

An example of engine building which starts on DLA and marks the output as layer 38:

```bash
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--start dla \
--output build/googlenet_mark_plans/googlenet_mark_at_38.plan \
--mark 38 \
--verbose
```

2. Profile and Log Generation

Similar to other analyses, the engines can be executed as such:
```bash
/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile \
--exportProfile=build/googlenet_transition_plans/profiles/googlenet_gpu_transition_at_38.profile --avgRuns=1  \
--warmUp=5000 --duration=0 --loadEngine=build/googlenet_transition_plans/googlenet_gpu_transition_at_38.plan > build/googlenet_transition_plans/profile_logs/googlenet_gpu_transition_at_38.log
```
The commands produce both log and profile files which are useful depending on the analysis. 

> `+` sign demonstrates the layers are merged. `||` demonstrates outputs of the layers will be concataned (as concatanation layer). `{}` demonstrates that DLA fuses the layers and profiling of all layers are treated as one layer(basically, this is a profiling limitation in DLA architectures). 

The generated profile is useful for layer analysis of GPU however the profile needs to be filtered from unnecessary information. Also, the fused layers do not provide an easily accessible layer count so we provided our own parsing and safe filtering for this profile.
To generate filtered layer timing information in Json:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 scripts/layer_analysis/layer_gpu_util.py --profile build/googlenet_transition_plans/profiles/googlenet_gpu_transition_at_-1.profile
```
The filtered output can be found here `build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json`.

Similarly in DLA the layer executions are analyzed from the profile. Due to inaccurate reading from the TensorRT platform a different method is utilized. Total layer execution time of transitioning engines are compared to find per layer group execution time. In addition, for increased accuracy, the transition cost which will be calculated in the transition analyses is added.

3. Results Compilation

To process the filtered json for the GPU run the script below.

```
python3 scripts/layer_analysis/layer_all_util.py --gpu_json build/googlenet_transition_plans/layer_times/googlenet_gpu_transition_at_-1_filtered.json --dla_json output/dla_compute_times.json
```

GPU results can be seen here.

``` 
> cat output/layer_results.json
{
    "0-9": {
        "gpu": {
            "total_gpu_time_ms": 0.3548035,
            "layer_count": 10
        }
    },
    "10-24": {
        "gpu": {
            "total_gpu_time_ms": 0.15014499,
            "layer_count": 15
        },
	...
```
Note: This output represents the numbers we have represented GPU column of Table 2 in our paper. The numbers do not match exactly for many reasons. First, results in different versions might be inconsistent (check  "Difference among results" subsection below). We also did long warmup periods and 1000 iterations to be consistent in our process. The pattern among the numbers are very similar to the ones in our paper. The output here better since we are fully utilizing the device at the maximum power mode (MAXN) unlike the default power mode. We have observed that this mode provides a better utilization out of the system. It is important to note that improvement on layer's profilings are better for each layer groups even though the execution time ratio among layers are similar to each other in this artifact and the actual results in the paper.

#### Difference among results

The results reported here may vary depending on the device and version. A clear demonstration of version difference is given below in the examples. The same DNN (GoogleNet) execution of Xavier AGX and AGX Orin are reported differently. The clear demonstration of the output logs is given below. The logs include a partial log file reporting the execution time of a whole DNN and layers in GoogleNet. An important point to note here is that "prob" layer (last layer of GoogleNet) can't run on DLA. So, prob is assigned to GPU whereas all other layers on DLA. We have listed some inconsistencies in this particular example below.

1- "data to nvm" represents the data copy time to SRAM in DLA. Yet, not reported in the Orin AGX (even though it is actually happening)
2- "all layers execution time" takes 0.2 (5.4%) of all time in Xavier, which does not seem correct.
3- "input reformatter" takes 3.33ms, which is 92.1% of all time.
4- "loss3/classifier" last layer on DLA reported as 0.00 ms to finish (due to the loss of precision)  
5- "data copy finish", "loss3/classifier finish" and "data to nvm" are not reported in Orin.

Different DNNs may generate more inconsistencies among devices/versions etc. This artifact does not list any of these since we hope that each upcoming version of TensorRT will solve these. Yet, we would like to point out that the experiments in the paper may not generate the same results as long as all requirements (software version /hardware details etc.) are exactly the same. The crucial point we would like to emphasize is to be consistent while collecting layer-wise input data to be used in the solver.

```bash
##OUTPUT FROM Xavier AGX JetPack 4.5.1 (including TensorRT 7.1.3)

[I] GPU Compute
[I] min: 3.60278 ms
[I] max: 4.14014 ms
[I] mean: 3.68068 ms
[I] median: 3.66748 ms
[I] percentile: 3.87695 ms at 99%
[I] total compute time: 3.68068 s
[I] 
[I] === Profile (1514 iterations ) ===
[I]                           Layer   Time (ms)   Avg. Time (ms)   Time %
[I]                     data to nvm       68.17             0.05      1.2
[I]  {conv1/7x7_s2,conv1/relu_7x7,pool1/3x3_s2,pool1/norm1,conv2/3x3_reduce,conv2/relu_3x3_reduce,conv2/3x3,conv2/relu_3x3,conv2/norm2,pool2/3x3_s2,inception_3a/1x1,inception_3a/relu_1x1,inception_3a/3x3_reduce,inception_3a/relu_3x3_reduce,inception_3a/3x3,inception_3a/relu_3x3,inception_3a/5x5_reduce,inception_3a/relu_5x5_reduce,inception_3a/5x5,inception_3a/relu_5x5,inception_3a/pool,inception_3a/pool_proj,inception_3a/relu_pool_proj,inception_3a/output,inception_3b/1x1,inception_3b/relu_1x1,inception_3b/3x3_reduce,inception_3b/relu_3x3_reduce,inception_3b/3x3,inception_3b/relu_3x3,inception_3b/5x5_reduce,inception_3b/relu_5x5_reduce,inception_3b/5x5,inception_3b/relu_5x5,inception_3b/pool,inception_3b/pool_proj,inception_3b/relu_pool_proj,inception_3b/output,pool3/3x3_s2,inception_4a/1x1,inception_4a/relu_1x1,inception_4a/3x3_reduce,inception_4a/relu_3x3_reduce,inception_4a/3x3,inception_4a/relu_3x3,inception_4a/5x5_reduce,inception_4a/relu_5x5_reduce,inception_4a/5x5,inception_4a/relu_5x5,inception_4a/pool,inception_4a/pool_proj,inception_4a/relu_pool_proj,inception_4a/output,inception_4b/1x1,inception_4b/relu_1x1,inception_4b/3x3_reduce,inception_4b/relu_3x3_reduce,inception_4b/3x3,inception_4b/relu_3x3,inception_4b/5x5_reduce,inception_4b/relu_5x5_reduce,inception_4b/5x5,inception_4b/relu_5x5,inception_4b/pool,inception_4b/pool_proj,inception_4b/relu_pool_proj,inception_4b/output,inception_4c/1x1,inception_4c/relu_1x1,inception_4c/3x3_reduce,inception_4c/relu_3x3_reduce,inception_4c/3x3,inception_4c/relu_3x3,inception_4c/5x5_reduce,inception_4c/relu_5x5_reduce,inception_4c/5x5,inception_4c/relu_5x5,inception_4c/pool,inception_4c/pool_proj,inception_4c/relu_pool_proj,inception_4c/output,inception_4d/1x1,inception_4d/relu_1x1,inception_4d/3x3_reduce,inception_4d/relu_3x3_reduce,inception_4d/3x3,inception_4d/relu_3x3,inception_4d/5x5_reduce,inception_4d/relu_5x5_reduce,inception_4d/5x5,inception_4d/relu_5x5,inception_4d/pool,inception_4d/pool_proj,inception_4d/relu_pool_proj,inception_4d/output,inception_4e/1x1,inception_4e/relu_1x1,inception_4e/3x3_reduce,inception_4e/relu_3x3_reduce,inception_4e/3x3,inception_4e/relu_3x3,inception_4e/5x5_reduce,inception_4e/relu_5x5_reduce,inception_4e/5x5,inception_4e/relu_5x5,inception_4e/pool,inception_4e/pool_proj,inception_4e/relu_pool_proj,inception_4e/output,pool4/3x3_s2,inception_5a/1x1,inception_5a/relu_1x1,inception_5a/3x3_reduce,inception_5a/relu_3x3_reduce,inception_5a/3x3,inception_5a/relu_3x3,inception_5a/5x5_reduce,inception_5a/relu_5x5_reduce,inception_5a/5x5,inception_5a/relu_5x5,inception_5a/pool,inception_5a/pool_proj,inception_5a/relu_pool_proj,inception_5a/output,inception_5b/1x1,inception_5b/relu_1x1,inception_5b/3x3_reduce,inception_5b/relu_3x3_reduce,inception_5b/3x3,inception_5b/relu_3x3,inception_5b/5x5_reduce,inception_5b/relu_5x5_reduce,inception_5b/5x5,inception_5b/relu_5x5,inception_5b/pool,inception_5b/pool_proj,inception_5b/relu_pool_proj,inception_5b/output,pool5/7x7_s1,loss3/classifier}      298.26             0.20      5.4
[I]                data copy finish       49.31             0.03      0.9
[I]        prob input reformatter 0     5046.37             3.33     92.1
[I]         loss3/classifier finish        4.58             0.00      0.1
[I]                            prob       11.27             0.01      0.2
[I]                           Total     5477.95             3.62    100.0
[I] 
```
```bash
##OUTPUT FROM AGX Orin JetPack 5.1.1 (including TensorRT 8.5.2)
[11/29/2023-16:19:02] [I] 
[11/29/2023-16:19:02] [I] === Profile (4932 iterations ) ===
[11/29/2023-16:19:02] [I]                                                                                       Layer   Time (ms)   Avg. Time (ms)   Median Time (ms)   Time %
[11/29/2023-16:19:02] [I]  Reformatting CopyNode for Input Tensor 0 to {ForeignNode[conv1/7x7_s2...loss3/classifier]}      202.26           0.0410             0.0379      3.7
[11/29/2023-16:19:02] [I]                                              {ForeignNode[conv1/7x7_s2...loss3/classifier]}     5089.98           1.0320             1.0321     93.1
[11/29/2023-16:19:02] [I]                                            Reformatting CopyNode for Input Tensor 0 to prob       85.65           0.0174             0.0162      1.6
[11/29/2023-16:19:02] [I]                                                                                        prob       91.72           0.0186             0.0170      1.7
[11/29/2023-16:19:02] [I]                                                                                       Total     5469.61           1.1090             1.1033    100.0
[11/29/2023-16:19:02] [I] 
```



### Step 3: Transition time profiling: 

#### Summary

This step explains how transition times are collect. Summary of comprehensive profiling result can be obtained by running the command below:

```bash
make layer
python3 scripts/transition_time_analysis/transition_util.py
cat output/transition_results.json
# Summary of last lines given below:
# ...
# "googlenet_gpu_mark_at_124": {
#     "mean_time": 1.98731,
#     "transition_cost": 0.0245
# }

#Expected output given below in detail for further comparison
cat output_expected/transition_results.json
```

#### Details

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

To build all necessary engines, measure their transition costs and save the output run the following. You can view the final results in the `output/emc_results.json` file. The command below gives overall experiments here.

```bash
make emc
cat output/transition_results.json
```

For detailed step-by-step experiments, we follow three-step experiments.

 1. Engine File Generation:
 The build_engine.py script is used to generate engine files for both GPU and DLA executions based on the GoogleNet model defined in the Prototxt file.
 Two sets of engine files with different transition layers are generated at this step. We give a transition at layer 24 below:

Makefile generates all the engines in every transition layer. Example builds for single engine:
```bash
python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output build/googlenet_transition_plans/googlenet_gpu_transition_at_24.plan \
--start gpu \
--mark 24 \
--verbose

python3 src/build_engine.py \
--prototxt prototxt_input_files/googlenet.prototxt \
--output build/googlenet_transition_plans/googlenet_dla_transition_at_24.plan \
--start dla \
--mark 24 \
--verbose
```

 2. Profile and Log Generation:
    Using trtexec, the model is run with each engine file, and detailed performance profiles are collected.
    These profiles are saved as intermediate files in TR_TIME_PROFILES_DIR, accompanied by logs in TR_TIME_PROF_LOGS_DIR. We give a 

```bash
/usr/src/tensorrt/bin/trtexec --iterations=10000  --dumpProfile --exportProfile=build/googlenet_transition_plans/profiles/googlenet_gpu_transition_at_0.profile \
--avgRuns=1 --warmUp=5000 --duration=0 --loadEngine=build/googlenet_transition_plans/googlenet_gpu_transition_at_24.plan > build/googlenet_transition_plans/profile_logs/googlenet_gpu_transition_at_24.log
```
The transition analysis makes use of mean compute values. You can view the logs to see the mean values:
```bash
> cat build/googlenet_transition_plans/profile_logs/googlenet_gpu_transition_at_24.log | grep -C 4 mean
[12/27/2023-21:42:32] [I] Average on 1 runs - GPU latency: 1.95703 ms - Host latency: 2.00195 ms (end to end 2.00781 ms, enqueue 1.96484 ms)
[12/27/2023-21:42:32] [I] Host Latency
[12/27/2023-21:42:32] [I] min: 1.97266 ms (end to end 1.97852 ms)
[12/27/2023-21:42:32] [I] max: 2.20117 ms (end to end 2.21094 ms)
[12/27/2023-21:42:32] [I] mean: 2.00712 ms (end to end 2.0141 ms)
[12/27/2023-21:42:32] [I] median: 2.00586 ms (end to end 2.01367 ms)
[12/27/2023-21:42:32] [I] percentile: 2.03125 ms at 99% (end to end 2.03906 ms at 99%)
[12/27/2023-21:42:32] [I] throughput: 496.392 qps
[12/27/2023-21:42:32] [I] walltime: 20.1454 s
--
[12/27/2023-21:42:32] [I] median: 1.9707 ms
[12/27/2023-21:42:32] [I] GPU Compute
[12/27/2023-21:42:32] [I] min: 1.92773 ms
[12/27/2023-21:42:32] [I] max: 2.14062 ms
[12/27/2023-21:42:32] [I] mean: 1.96148 ms
[12/27/2023-21:42:32] [I] median: 1.96094 ms
[12/27/2023-21:42:32] [I] percentile: 1.98242 ms at 99%
[12/27/2023-21:42:32] [I] total compute time: 19.6148 s
[12/27/2023-21:42:32] [I] 
```

3. Results Compilation

The final python script parses the mean values, processes the difference between baseline value and compiles all of the data into a single json:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 scripts/transition_time_analysis/transition_util.py
```

You can view the transition cost analysis results in `output/transition_results.json`


```bash
> cat output/transition_results.json                         
{
    "googlenet_dla_mark_at_-1": {
        "mean_time": 3.64919,
        "transition_cost": 0.0
    },
    "googlenet_dla_mark_at_10": {
        "mean_time": 3.718,
        "transition_cost": 0.06881
    },
    "googlenet_dla_mark_at_24": {
        "mean_time": 3.73284,
        "transition_cost": 0.08365
.
.
.
.
    },
    "googlenet_gpu_mark_at_-1": {
        "mean_time": 1.96281,
        "transition_cost": 0.0
    },
    "googlenet_gpu_mark_at_10": {
        "mean_time": 1.98737,
        "transition_cost": 0.02456
    },
    "googlenet_gpu_mark_at_24": {
        "mean_time": 2.00455,
        "transition_cost": 0.04174
```
Note: We list the transition costs here as similar to Table 2 transition cost column. In the submitted version, we have included only GPU transition cost whereas DLA transition cost will be also added in the final version.


### Step 4: EMC Analysis 

#### Summary

This step explains how emc utilizations are collected. Summary of comprehensive profiling result can be obtained by running the command below:

```bash
make emc
cat output/emc_results.json
# Summary of last lines given below:
# {
#    "conv1": {
#        "kernel1": "89%",
#    ...
# }

#Expected output given below in detail for further comparison
cat output_expected/emc_results.json
```

#### Details

Input Files:

 - Prototxt Files: Located in `PROTOTXT_DIR` (`convolution_characterization_prototxts` directory). These files describe the convolution layer configurations, including input sizes and filter (kernel) sizes.

Intermediate Files:

 - Engine Plan Files: Located in `EMC_PLANS_DIR` (`build/convolution_characterization_plans` directory). These are the TensorRT engine files (.plan) that are generated from the input Prototxt files. Each engine file represents a specific convolution layer configuration and is used to measure EMC utilization.
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
 Note: this command requires sudo privilege.
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
 export PYTHONPATH="$(pwd):$PYTHONPATH"
 python3 scripts/emc_analysis/emc_util_all.py
 ```
 View the output: 
 ```bash
 > cat output/emc_results.json
{
    "conv1": {
        "kernel1": "89%",
    ...
 ```

Reference from paper: The outputs in Json file demonstrates very similar pattern in the Figure 3. Input and filter size for a convolution layer is illustrated in Figure 3 and encoded into json file in this artifact(output/emc_results.json). Besides, using maxn power mode helped us to get more EMC utilization (5-10% average increase)

### Step 5: Memory Throughput Profiling

This step targets to profile memory throughput.  This code profiles the memory throughput in kernel level(Nsight compute automatically converts layers to kernels).

```bash
mkdir nsight_compute_logs
# This prompt requests sudo privilege. Takes a couple of minutes to run
python3 src/nsight_compute.py
```

Note: If you face any "Error opening engine file: starter_guide_logs/googlenet_only_gpu.plan" error, please build the only googlenet plan  by running this: "python3 starter_guide_experiment.py"

This code outputs a nsight_compute_$DNN_.report. TensorRT has its own naming and output report structure that becomes very complicated. This requires a lot of effort to match the layers and their instructions. We leave this script here for a reference.

More in detail, we suggest to follow this strategy explained as a summary below. This gives the memory throughput of a layer. We use a recorded memory throughput data in z3 solvers below. Memory throughput of this layer groups are represented in the last column of Table 2. Since we are targetting to assign at the layer at the layers by using groups, we are calculating the cumulative average as given below.

To calculate a memory throughput of a layer group:  (memory throughput of a layer) * (duration of a layer in the group) / (duration of all layers in the group).   

We can't profile memory throughput of DLA since Nsight Compute does not allow to use DLA(targets only GPU). So, we convert the profiling data we have obtained in EMC analysis by using each layer group's EMC GPU and DLA utilization. The formulation given below. EMC data is obtained as explained above and memory throughput is profiled via Nsight Compute.

DLA's memory throughput for a layer X: (EMC utilization of layer X on DLA / EMC utilization of layer X on GPU) * memory throughput of layer X on GPU


### Step 6: Synchronous multiple DNN execution

So far. we have run DNNs standalone to collect layer profiling. This section enables to run DNNs simultaneously  This session enables to synchronously run DNNs briefly explained in "Neural network synchronization" in Session 4.

* create two distinct copies of the original Tensorrt directory to an empty directories
* replace sampleInference.cpp with the corresponding directories
* build the directories & write 0 to a tmp shared file.
* build googlenet only gpu and dla engines
* run the multiple DNN
```bash
chmod +x sync_multi_dnn_exec.sh
./sync_multi_dnn_exec.sh

# Expected Output
# [12/28/2023-06:36:24] [I] mean: 4.67762 ms # this is DLA exec
# [12/28/2023-06:36:21] [I] mean: 3.30332 ms # this is GPU exec

```

### Step 7: Z3 Solver execution

Until now, we followed the steps on how to profile and execute DNNs. We profiled execution time of layers, transition time and memory throughput of layers. All these data are used as an input, i.e., nn_times_acc, nn_trans_acc, and nn_slowdown_acc.

> Slowdown values are found by [PCCS](https://dl.acm.org/doi/abs/10.1145/3466752.3480101) and their [Source Code](https://github.com/processorcentricmodel/PCCS). If a user targets to use a different environment, the model needs to be reconstructed. We follow the steps defined by the authors.

The code implements the model given in Section 3.5. To run the model:

```bash
python3 src/z3_solver_multi_dnn.py > output/schedule_summary.txt
```

The expected output from the solver is the schedule of DNNs.

```bash
> cat output/schedule_summary.txt
GoogleNet starts on GPU
GoogleNet applies transition at layer 52
AlexNet starts on DLA
AlexNet applies transition at layer 2
```

For other DNNs execution scenarios, previous steps needs to be followed and the profile data needs to be given the corresponding files.


### Step 8: Multi DNN, HaX-CoNN results

After collecting profile data, we use z3 solver to find the corresponding schedule. Now, to verify the performance of our model, we will execute the corresponding schedules.

For the baselines, we generate GPU executions, GPU&DLA executions, and [H2H](https://dl.acm.org/doi/10.1145/3489517.3530509) and [Herald](https://www.computer.org/csdl/proceedings-article/hpca/2021/223500a071/1t0HUXqfspW). We follow their implementation as given in [source code](https://github.com/xyzxinyizhang/H2H). For Herald, we stop the execution at step 2 (as H2H did in their methodology) and for H2H baseline, we follow the execution for all steps (including step 4). 

For the sake of simplicity of the artifact, we provide the schedules previously found in our python file. We build the engines, run them and compare the results. To run the experiment, run the script below (Summary of experiments are printed at the last step):

```bash
chmod +x baseline_engine_building.sh
chmod +x collect_data_multidnn_experiment.sh
./baseline_engine_building.sh
./collect_data_multidnn_experiment.sh
python3 src/summarize_multi_dnn_executions.py

```

Summary of experiments prints out the baseline values per baseline, HaX-CoNN value and the improvement over the best baseline. A short output given below and the real execution prints for each experiment design. 
```bash
# Summary of Exp3. Alexnet Resnet101
# Only GPU: 13.4 ms
# GPU&DLA: 10.4 ms
# Herald: 11.5 ms
# H2H: 11.4 ms
# HaX-CoNN: 8.4 ms
# Overall improvement over best-baseline: 23.87
# This is claimed in the paper as 26%.

#Expected output given below in detail for further comparison
cat output_expected/transition_results.json
```


#### Reusability on Orin AGX
For a demonstrationg of reusability of our setup, we target to run the same DNNs. We have to use TensorRT to be able to use DLA. However, TensorRT version is 8.5 in Orin AGX (unlike Xavier AGX which has 7.1.3).
Moreover, in order to add another dimension for reusability, we proposa to use INT8 setting, whereas we use FP16 in Xavier AGX settings. Even though these changes seems easy enough, using a new devices with a different setting could be a good candidate for the reusability bagde. [A recent paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9638444) titled as "Performance Evaluation of INT8 Quantized Inference on Mobile GPUs" evaluates and characterizes the importance of difference between INT8 and FP16. 

For the sake of simplicity during reviewing process, profiling data has been preprocessed and the functions are adapted for new TensorRT version. (There has been changes in the function name in TensorRT's API). Plus, we have integrated the calibration files for DNNs from FP16 to INT8. 


Follow the steps to build tensorrt binary files. Since the version is different, sampleInference is given in a different file (sampleInference$Number_orinagx.cpp) Also, building engine file is different too(src/build_engine_orin.py). For the sake of easiness of the reviewer, we add the commands to run below for Orin AGX.

To run the experiments, please login Orin AGX(credentials given in desktop)

```bash
chmod +x orin_sync_multi_dnn_exec.sh
./orin_sync_multi_dnn_exec.sh
```

Expected output given below:
```bash
Summary of Exp1. GoogleNet and ResNet
Average time of using only GPU: 2.42
Average time of Resnet101 on DLA and Googlenet on GPU: 2.99
Average time of the schedule found by Herald: 2.17
Average time of the schedule found by H2H: 2.06
Average time of the schedule found by HaX-CoNN: 1.93
Overall improvement over best-baseline: 6.8%
```

Note: These values on Orin AGX may vary what we have reported in the paper (Table 6). The authors believe that INT8 and FP16 performances of devices are significantly different. Plus, another reason for the variation is caused by the different Jetpack settings. We have used Jetpack 5.0.1 as reported in Table 4 and it brings with TensorRT 8.4. The target system for our experiments should have (or have if you can remotely access) 5.1.1, which has TensorRT 8.5.2.

Note2: As can be observed by comparing Xavier AGX and Orin AGX, the improvement over baselines may vary depending on the characteristics of the hardware(target device) and the DNN set(target applications).


### Step 9: Overhead Analysis


Before starting make sure:
```bash
chmod +x scripts/run_all_plan.sh scripts/run_forever.sh
mkdir -p build/overhead_gpu/logs build/overhead_dla/alexnet_dla
```
- Build AlexNet DLA engine.

```bash
python3 src/build_engine.py --prototxt prototxt_input_files/alexnet.prototxt --output build/overhead_dla/alexnet_dla.plan --start dla
```

- Build GPU engines for DenseNet GoogleNet Inc-res-v2 Inception MobileNet ResNet18 ResNet50 ResNet101 ResNet152 VGG16 VGG19

```bash
python3 src/build_engine.py --prototxt prototxt_input_files/ --output build/overhead_gpu --start gpu
```

- Run AlexNet DLA with each GPU engines. Collect data for each execution. it should include the execution of 121 DNNs Example: AlexnetDLA-DenseNetGPU, AlexnetDLA-GoogleNetGPU, AlexNet-InceptionGPU etc.
    
```bash
./scripts/run_all_plan.sh build/overhead_gpu
grep -r "                            Total   " alexnet_dla_*.log >> without_contention_alexnet_dla_time.log

#IMPORTANT COMMENT: Open two different command lines. 
#In the first, run run_forever script that constantly searches for the schedules by using z3.
./scripts/run_forever.sh
#In the second, run DNN pairs similar to above.
./scripts/run_all_plan.sh build/overhead_gpu 
grep -r "                            Total   " alexnet_dla_*.log >> with_contention_alexnet_dla_time.log
```

The log files might give inconsistent output results where parsing might cause an error. Instead of throwing an error, we leave the naive comparison of the results between without contention log and with contention log. The outputs are also given in Table 7 of the paper.