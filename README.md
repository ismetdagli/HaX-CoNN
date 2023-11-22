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

* Layer profiling: This creates a text file of a DNN. The line after " [I] GPU Compute" are our target data. We use mean data as the average of X number of iterations   #TODO_ISMET
```bash
python3 collect_data_single_layers.py
```
Note: `+` sign demonstrates the layers are merged. `||` demonstrates outputs of the layers will be concataned (as concatanation layer). `{}` demonstrates that DLA fuses the layers and profiling of all layers are treated as one layer(basically, this is a profiling limitation in DLA architectures).

* Transition time profiling: The easiest way to profile the layer's transition cost is to generate transition per layer engines. ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#abstract) refers to executable DNN files, we follow the same terms to prevent any confusion)
```bash
python3 build_transition_time_engines.py
```

