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