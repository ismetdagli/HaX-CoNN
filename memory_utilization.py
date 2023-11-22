"""
This script profiles the engines and generates a nsight compute file. 
Nsight Compute profile file(.report extension) profiles the number of reads and writes per layer.
Its worth noting that we do not use memory throughput on the .report since
the execution time overhead of profiling is high on memory compute. 
They clearly calculate the memory throughput as (Read+Write)/Time. 
We follow the same metric however, we use Time in our standalone profiling from previous studies. 
"""

import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import time

TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"
input_dir_path = "batch_norm_plans/"
input_engines = glob.glob(f"{input_dir_path}/*.plan")
privilege = "sudo"
NSIGHT_COMPUTE_PATH = "/home/ismetdagli/nsight-compute-tools/target/linux-v4l_l4t-t210-a64/nv-nsight-cu-cli"
section = "full"
avgRun = 1
count = 0
warmUpList = [0]
iteration = 2


input_engines = glob.glob(f"{input_dir_path}/*.plan")
for warmUp in warmUpList:
    for engine in input_engines:
        count += 1

        test_network = Path(str(engine))
        # engine="resnet101_15_.plan"
        print(engine)
        network_name = test_network.stem
        # if "output128" in network_name:
        with open(
            "nsight_batch_norm_logs/" + str(network_name) + "_fullset_iter2.logs", "w"
        ) as log_file:
            dt = datetime.now()
            current_time = (
                "Time:"
                + str(dt.hour)
                + ":"
                + str(dt.minute)
                + ":"
                + str(dt.second)
                + " "
                + str(dt.microsecond)
            )
            print("CURRENT TIME: ", current_time)
            subprocess.run(
                [
                    privilege,
                    NSIGHT_COMPUTE_PATH,
                    "--set=" + str(section),
                    "-f",
                    "-o nsight_compute_" + str(network_name) + str("_set_full.report"),
                    TRTEXEC_PATH,
                    "--iterations=" + str(iteration),
                    "--dumpProfile",
                    "--avgRuns=" + str(avgRun),
                    "--warmUp=" + str(warmUp),
                    "--duration=0",
                    f"--loadEngine={engine}",
                ],
                stdout=log_file,
            )
            if count == 1:
                exit()
