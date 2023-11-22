"""
This is a transition cost profiling script. 
"""
import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import time

TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"
input_dir_path = ""  ## We can give a path to engine's folders
input_engines = glob.glob(f"{input_dir_path}/*.plan")
avgRun = 1
count = 0
warmUp = 5000  # warmup time in milliseconds
iteration = 10000  # number of iterations to run
for engine in input_engines:
    count += 1
    test_network = Path(str(engine))
    print(engine)
    network_name = test_network.stem
    # Define a path
    with open(
        "VGG19_transition_logs/" + str(network_name) + "_standalone.log", "w"
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
                TRTEXEC_PATH,
                "--iterations=" + str(iteration),
                "--dumpProfile",
                "--exportProfile=temp/" + str(network_name) + ".profile",
                "--avgRuns=" + str(avgRun),
                "--warmUp=" + str(warmUp),
                "--duration=0",
                f"--loadEngine={engine}",
            ],
            stdout=log_file,
        )
    # if count == 1:
    #     exit()


## TODO_ISMET
"""
Write a script to find the mean of gpu compute (similar to this [05/31/2022-08:49:38] [I] mean: 5.82935 ms) in logs files
The baseline is transition at -1
The other layers have extra transition cost
Then, calculate the difference between transition per layer and baseline. 
Each layer's transition cost should be reported.
"""
