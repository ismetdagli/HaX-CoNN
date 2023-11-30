import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import time
import argparse
from pathlib import Path




parser = argparse.ArgumentParser()
parser.add_argument('--plan', nargs='*', type=str, help='path to plan file')
# argString = "--dummy_opt 128 128"

args = parser.parse_args()
print("this is the plan path: ", args.plan[0])
# exit()
# args = parser.parse_args(argString.split())

avgRun = 1
#warmUp can be eliminated and iteration might be less for a quick profiling, yet produces noisy data. (not suggested)
warmUp = 2000
iteration = 1000
TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"

##Update this plan if you target to run different engines/plans/DNNs
engine=args.plan[0]
test_network = Path(str(engine))
network_name = test_network.stem
print("Execution starts, DNN: ",engine)
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
    ]
)
