import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import time

avgRun = 1
#warmUp can be eliminated and iteration might be less for a quick profiling, yet produces noisy data. (not suggested)
warmUp = 2000
iteration = 1000
TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"

##Update this plan if you target to run different engines/plans/DNNs
engine="starter_guide_logs/googlenet_only_gpu.plan"
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
