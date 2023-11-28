import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import time

TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"
input_dir_path="all_gpu_dla_engines/"
input_engines = glob.glob(f"{input_dir_path}/*.engine")
privilege="sudo"
NSIGHT_COMPUTE_PATH ="/home/ismetdagli/nsight-compute-tools/target/linux-v4l_l4t-t210-a64/nv-nsight-cu-cli"
section= "full"
avgRun=1
warmUp=0
iteration=1

engine="starter_guide_logs/googlenet_only_gpu.plan"

        
test_network = Path(str(engine))
network_name = test_network.stem
with open("nsight_compute_logs/"+str(network_name)+"_fullset_profiling.logs", 'w') as log_file:
    dt = datetime.now()
    current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
    print("Execution starts, TIME: ",current_time)
    subprocess.run([privilege, NSIGHT_COMPUTE_PATH, "--set="+str(section),"-f", "-o nsight_compute_"+str(network_name)+str("_set_full.report"), TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile", "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", f"--loadEngine={engine}"], stdout=log_file)


