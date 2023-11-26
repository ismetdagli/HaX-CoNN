import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
# from natsort import natsorted
import time


print("Code started")
avgRun = 1
count = 0
warmUp = 0
iteration = 1000


def process_run(
    network_name,
    batch,
    warmUp,
    iteration,
    engine,
    output_dir,
    threadno,
    other_network,
    TRTEXEC_PATH,
    count,
):
    dt = datetime.now()
    print("iter:", dt)
    with open(
        output_dir
        + "/"
        + str(network_name)
        + "_"
        + str(other_network)
        + "_"
        + str(threadno)
        + "_results.log",
        "w",
    ) as log_file:
        subprocess.run(
            [
                TRTEXEC_PATH,
                "--iterations=" + str(iteration),
                "--dumpProfile",
                "--batch=" + str(batch),
                "--avgRuns=" + str(avgRun),
                "--warmUp=" + str(warmUp),
                "--duration=0",
                f"--loadEngine={engine}",
            ],
            stdout=log_file,
        )
        # subprocess.run(["nsys","profile","--accelerator-trace=nvmedia","--trace=cuda,opengl,nvtx,nvmedia","--process-scope=system-wide", TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile", "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", f"--loadEngine={engine}"], stdout=log_file)


test_network = Path(str("google_only_gpu.plan"))
test_network_2 = Path(str("google_only_dla.plan"))
network_name = test_network.stem
network_name_2 = test_network_2.stem
print("Network1:", network_name)
print("Network2:", network_name_2)
print("--------------------")
batch1 = 1
batch2 = 1
t1 = threading.Thread(
    target=process_run,
    args=(
        network_name,
        batch1,
        warmUp,
        iteration,
        "google_only_gpu.plan",
        "multi_dnn_execution_logs",
        1,
        network_name_2,
        "tensorrt_sharedMem1/bin/trtexec",
        count,
    ),
)
t2 = threading.Thread(
    target=process_run,
    args=(
        network_name_2,
        batch2,
        warmUp,
        iteration - 2,
        "google_only_dla.plan",
        "multi_dnn_execution_logs",
        2,
        network_name,
        "tensorrt_sharedMem2/bin/trtexec",
        count,
    ),
)
t1.start()
time.sleep(1)
t2.start()
t1.join()
t2.join()
