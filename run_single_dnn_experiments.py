import subprocess
import threading
from datetime import datetime
from pathlib import Path
import time
import argparse


print("Code started")
avgRun = 1
count = 0
warmUp = 0
iteration = 1000


parser = argparse.ArgumentParser()
parser.add_argument('--plan1', nargs='*', type=str, help='path to plan DNN1 file')

parser.add_argument('--plan2', nargs='*', type=str, help='path to plan DNN2 file')
# argString = "--dummy_opt 128 128"

args = parser.parse_args()
print("this is the plan1 path: ", args.plan1[0])

print("this is the plan2 path: ", args.plan2[0])


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


test_network = Path(str(args.plan1[0]))
test_network_2 = Path(str( args.plan2[0]))
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
        args.plan1[0],
        "baseline_singlednn_engine_logs",
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
        args.plan2[0],
        "baseline_singlednn_engine_logs",
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
