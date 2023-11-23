import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import time


output_dir = "group_layers_gpu_logs"
input_dir_path = "group_layers_gpu/"
input_engines = glob.glob(f"{input_dir_path}/*.plan")
input_engines = natsorted(input_engines)

output_dir_2 = "group_layers_dla_logs"
input_dir_path_2 = "group_layers_dla/"
input_engines_2 = glob.glob(f"{input_dir_path_2}/*.plan")
input_engines_2 = natsorted(input_engines_2)

avgRun = 1
print("Code started")
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


engine_iter_1 = 0
engine_iter_2 = 0
for engine in input_engines:
    engine_iter_1 += 1
    for engine_2 in input_engines_2:
        engine_iter_2 += 1
        count += 1
        test_network = Path(str(engine))
        test_network_2 = Path(str(engine_2))
        # print("test_network:",test_network)
        # print("test_network2:",test_network_2)
        network_name = test_network.stem
        network_name_2 = test_network_2.stem
        # if ("gpu" in network_name) and ("gpu" in network_name_2):
        #     continue
        # if ("dla" in network_name) and ("dla" in network_name_2):
        #     continue
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
                engine,
                output_dir,
                1,
                network_name_2,
                "/home/ismetdagli/tensorrt_sharedMem1/bin/trtexec",
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
                engine_2,
                output_dir_2,
                2,
                network_name,
                "/home/ismetdagli/tensorrt_sharedMem2/bin/trtexec",
                count,
            ),
        )
        t1.start()
        time.sleep(1)  #
        t2.start()
        t1.join()
        t2.join()
    engine_iter_2 = 0
engine_iter_1 = 0
