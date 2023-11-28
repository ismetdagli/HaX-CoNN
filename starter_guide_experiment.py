import subprocess
import threading
from datetime import datetime
from pathlib import Path
import time


print("starter_guide_experiment.py started")
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

def Run_plans(first_dnn_plan,second_dnn_plan):
    test_network = Path(str(first_dnn_plan))
    test_network_2 = Path(str(second_dnn_plan))
    network_name = test_network.stem
    network_name_2 = test_network_2.stem
    print("Running DNN1:", network_name, "with DNN2 :", network_name_2, " together")
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
            first_dnn_plan,
            "starter_guide_logs/",
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
            second_dnn_plan,
            "starter_guide_logs/",
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


print("Serial GPU execution starts")
Run_plans("starter_guide_logs/googlenet_only_gpu.plan","starter_guide_logs/resnet101_only_gpu.plan")

print("Resnet101 on GPU and Googlenet on DLA starts")
Run_plans("starter_guide_logs/googlenet_only_gpu.plan","starter_guide_logs/resnet101_only_dla.plan")

print("Resnet101 on DLA and Googlenet on GPU starts")
Run_plans("starter_guide_logs/googlenet_only_dla.plan","starter_guide_logs/resnet101_only_gpu.plan")

print("Schedule found by HaX-CoNN starts")
Run_plans("starter_guide_logs/googlenet_gpu_transition_at_80.plan","starter_guide_logs/resnet101_dla_transition_at_24.plan")
