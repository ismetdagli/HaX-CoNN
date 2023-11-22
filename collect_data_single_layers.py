import glob
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import time

avgRun = 1
warmUp = 2000
warmUpList = [2000]
iteration = 1000
TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"

# output_dir = "inception_dla_logs"
# input_dir_path="inception_dla/"
# input_engines = glob.glob(f"{input_dir_path}/*.plan")
# input_engines=natsorted(input_engines)

# output_dir_2 = "inception_gpu_logs"
# input_dir_path_2 ="inception_gpu/"
# input_engines_2 = glob.glob(f"{input_dir_path_2}/*.plan")
# input_engines_2=natsorted(input_engines_2)

# avgRun=1
# print("Code started")
# count=0
# # iteration=20
# warmUpList=[20000]
# iteration=1000
# def process_run(network_name,warmUp,iteration,engine,output_dir,threadno,other_network,TRTEXEC_PATH):
#     dt = datetime.now()
#     print("iter:", dt)
#     # print("TRTEXEC_PATH:",TRTEXEC_PATH )
#     # print("iteration:",iteration )
#     # print("network_name:", network_name)
#     # print("avgRun:",avgRun )
#     # print("warmUp:",warmUp )
#     # print("engine:",engine )
#     # print("threadno: ",threadno, network_name )
#     with open(output_dir+"/"+str(network_name)+"_"+str(other_network)+"_"+str(threadno)+"_results.log", 'w') as log_file:
#         # print("log_file:",log_file )
#         subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile", "--exportProfile=temp/"+str(network_name)+".profile", "--avgRuns="+str(avgRun), "--warmUp=0", "--duration=0", f"--loadEngine={engine}"], stdout=log_file)

# # print("Print engine:", input_engines)
# # print("len engine:", len(input_engines))
# # print("len engine2:", len(input_engines_2))
# # print("Print engine2:", input_engines_2)
# # exit()
# for warmUp in warmUpList:
#     for engine in input_engines:
#         for engine_2 in input_engines_2:
#             count+=1
#             # if (count < 382):
#             #     continue
#             # engine = input_engines[index]
#             # engine_2 = input_engines[index_2]
#             test_network = Path(str(engine))
#             test_network_2 = Path(str(engine_2))
#             network_name = test_network.stem
#             network_name_2 = test_network_2.stem
#             print("Network1:",network_name)
#             print("Network2:",network_name_2)
#             print("--------------------")

#             # with open(output_dir+"/"+str(network_name)+"_single_network.log", 'w') as log_file:
#             dt = datetime.now()
#             current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
#             # print("CURRENT TIME: ",current_time)
#             tempfile_1="File1"
#             tempfile_2="File2"
#             t1 = threading.Thread(target=process_run,args=(network_name,warmUp,iteration,engine,output_dir,1,network_name_2,"/home/ismetdagli/7tensorrt_sharedMem1/bin/trtexec"))
#             t2 = threading.Thread(target=process_run,args=(network_name_2,warmUp,iteration-2,engine_2,output_dir_2,2,network_name,"/home/ismetdagli/7tensorrt_sharedMem2/bin/trtexec"))
#             t1.start()
#             time.sleep(2)
#             t2.start()
#             t1.join()
#             t2.join()
#             # exit()
#                 # subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile", "--exportProfile=temp/"+str(network_name)+".profile", "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", f"--loadEngine={engine}"], stdout=log_file)
#             # if count > 0:


# input_dir_path="all_gpu_dla_engines/"
# input_dir_path_list= ["multiple_batch_engines"]
# for input_dir_path in input_dir_path_list:
#     input_engines = glob.glob(f"{input_dir_path}/*.plan")
#     input_engines = natsorted(input_engines)
#     for engine in input_engines:
#         test_network = Path(str(engine))
#         network_name = test_network.stem
#         print(network_name)
#         if "dla" in network_name:
#             with open("multiple_batch_engines_batch16_v2_logs/"+str(network_name)+"_standalone.log", 'w') as log_file:
#                 dt = datetime.now()
#                 current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
#                 print("CURRENT TIME: ",current_time)
#                 subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile","--batch=16", "--useDLACore=0", "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", "--loadEngine="+str(engine)], stdout=log_file)
#                 # exit()
#         else:
#             with open("multiple_batch_engines_batch16_v2_logs/"+str(network_name)+"_standalone.log", 'w') as log_file:
#                 dt = datetime.now()
#                 current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
#                 print("CURRENT TIME: ",current_time)
#                 subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile","--batch=16",  "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", "--loadEngine="+str(engine)], stdout=log_file)
#                 # exit()


# for input_dir_path in input_dir_path_list:
#     input_engines = glob.glob(f"{input_dir_path}/*.plan")
#     input_engines = natsorted(input_engines)
#     for engine in input_engines:
#         test_network = Path(str(engine))
#         network_name = test_network.stem
#         print(network_name)
#         if "dla" in network_name:
#             with open("multiple_batch_engines_batch1_v2_logs/"+str(network_name)+"_standalone.log", 'w') as log_file:
#                 dt = datetime.now()
#                 current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
#                 print("CURRENT TIME: ",current_time)
#                 subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile","--batch=1", "--useDLACore=0", "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", "--loadEngine="+str(engine)], stdout=log_file)
#                 # exit()
#         else:
#             with open("multiple_batch_engines_batch1_v2_logs/"+str(network_name)+"_standalone.log", 'w') as log_file:
#                 dt = datetime.now()
#                 current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
#                 print("CURRENT TIME: ",current_time)
#                 subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile","--batch=1",  "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", "--loadEngine="+str(engine)], stdout=log_file)
#                 # exit()

TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"
input_dir_path = "group_layers_dla/"
input_engines = glob.glob(f"{input_dir_path}/*.plan")

avgRun = 1
print("test")
count = 0
# iteration=20
warmUpList = [0]
iteration = 1000
for warmUp in warmUpList:
    for engine in input_engines:
        if "vgg" not in engine:
            continue
        count += 1
        test_network = Path(str(engine))
        # engine="resnet101_15_.plan"
        print(engine)
        network_name = test_network.stem
        with open(
            "group_layers_gpu_and_dla_standalone/"
            + str(network_name)
            + "_standalone.log",
            "w",
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

# TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"

# output_dir = "multiple_network_attempts"
# input_dir_path="resnet101_dla/"
# input_engines = glob.glob(f"{input_dir_path}/*.plan")

# output_dir_2 = "multiple_network_attempts"
# input_dir_path_2 ="resnet101_gpu/"
# input_engines_2 = glob.glob(f"{input_dir_path}/*.plan")

# avgRun=1
# print("Code started")
# count=0
# # iteration=20
# warmUpList=[1]
# iteration=1
# def process_run(network_name,warmUp,engine,output_dir,threadno):
#     dt = datetime.now()
#     print("iter:", dt)
#     print("TRTEXEC_PATH:",TRTEXEC_PATH )
#     print("iteration:",iteration )
#     print("network_name:", network_name)
#     print("avgRun:",avgRun )
#     print("warmUp:",warmUp )
#     print("engine:",engine )
#     print("threadno: ",threadno, network_name )
#     with open(output_dir+"/"+str(network_name)+"_"+str(threadno)+"_Gpu_Dla_Exec.log", 'w') as log_file:
#         print("log_file:",log_file )
#         subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile", "--exportProfile=temp/"+str(network_name)+".profile", "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", f"--loadEngine={engine}"], stdout=log_file)


# for warmUp in warmUpList:
#     for engine in input_engines:
#         for engine in input_engines_2:
#             count+=1
#             test_network = Path(str(engine))

#             print(input_engines)
#             input_engines=natsorted(input_engines)
#             print(input_engines)
#             network_name = test_network.stem
#             with open(output_dir+"/"+str(network_name)+"_single_network.log", 'w') as log_file:
#                 dt = datetime.now()
#                 current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
#                 # print("CURRENT TIME: ",current_time)
#                 tempfile_1="File1"
#                 tempfile_2="File2"
#                 t1 = threading.Thread(target=process_run,args=(network_name,warmUp,"convolution_test_plans/conv1_gpu.plan",output_dir,1))
#                 t2 = threading.Thread(target=process_run,args=(network_name,warmUp,"convolution_test_plans/conv1_dla.plan",output_dir,2))
#                 t1.start()
#                 t2.start()
#                 t1.join()
#                 t2.join()
#                 # subprocess.run([TRTEXEC_PATH, "--iterations="+str(iteration),"--dumpProfile", "--exportProfile=temp/"+str(network_name)+".profile", "--avgRuns="+str(avgRun), "--warmUp="+str(warmUp), "--duration=0", f"--loadEngine={engine}"], stdout=log_file)
#             if count == 1:
#                 exit()


# input_engines=natsorted(input_engines)
# for warmUp in warmUpList:
#     for index in range(5):
#         for index_2 in range(len(input_engines)-5, len(input_engines)):
#             count+=1
#             engine = input_engines[index]
#             engine_2 = input_engines[index_2]
#             test_network = Path(str(engine))
#             test_network_2 = Path(str(engine_2))
#             network_name = test_network.stem
#             network_name_2 = test_network_2.stem
#             print("Network1:",network_name)
#             print("Network2:",network_name_2)
#             print("--------------------")
#             with open(output_dir+"/"+str(network_name)+"_single_network.log", 'w') as log_file:
#                 dt = datetime.now()
#                 current_time="Time:"+str(dt.hour)+":"+str(dt.minute)+":"+str(dt.second)+" "+str(dt.microsecond)
#                 # print("CURRENT TIME: ",current_time)
#                 tempfile_1="File1"
#                 tempfile_2="File2"
#                 t1 = threading.Thread(target=process_run,args=(network_name,warmUp,engine,output_dir,1,network_name_2))
#                 t2 = threading.Thread(target=process_run,args=(network_name_2,warmUp,engine_2,output_dir,2,network_name))
#                 t1.start()
#                 t2.start()
#                 t1.join()
#                 t2.join()
