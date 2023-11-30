#Serial execution summary

file1 = open('baseline_singlednn_engine_logs/mean_results_of_executions.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character

def index_mean(line):
    index_of_mean= line.index('mean')
    return float(line[index_of_mean+6:index_of_mean+11])


#--------------------------EXP1--------------------------


print("Summary of Exp1. Resnet152")
resnet1_average_time=0
resnet2_average_time=0
for line in Lines:
    if "resnet152_gpu_only1_resnet152_gpu_only" in line:
        resnet1_average_time = index_mean(line)
    
    if "resnet152_gpu_only_resnet152_gpu_only1" in line:
        resnet2_average_time = index_mean(line)
only_gpu_exec_time=max(resnet1_average_time,resnet2_average_time)

print("Average FPS of using only GPU:", round(only_gpu_exec_time,1) )


for line in Lines:
    if "resnet152_gpu_only_resnet152_dla_only" in line:
        resnet1_average_time = index_mean(line)
    
    if "resnet152_dla_only_resnet152_gpu_only" in line:
        resnet2_average_time = index_mean(line)
gpu_dla_exec_time=max(resnet1_average_time,resnet2_average_time)
print("Average FPS of ResNet152 on GPU and DLA:",round(gpu_dla_exec_time,1) )




for line in Lines:
    if "resnet152_dla_transition_at_165_resnet152_gpu_transition_at_364" in line:
        resnet1_average_time = index_mean(line)
    
    if "resnet152_gpu_transition_at_364_resnet152_dla_transition_at_165" in line:
        resnet2_average_time = index_mean(line)
mensa_exec=max(resnet1_average_time,resnet2_average_time)

print("Average FPS of the schedule found by Mensa:",round(mensa_exec,1))



for line in Lines:
    if "resnet152_dla_transition_at_636_resnet152_gpu_transition_at_165" in line:
        resnet1_average_time = index_mean(line)
    
    if "resnet152_gpu_transition_at_165_resnet152_dla_transition_at_636" in line:
        resnet2_average_time = index_mean(line)
hax_conn_exec=max(resnet1_average_time,resnet2_average_time)

print("Average FPS of the schedule found by HaX-CoNN:",round(hax_conn_exec,1))

print("Overall improvement over best-baseline: "+str(round((hax_conn_exec/max(only_gpu_exec_time,gpu_dla_exec_time,mensa_exec)-1)*100,2))+"%\n")








