#Serial execution summary

file1 = open('starter_guide_logs/mean_results_of_executions.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character


def index_mean(line):
    index_of_mean= line.index('mean')
    return float(line[index_of_mean+6:index_of_mean+11])

googlenet_average_time=0
resnet_average_time=0
for line in Lines:
    if "googlenet_only_gpu_resnet101_only_gpu" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_gpu_googlenet_only_gpu" in line:
        resnet_average_time = index_mean(line)
exec_time=googlenet_average_time+resnet_average_time
print("Average time of using only GPU: ", exec_time )


for line in Lines:
    if "googlenet_only_dla_resnet101_only_gpu" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_gpu_googlenet_only_dla" in line:
        resnet_average_time = index_mean(line)

exec_time2=googlenet_average_time+resnet_average_time
print("Average time of Resnet101 on GPU and Googlenet on DLA: ",exec_time2 )


for line in Lines:
    if "googlenet_only_gpu_resnet101_only_dla" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_dla_googlenet_only_gpu" in line:
        resnet_average_time = index_mean(line)

exec_time3=googlenet_average_time+resnet_average_time
print("Average time of Resnet101 on DLA and Googlenet on GPU: ", exec_time3 )
    


for line in Lines:
    if "googlenet_gpu_transition_at_80_resnet101_dla_transition_at_24" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_dla_transition_at_24_googlenet_gpu_transition_at_80" in line:
        resnet_average_time = index_mean(line)
hax_conn_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by HaX-CoNN: ",hax_conn_exec )

print("Overall improvement over best-baseline: ", (min(exec_time,exec_time2,exec_time3)/hax_conn_exec-1)*100,"%")



