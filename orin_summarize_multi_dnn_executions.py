#Serial execution summary

file1 = open('orin_baseline_engines_logs/orin_mean_results_of_executions.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character

def index_mean(line):
    index_of_mean= line.index('Total')
    return float(line[index_of_mean+28:index_of_mean+33])

# for line in Lines:
#     print(line)
#     print(index_mean(line))

# exit()

print("Summary of Exp1. GoogleNet and ResNet101")
googlenet_average_time=0
resnet_average_time=0
for line in Lines:
    if "googlenet_only_gpu_resnet101_only_gpu" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_gpu_googlenet_only_gpu" in line:
        resnet_average_time = index_mean(line)
only_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of using only GPU:", round(only_gpu_exec_time,2) )



for line in Lines:
    if "googlenet_only_gpu_resnet101_only_dla" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_dla_googlenet_only_gpu" in line:
        resnet_average_time = index_mean(line)

dla_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of Resnet101 on DLA and Googlenet on GPU:", round(dla_gpu_exec_time,2) )
    


for line in Lines:
    if "googlenet_dla_transition_at_10_resnet101_gpu_transition_at_101" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_transition_at_101_googlenet_dla_transition_at_10" in line:
        resnet_average_time = index_mean(line)
herald_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by Herald:",round(herald_exec,2))


for line in Lines:
    if "googlenet_dla_transition_at_81_resnet101_gpu_transition_at_415" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_transition_at_415_googlenet_gpu_transition_at_81" in line:
        resnet_average_time = index_mean(line)
h2h_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by H2H:",round(h2h_exec,2))


for line in Lines:
    if "googlenet_dla_transition_at_38_resnet101_gpu_transition_at_312" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_transition_at_312_googlenet_dla_transition_at_38" in line:
        resnet_average_time = index_mean(line)
hax_conn_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by HaX-CoNN:",round(hax_conn_exec,2))

print("Overall improvement over best-baseline: "+str(round((min(only_gpu_exec_time,dla_gpu_exec_time,herald_exec,h2h_exec)/hax_conn_exec-1)*100,2))+"%\n\n")
