#Serial execution summary

file1 = open('baseline_engine_logs/mean_results_of_executions.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character

def index_mean(line):
    index_of_mean= line.index('mean')
    return float(line[index_of_mean+6:index_of_mean+11])

print("Summary of Exp1. GoogleNet and ResNet")
googlenet_average_time=0
resnet_average_time=0
for line in Lines:
    if "googlenet_only_gpu_resnet101_only_gpu" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_gpu_googlenet_only_gpu" in line:
        resnet_average_time = index_mean(line)
only_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of using only GPU:", round(only_gpu_exec_time,1) )


for line in Lines:
    if "googlenet_only_dla_resnet101_only_gpu" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_gpu_googlenet_only_dla" in line:
        resnet_average_time = index_mean(line)

gpu_dla_exec_time=googlenet_average_time+resnet_average_time
print("Average time of Resnet101 on GPU and Googlenet on DLA:",round(gpu_dla_exec_time,1) )


for line in Lines:
    if "googlenet_only_gpu_resnet101_only_dla" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_only_dla_googlenet_only_gpu" in line:
        resnet_average_time = index_mean(line)

dla_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of Resnet101 on DLA and Googlenet on GPU:", round(dla_gpu_exec_time,1) )
    


for line in Lines:
    if "googlenet_dla_transition_at_10_resnet101_gpu_transition_at_4" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_transition_at_4_googlenet_dla_transition_at_10" in line:
        resnet_average_time = index_mean(line)
herald_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by Herald:",round(herald_exec,1))


for line in Lines:
    if "googlenet_dla_transition_at_39_resnet101_gpu_transition_at_101" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_transition_at_101_googlenet_dla_transition_at_39" in line:
        resnet_average_time = index_mean(line)
h2h_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by H2H:",round(h2h_exec,1))


for line in Lines:
    if "googlenet_gpu_transition_at_80_resnet101_dla_transition_at_24" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_dla_transition_at_24_googlenet_gpu_transition_at_80" in line:
        resnet_average_time = index_mean(line)
hax_conn_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by HaX-CoNN:",round(hax_conn_exec,1))

print("Overall improvement over best-baseline: "+str(round((min(only_gpu_exec_time,gpu_dla_exec_time,dla_gpu_exec_time,herald_exec,h2h_exec)/hax_conn_exec-1)*100,2))+"%\n\n")



#--------------------------EXP2--------------------------


print("Summary of Exp2. Inception and Resnet152")
googlenet_average_time=0
resnet_average_time=0
for line in Lines:
    if "inception_gpu_only_resnet152_gpu_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_gpu_only_inception_gpu_only" in line:
        resnet_average_time = index_mean(line)
only_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of using only GPU:", round(only_gpu_exec_time,1) )


for line in Lines:
    if "inception_dla_only_resnet152_gpu_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_gpu_only_inception_dla_only" in line:
        resnet_average_time = index_mean(line)

gpu_dla_exec_time=googlenet_average_time+resnet_average_time
print("Average time of ResNet152 on GPU and Inception on DLA:",round(gpu_dla_exec_time,1) )


for line in Lines:
    if "inception_gpu_only_resnet152_dla_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_dla_only_inception_gpu_only" in line:
        resnet_average_time = index_mean(line)

dla_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of ResNet152 on DLA and Inception on GPU:", round(dla_gpu_exec_time,1) )
    


for line in Lines:
    if "inception_dla_transition_at_30_resnet152_gpu_transition_at_46" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_transition_at_4_inception_dla_transition_at_30" in line:
        resnet_average_time = index_mean(line)
herald_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by Herald:",round(herald_exec,1))


for line in Lines:
    if "inception_dla_transition_at_95_resnet152_gpu_transition_at_101" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_gpu_transition_at_101_inception_dla_transition_at_95" in line:
        resnet_average_time = index_mean(line)
h2h_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by H2H:",round(h2h_exec,1))


for line in Lines:
    if "inception_gpu_transition_at_510_resnet152_dla_transition_at_636" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_dla_transition_at_636_inception_gpu_transition_at_510" in line:
        resnet_average_time = index_mean(line)
hax_conn_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by HaX-CoNN:",round(hax_conn_exec,1))

print("Overall improvement over best-baseline: "+str(round((min(only_gpu_exec_time,gpu_dla_exec_time,dla_gpu_exec_time,herald_exec,h2h_exec)/hax_conn_exec-1)*100,2))+"%\n\n")






#--------------------------EXP3--------------------------

print("Summary of Exp3. Alexnet Resnet101")
googlenet_average_time=0
resnet_average_time=0
for line in Lines:
    if "alexnet_gpu_only_resnet101_gpu_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_only_alexnet_gpu_only" in line:
        resnet_average_time = index_mean(line)
only_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of using only GPU:", round(only_gpu_exec_time,1) )


for line in Lines:
    if "alexnet_gpu_only_resnet101_dla_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_dla_only_alexnet_gpu_only" in line:
        resnet_average_time = index_mean(line)

gpu_dla_exec_time=googlenet_average_time+resnet_average_time
print("Average time of Alexnet on GPU and Resnet101 on DLA:",round(gpu_dla_exec_time,1) )


for line in Lines:
    if "alexnet_dla_only_resnet101_gpu_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_gpu_only_alexnet_dla_only" in line:
        resnet_average_time = index_mean(line)

dla_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of Alexnet on DLA and Resnet101 on GPU:", round(dla_gpu_exec_time,1) )
    


for line in Lines:
    if "alexnet_gpu_transition_at_16_resnet101_dla_transition_at_58" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_dla_transition_at_58_alexnet_gpu_transition_at_16" in line:
        resnet_average_time = index_mean(line)
herald_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by Herald:",round(herald_exec,1))


for line in Lines:
    if "alexnet_gpu_transition_at_14_resnet101_dla_transition_at_101" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_dla_transition_at_101_alexnet_gpu_transition_at_14" in line:
        resnet_average_time = index_mean(line)
h2h_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by H2H:",round(h2h_exec,1))


for line in Lines:
    if "alexnet_gpu_transition_at_16_resnet101_dla_transition_at_4" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet101_dla_transition_at_4_alexnet_gpu_transition_at_16" in line:
        resnet_average_time = index_mean(line)
hax_conn_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by HaX-CoNN:",round(hax_conn_exec,1))

print("Overall improvement over best-baseline: "+str(round((min(only_gpu_exec_time,gpu_dla_exec_time,dla_gpu_exec_time,herald_exec,h2h_exec)/hax_conn_exec-1)*100,2))+"%\n\n")










#--------------------------EXP3--------------------------

print("Summary of Exp4. VGG19 Resnet152")
googlenet_average_time=0
resnet_average_time=0
for line in Lines:
    if "vgg19_gpu_only_resnet152_gpu_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_gpu_only_vgg19_gpu_only" in line:
        resnet_average_time = index_mean(line)
only_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of using only GPU:", round(only_gpu_exec_time,1) )


for line in Lines:
    if "vgg19_gpu_only_resnet152_dla_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_dla_only_vgg19_gpu_only" in line:
        resnet_average_time = index_mean(line)

gpu_dla_exec_time=googlenet_average_time+resnet_average_time
print("Average time of VGG on GPU and Resnet152 on DLA:",round(gpu_dla_exec_time,1) )


for line in Lines:
    if "vgg19_dla_only_resnet152_gpu_only" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_gpu_only_vgg19_dla_only" in line:
        resnet_average_time = index_mean(line)

dla_gpu_exec_time=googlenet_average_time+resnet_average_time
print("Average time of VGG on DLA and Resnet152 on GPU:", round(dla_gpu_exec_time,1) )
    


for line in Lines:
    if "vgg19_gpu_transition_at_9_resnet152_dla_transition_at_46" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_dla_transition_at_46_vgg19_gpu_transition_at_9" in line:
        resnet_average_time = index_mean(line)
herald_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by Herald:",round(herald_exec,1))


for line in Lines:
    if "vgg19_gpu_transition_at_27_resnet152_dla_transition_at_286" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_dla_transition_at_286_vgg19_gpu_transition_at_27" in line:
        resnet_average_time = index_mean(line)
h2h_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by H2H:",round(h2h_exec,1))


for line in Lines:
    if "vgg19_dla_transition_at_9_resnet152_gpu_transition_at_165" in line:
        googlenet_average_time = index_mean(line)
    
    if "resnet152_gpu_transition_at_165_vgg19_dla_transition_at_9" in line:
        resnet_average_time = index_mean(line)
hax_conn_exec=max(googlenet_average_time,resnet_average_time)

print("Average time of the schedule found by HaX-CoNN:",round(hax_conn_exec,1))

print("Overall improvement over best-baseline: "+str(round((min(only_gpu_exec_time,gpu_dla_exec_time,dla_gpu_exec_time,herald_exec,h2h_exec)/hax_conn_exec-1)*100,2))+"%")





