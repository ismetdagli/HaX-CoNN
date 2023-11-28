from z3 import *


#ResNet101: Execution times for each layer on each accelerator
nn1_times_acc1 = [0.14, 0.11, 0.39, 0.58, 1.4, 0.9, 0.75, 0.52, 0.59, 0.07] #5.45 GPU
nn1_times_acc2 = [0,28, 0.15, 1.99, 1.73, 2.8, 1.9, 1.41, 0.99, 1.14, 0.15] #10.6 DLA

#Alexnet: Execution times for each layer on each accelerator
nn2_times_acc1 = [0.15, 0.06, 0.22, 0.08, 0.16, 0.70, 0.32, 0.10] #1.80
nn2_times_acc2 = [0.41, 0.20, 0.45, 0.22, 0.41, 1.91, 0.68, 0.13] #4.41

#Alexnet Slowdown ratios for each layer on each accelerator (example values, adjust as needed)
slowdown_ratio_nn2_on_acc1 = [0.9, 0.3, 0.2, 0.51, 0.41, 0.5, 0.33, 0,4]   # Slowdown of nn1 layers due to nn2 layers on acc2
slowdown_ratio_nn2_on_acc2 = [0.7, 0.2, 0.5, 0.55, 0.42, 0.7, 0.71, 0.5]   # Slowdown of nn1 layers due to nn2 layers on acc1


#Alexnet Slowdown ratios for each layer on each accelerator (example values, adjust as needed)
slowdown_ratio_nn1_on_acc1 = [0.9, 0.3, 0.2, 0.51, 0.41, 0.5, 0.33, 0,4, 0.1, 0.4] # Slowdown of nn1 layers due to nn2 layers on acc2
slowdown_ratio_nn1_on_acc2 = [0.7, 0.2, 0.5, 0.55, 0.42, 0.7, 0.71, 0.5, 0.3, 0.4] # Slowdown of nn1 layers due to nn2 layers on acc1

#Alexnet: Execution times for each layer on each accelerator
nn2_times_acc1 = [0.15, 0.06, 0.22, 0.08, 0.16, 0.70, 0.32, 0.10] #1.80
nn2_times_acc2 = [0.41, 0.20, 0.45, 0.22, 0.41, 1.91, 0.68, 0.13]

# # Execution times for each layer on each accelerator
# nn1_times_acc1 = [0.25,0.06, 0.22, 0.08, 0.16, 0.7, 1.93,0.10]
# nn1_times_acc2 = [0.3, 0.10, 0.34, 0.2,  0.4,  3.3, 2.6, 0.85]
# nn2_times_acc1 = [0.14, 0.11, 0.39, 0.58, 1.1, 0.9, 0.75, 0.52, 0.59, 0.07] #5.15
# nn2_times_acc2 = [0.20, 0.18, 0.74, 1.33, 2.14, 1.58, 1.65, 1.16, 1.17, 0.35]

# Number of layers for each neural network
nn1_layers = len(nn1_times_acc1)
nn2_layers = len(nn2_times_acc1)


# Creating Z3 variables for start times and decision variables
start_nn1 = [Real(f'start_nn1_{i}') for i in range(nn1_layers)]
start_nn2 = [Real(f'start_nn2_{i}') for i in range(nn2_layers)]
acc_decision_nn1 = [Bool(f'acc_decision_nn1_{i}') for i in range(nn1_layers)]
acc_decision_nn2 = [Bool(f'acc_decision_nn2_{i}') for i in range(nn2_layers)]

# Creating the optimizer
opt = Optimize()

# Function to calculate overlap between two intervals
def calculate_overlap(start1, duration1, start2, duration2):
    end1 = start1 + duration1
    end2 = start2 + duration2
    return If(Or(start1 >= end2, start2 >= end1), 
              0, 
              If(end1 < end2, end1, end2) - If(start1 > start2, start1, start2))

# Function to calculate adjusted execution time
def adjusted_exec_time(layer, start_times, acc_decision, nn_times_acc1, nn_times_acc2, other_start_times, other_nn_times_acc1, other_nn_times_acc2, slowdown_ratio):
    base_time = If(acc_decision[layer], nn_times_acc1[layer], nn_times_acc2[layer])
    slowdown_effect = Sum([calculate_overlap(start_times[layer], base_time, 
                                             other_start_times[j], 
                                             If(acc_decision[layer], other_nn_times_acc2[j], other_nn_times_acc1[j])) 
                           * slowdown_ratio[j] 
                           for j in range(len(other_nn_times_acc1))])
    return base_time + slowdown_effect

# Adding constraints for sequential execution and accelerator decision with adjusted times
for i in range(nn1_layers):
    exec_time = adjusted_exec_time(i, start_nn1, acc_decision_nn1, nn1_times_acc1, nn1_times_acc2, start_nn2, nn2_times_acc1, nn2_times_acc2, slowdown_ratio_nn2_on_acc1)
    opt.add(start_nn1[i] >= If(i > 0, start_nn1[i-1] + exec_time, 0))

for i in range(nn2_layers):
    exec_time = adjusted_exec_time(i, start_nn2, acc_decision_nn2, nn2_times_acc1, nn2_times_acc2, start_nn1, nn1_times_acc1, nn1_times_acc2, slowdown_ratio_nn1_on_acc2)
    opt.add(start_nn2[i] >= If(i > 0, start_nn2[i-1] + exec_time, 0))


# # Adding constraints for sequential execution and accelerator decision
# for i in range(nn1_layers):
#     opt.add(If(acc_decision_nn1[i], 
#                start_nn1[i] >= If(i > 0, start_nn1[i-1] + nn1_times_acc1[i-1], 0), 
#                start_nn1[i] >= If(i > 0, start_nn1[i-1] + nn1_times_acc2[i-1], 0)))

# for i in range(nn2_layers):
#     opt.add(If(acc_decision_nn2[i], 
#                start_nn2[i] >= If(i > 0, start_nn2[i-1] + nn2_times_acc1[i-1], 0), 
#                start_nn2[i] >= If(i > 0, start_nn2[i-1] + nn2_times_acc2[i-1], 0)))

# Adding constraints to avoid parallel execution on the same accelerator
for i in range(nn1_layers):
    for j in range(nn2_layers):
        # For accelerator 1
        opt.add(Or(Not(acc_decision_nn1[i]), Not(acc_decision_nn2[j]), 
                   start_nn1[i] + nn1_times_acc1[i] <= start_nn2[j],
                   start_nn2[j] + nn2_times_acc1[j] <= start_nn1[i]))

        # For accelerator 2
        opt.add(Or(Not(acc_decision_nn1[i]), acc_decision_nn2[j], 
                   start_nn1[i] + nn1_times_acc2[i] <= start_nn2[j],
                   start_nn2[j] + nn2_times_acc2[j] <= start_nn1[i]))

        opt.add(Or(acc_decision_nn1[i], Not(acc_decision_nn2[j]), 
                   start_nn1[i] + nn1_times_acc2[i] <= start_nn2[j],
                   start_nn2[j] + nn2_times_acc2[j] <= start_nn1[i]))

        opt.add(Or(acc_decision_nn1[i], acc_decision_nn2[j], 
                   start_nn1[i] + nn1_times_acc1[i] <= start_nn2[j],
                   start_nn2[j] + nn2_times_acc1[j] <= start_nn1[i]))


# Adding constraints for at most one switch between accelerators for each neural network
# For neural network 1
for i in range(1, nn1_layers):
    opt.add(Implies(acc_decision_nn1[i-1] != acc_decision_nn1[i], 
                    And([acc_decision_nn1[j] == acc_decision_nn1[i] for j in range(i+1, nn1_layers)])))

# For neural network 2
for i in range(1, nn2_layers):
    opt.add(Implies(acc_decision_nn2[i-1] != acc_decision_nn2[i], 
                    And([acc_decision_nn2[j] == acc_decision_nn2[i] for j in range(i+1, nn2_layers)])))                   

# Defining the objective: Minimize the completion time of the last layer
last_nn1_layer_time = If(acc_decision_nn1[-1], start_nn1[-1] + nn1_times_acc1[-1], start_nn1[-1] + nn1_times_acc2[-1])
last_nn2_layer_time = If(acc_decision_nn2[-1], start_nn2[-1] + nn2_times_acc1[-1], start_nn2[-1] + nn2_times_acc2[-1])
opt.minimize(If(last_nn1_layer_time > last_nn2_layer_time, last_nn1_layer_time, last_nn2_layer_time))

def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)
while(True):
    # Solving the optimization problem
    if opt.check() == sat:
        m = opt.model()
        for i in range(nn1_layers):
            acc = 1 if is_true(m[acc_decision_nn1[i]]) else 2
            # print(f'NN2 Layer {i+1}, Accelerator {acc}, Start Time: {convert(str(m[start_nn1[i]]))}')
        for i in range(nn2_layers):
            acc = 1 if is_true(m[acc_decision_nn2[i]]) else 2
            # print(f'NN2 Layer {i+1}, Accelerator {acc}, Start Time: {convert(str(m[start_nn2[i]]))}')
            # print(f'NN2 Layer {i+1}, Accelerator {acc}, Start Time: {m[start_nn2[i]]}')
        
        result=sorted ([(d, m[d]) for d in m], key = lambda x: str(x[0]))   
        print ("\n \nResult:",result) #"Check: "
    else:
        print("No solution found")
