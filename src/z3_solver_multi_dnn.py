from z3 import *

# Execution times for each layer group on each accelerators
nn1_times_acc1 = [0.25,0.06, 0.22, 0.08, 0.16, 0.7, 1.93,0.10]
nn1_times_acc2 = [0.3, 0.10, 0.34, 0.2,  0.4,  3.3, 2.6, 0.85]
nn2_times_acc1 = [0.14, 0.11, 0.39, 0.58, 1.1, 0.9, 0.75, 0.52, 0.59, 0.07] #5.15
nn2_times_acc2 = [0.20, 0.18, 0.74, 1.33, 2.14, 1.58, 1.65, 1.16, 1.17, 0.35]


# Transition cost after each layer, the current data is dummy.
nn1_trans_acc1 = [0.01]*8
nn1_trans_acc2 = [0.01]*8
nn2_trans_acc1 = [0.01]*10
nn2_trans_acc2  =[0.01]*10

# Memory throughput for each layer group on each accelerator, the current data is dummy.
nn1_slowdown_acc1 = [40, 35, 41, 53, 71, 52, 49, 76, 66]
nn1_slowdown_acc2 = [25, 28, 33, 38, 59, 45, 41, 64, 58]
nn2_slowdown_acc1 = [50]*11
nn2_slowdown_acc2 = [20]*11

# Number of layers for each neural network
nn1_layers = len(nn1_times_acc1)
nn2_layers = len(nn2_times_acc1)

# Creating Z3 variables for start times and decision variables
start_nn1 = [Real(f'start_nn1_{i}') for i in range(nn1_layers)]
start_nn2 = [Real(f'start_nn2_{i}') for i in range(nn2_layers)]

# True represents GPU and False represent DLA.
acc_decision_nn1 = [Bool(f'acc_decision_nn1_{i}') for i in range(nn1_layers)]
acc_decision_nn2 = [Bool(f'acc_decision_nn2_{i}') for i in range(nn2_layers)]

# Creating the optimizer
opt = Optimize()


def trans_cost_nn1(layer_index):
    if layer_index > 0:
        trans_cost_layer=If(acc_decision_nn1[i] != acc_decision_nn1[i-1], 
                                        nn1_trans_acc1[i-1], 
                                        0)
        return trans_cost_layer
    return 0

def trans_cost_nn2(layer_index):
    if layer_index > 0:
        trans_cost_layer=If(acc_decision_nn2[i] != acc_decision_nn2[i-1], 
                                        nn2_trans_acc1[i-1], 
                                        0)
        return trans_cost_layer
    return 0

#Slowdown on GPU
def slowdown_acc1(index_of_layer,nn_slowdown_acc1):
    if nn_slowdown_acc1[index_of_layer] < 38.1:
        return 100/(100-(nn_slowdown_acc1[index_of_layer] *4.9/127.5))
    
    elif nn_slowdown_acc1[index_of_layer] < 96.2:
        return 100/(100-(nn_slowdown_acc1[index_of_layer]))
    
    else:
        return 100/(100-(nn_slowdown_acc1[index_of_layer]-41.9)*1.11)

#Slowdown on GPU
def slowdown_acc2(index_of_layer,nn_slowdown_acc2):
    if nn_slowdown_acc2[index_of_layer] < 27.9:
        return 100/(100-(nn_slowdown_acc2[index_of_layer]))
    
    else:
        return 100/(100-(nn_slowdown_acc2[index_of_layer]+49)*0.35)


# Adding constraints for sequential execution and accelerator decision
for i in range(nn1_layers):
    opt.add(If(acc_decision_nn1[i], 
               start_nn1[i] >= If(i > 0, start_nn1[i-1] +trans_cost_nn1(i) + nn1_times_acc1[i-1]*slowdown_acc1(i-1,nn1_slowdown_acc1), 0), 
               start_nn1[i] >= If(i > 0, start_nn1[i-1] +trans_cost_nn1(i) + nn1_times_acc2[i-1]*slowdown_acc2(i-1,nn1_slowdown_acc2), 0)))

for i in range(nn2_layers):
    opt.add(If(acc_decision_nn2[i], 
               start_nn2[i] >= If(i > 0, start_nn2[i-1] +trans_cost_nn2(i) + nn2_times_acc1[i-1]*slowdown_acc1(i-1,nn2_slowdown_acc1), 0), 
               start_nn2[i] >= If(i > 0, start_nn2[i-1] +trans_cost_nn2(i) + nn2_times_acc2[i-1]*slowdown_acc2(i-1,nn2_slowdown_acc2), 0)))

# Adding constraints to avoid parallel execution on the same accelerator
for i in range(nn1_layers):
    for j in range(nn2_layers):
        # For accelerator 1
        opt.add(Or(Not(acc_decision_nn1[i]), Not(acc_decision_nn2[j]), 
                   start_nn1[i] + nn1_times_acc1[i] <= start_nn2[j],
                   start_nn2[j] + nn2_times_acc1[j] <= start_nn1[i]))
        # For accelerator 2
        opt.add(Or(acc_decision_nn1[i], acc_decision_nn2[j], 
                   start_nn1[i] + nn1_times_acc2[i] <= start_nn2[j],
                   start_nn2[j] + nn2_times_acc2[j] <= start_nn1[i]))

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

# Solving the optimization problem
if opt.check() == sat:
    m = opt.model()
    for i in range(nn1_layers):
        acc = 1 if is_true(m[acc_decision_nn1[i]]) else 2
        print(f'NN2 Layer {i+1}, Accelerator {acc}, Start Time: {convert(str(m[start_nn1[i]]))}')
    for i in range(nn2_layers):
        acc = 1 if is_true(m[acc_decision_nn2[i]]) else 2
        print(f'NN2 Layer {i+1}, Accelerator {acc}, Start Time: {convert(str(m[start_nn2[i]]))}')
        # print(f'NN2 Layer {i+1}, Accelerator {acc}, Start Time: {m[start_nn2[i]]}')
    
    result=sorted ([(d, m[d]) for d in m], key = lambda x: str(x[0]))   
    print ("\n \nResult:",result) #"Check: "
else:
    print("No feasible solution found")


#This is written to mark the layer groups. The index of each layer group is represented the index of the layer in the prototxt
layer_group_nn1=[10,24,39,52,66,81,94,109,124,140]
layer_group_nn2=[0,10,24,39,52,66,81,94,109,124,140]

# Solving the optimization problem
if opt.check() == sat:
    m = opt.model()

    previous_acc=""
    for i in range(nn1_layers):
        acc = "GPU" if is_true(m[acc_decision_nn1[i]]) else "DLA"
        if i == 0:
            print(f'NN1 starts on {acc}')
        else:
            if previous_acc != acc:
                print(f'NN1 applies transition at {layer_group_nn1[i]}')
        previous_acc=acc
    
    previous_acc=""
    for i in range(nn2_layers):
        acc = "GPU" if is_true(m[acc_decision_nn2[i]]) else "DLA"
        if i == 0:
            print(f'NN2 starts on {acc}')
        else:
            if previous_acc != acc:
                print(f'NN2 applies transition at {layer_group_nn2[i]}')
        previous_acc=acc
else:
    print("No feasible solution found")

