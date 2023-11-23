"""
This script manages all of the inputs and collects the output of 
EMC utilization data. 
"""

import subprocess
import yaml

script_path = "./emc_single_run.sh"
input_arg = "convolution_characterization_plans/conv1_kernel1.plan"

def emc_single_run(script_path, input_arg):
    try:
        result = subprocess.run(
            ["sudo", script_path, input_arg], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True, 
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error occurred running {script_path}: {e}")
        return None

script_path = "./emc_single_run.sh"

emc_results = {}

# Iterate over the conv and kernel values
for conv in range(1, 6):
    emc_results[f"conv{conv}"] = {}
    for kernel in range(1, 6):
        input_arg = f"convolution_characterization_plans/conv{conv}_kernel{kernel}.plan"
        output = emc_single_run(script_path, input_arg)
        if output is not None:
            emc_results[f"conv{conv}"][f"kernel{kernel}"] = output
            print(f"Saved result {output} in conv {conv} and kernel {kernel}")

# Write the results to a YAML file
with open('emc_results.yaml', 'w') as file:
    yaml.dump(emc_results, file, default_flow_style=False)

print("EMC utilization data collected and saved to emc_results.yaml.")

