"""
This script manages all of the inputs and collects the output of 
EMC utilization data. 
"""

import subprocess
import yaml
import os

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

script_dir = os.path.dirname(os.path.realpath(__file__))
single_run_path = os.path.join(script_dir, "emc_single_run.sh")
root_path = os.path.join(script_dir, "../../")


emc_results = {}

# Iterate over the conv and kernel values
for conv in range(1, 6):
    emc_results[f"conv{conv}"] = {}
    for kernel in range(1, 6):
        input_arg = f"build/convolution_characterization_plans/conv{conv}_kernel{kernel}.plan"
        output = emc_single_run(single_run_path, root_path + input_arg)
        if output is not None:
            emc_results[f"conv{conv}"][f"kernel{kernel}"] = output
            print(f"Saved result {output} in conv {conv} and kernel {kernel}")

# Write the results to a YAML file
with open(root_path + 'output/emc_results.yaml', 'w') as file:
    yaml.dump(emc_results, file, default_flow_style=False)

print("EMC utilization data collected and saved to emc_results.yaml.")

