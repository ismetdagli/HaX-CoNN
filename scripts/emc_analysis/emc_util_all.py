"""
This script manages all of the inputs and collects the output of 
EMC utilization data. 
"""
import json
from pathlib import Path
from collections import defaultdict
import re


def get_max_emc_value(file_path):
    with open(file_path, "r") as file:
        values = [int(line.strip("%\n")) for line in file.readlines()]
    return f"{max(values)}%" if values else "0%"


def parse_emc_filename(filename):
    match = re.match(r"conv(\d+)_kernel(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


script_dir = Path(__file__).resolve().parent
root_path = script_dir.parent.parent
emc_data_dir = root_path / "build/convolution_characterization_plans/times/"
emc_files = emc_data_dir.glob("*.txt")
output_dir = root_path / "output/"
output_dir.mkdir(parents=True, exist_ok=True)

emc_results = defaultdict(dict)

for file_path in emc_files:
    conv_num, kernel_num = parse_emc_filename(file_path.stem)
    if conv_num is not None and kernel_num is not None:
        emc_results[f"conv{conv_num}"][f"kernel{kernel_num}"] = get_max_emc_value(
            file_path
        )

# Sort the results
sorted_emc_results = {
    conv: dict(sorted(kernels.items())) for conv, kernels in sorted(emc_results.items())
}

json_output_path = output_dir / "emc_results.json"
with json_output_path.open("w") as json_file:
    json.dump(sorted_emc_results, json_file, indent=4)

print(f"EMC utilization data collected and saved to {json_output_path}.")
