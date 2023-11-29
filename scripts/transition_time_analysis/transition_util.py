import json
from pathlib import Path
import sys
import re

import src.trtexec_log_parser as parser


def sort_key_func(key):
    parts = key.split("_")
    type_part = parts[1]  # 'gpu' or 'dla'
    transition_part = int(parts[-1])  # numerical part
    return (type_part, transition_part)

def parse_all_logs(logs_dir):
    results = {}
    for log_file in logs_dir.glob("*.log"):
        mean_time = parser.parse_gpu_compute_mean(log_file)
        if mean_time:
            results[log_file.stem] = mean_time
    return results


def calculate_differences(results, base_plan_dla, base_plan_gpu):
    diffs = {}
    for key, value_str in results.items():
        value = float(value_str)
        if "dla" in key:
            base_value = float(base_plan_dla)
        elif "gpu" in key:
            base_value = float(base_plan_gpu)
        else:
            continue  # Skip if the key does not contain 'dla' or 'gpu'

        diffs[key] = {
            "mean_time": value,
            "transition_cost": round(value - base_value, 5)
        }
    return diffs

def main():
    script_dir = Path(__file__).resolve().parent
    root_path = script_dir.parent.parent
    logs_dir = root_path / "build/googlenet_mark_plans/profile_logs/"
    output_dir = root_path / "output/"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = parse_all_logs(logs_dir)
    sorted_results = dict(sorted(results.items(), key=lambda item: sort_key_func(item[0])))

    # Assuming base plans are the ones with transition_at_-1
    base_plan_dla = results.get("googlenet_dla_mark_at_-1", "0")
    base_plan_gpu = results.get("googlenet_gpu_mark_at_-1", "0")

    diffs = calculate_differences(sorted_results, base_plan_dla, base_plan_gpu)

    output_file = output_dir / "transition_results.json"
    with output_file.open("w") as f:
        json.dump(diffs, f, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
