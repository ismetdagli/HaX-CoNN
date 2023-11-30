"""
In contrast with gpu parsing we only need to sum the execution
times of {} layers here.
"""
import re
import json
from pathlib import Path
import argparse

def parse_profile_for_dla(profile_path):
    with open(profile_path, "r") as file:
        profile_data = json.load(file)

    total_dla_time = 0
    dla_regex = r'\{.*?\}'  # Regex pattern to find substrings enclosed in braces

    for entry in profile_data:
        if 'name' in entry and 'averageMs' in entry:
            layer_name = entry['name']
            if re.search(dla_regex, layer_name):  # Check if the layer name matches the regex pattern
                time = entry['averageMs']
                total_dla_time += time
                print(f"Time {time} for {layer_name[:40]}...{layer_name[-40:]}\n in profile {profile_path}")

    return total_dla_time

def process_all_dla_profiles(profiles_dir):
    profiles_dir = Path(profiles_dir)
    dla_times = {}

    baseline_dla = parse_profile_for_dla(profiles_dir / "googlenet_dla_transition_at_-1.profile")

    # Process GPU transition profiles
    for profile_file in profiles_dir.glob("googlenet_gpu_transition_at_*.profile"):
        profile_name = profile_file.stem
        total_dla_time = parse_profile_for_dla(profile_file)
        transition_number = int(profile_name.split('_')[-1])
        dla_times[transition_number] = {"total_time": total_dla_time, "layer_diff": 0}

    # Sort the keys based on transition number and compute differences
    sorted_keys = sorted(dla_times.keys(), reverse=True)

    # Calculate differences with the next lower transition
    for i in range(len(sorted_keys) - 1):
        curr_key = sorted_keys[i]
        next_lower_key = sorted_keys[i + 1]
        time_diff = dla_times[next_lower_key]["total_time"] - dla_times[curr_key]["total_time"]
        dla_times[next_lower_key]["layer_diff"] = time_diff

    # For the lowest transition (-1), set the difference to baseline_dla
    dla_times[-1] = {"total_time": baseline_dla, "layer_diff": baseline_dla}

    # Convert the sorted keys back to the original profile format and adjust the keys
    sorted_dla_times = {}
    for key in sorted_keys:
        if key == -1:
            new_key = "googlenet_dla_transition_at_-1"
        else:
            new_key = f"googlenet_gpu_transition_at_{key}"
        sorted_dla_times[new_key] = dla_times[key]

    return sorted_dla_times






def main():
    parser = argparse.ArgumentParser(description="Process all DLA profile JSON files.")
    parser.add_argument('--profiles_dir', help='Directory containing DLA profile JSON files', required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_path = script_dir.parent.parent
    output_dir = root_path / "output/"
    output_dir.mkdir(parents=True, exist_ok=True)

    dla_times = process_all_dla_profiles(args.profiles_dir)

    output_file_name = "dla_compute_times.json"
    output_file = output_dir / output_file_name

    # Save DLA timing information
    with open(output_file, "w") as file:
        json.dump(dla_times, file, indent=4)

    print(f"DLA timing information saved to {output_file}")

if __name__ == "__main__":
    main()
