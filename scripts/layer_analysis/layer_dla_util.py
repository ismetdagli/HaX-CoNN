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

    for profile_file in profiles_dir.glob("googlenet_dla_transition_at_*.profile"):
        profile_name = profile_file.stem
        total_dla_time = parse_profile_for_dla(profile_file)
        transition_number = int(profile_name.split('_')[-1])
        if transition_number != -1:  # Ignoring -1 transition
            dla_times[transition_number] = {"total_time": total_dla_time, "layer_diff": 0}

    # Sort the keys based on transition number and compute differences
    sorted_keys = sorted(dla_times.keys())
    dla_times[sorted_keys[0]]["layer_diff"] = dla_times[sorted_keys[0]]["total_time"]  # Initialize first transition difference
    for i in range(1, len(sorted_keys)):
        prev_key = sorted_keys[i - 1]
        curr_key = sorted_keys[i]
        time_diff = dla_times[curr_key]["total_time"] - dla_times[prev_key]["total_time"]
        dla_times[curr_key]["layer_diff"] = time_diff

    # Convert the sorted keys back to the original profile format
    sorted_dla_times = {f"googlenet_dla_transition_at_{key}": dla_times[key] for key in sorted_keys}

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
