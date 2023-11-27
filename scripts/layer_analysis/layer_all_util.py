import json
import argparse
from pathlib import Path

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_group_summaries(gpu_data, layer_ranges):
    group_summaries = {}
    current_index = 0

    for group_range in layer_ranges:
        start, end = group_range
        group_gpu_time = 0
        group_layer_count = 0

        while current_index < len(gpu_data) and gpu_data[current_index]['layer_count'] <= end:
            if gpu_data[current_index]['layer_count'] >= start:
                group_gpu_time += gpu_data[current_index]['average_time_ms']
                group_layer_count += 1
            current_index += 1

        group_name = f"{start}-{end}"
        group_summaries[group_name] = {
            "gpu": {
                "total_gpu_time_ms": group_gpu_time,
                "layer_count": group_layer_count
            }
        }

    return group_summaries

def add_dla_data_to_groups(group_summaries, dla_data):
    for group_name in group_summaries:
        dla_time = dla_data.get(group_name, {"total_time": 0, "diff_from_prev": 0})
        group_summaries[group_name]['dla'] = dla_time

def main():
    parser = argparse.ArgumentParser(description="Aggregate layer timing information.")
    parser.add_argument('--gpu_json', help='Path to GPU JSON file', required=True)
    parser.add_argument('--dla_json', help='Path to DLA JSON file', required=True)
    parser.add_argument('--output', help='Output file path (optional)', required=False)
    args = parser.parse_args()

    gpu_data = load_json(args.gpu_json)
    dla_data = load_json(args.dla_json)

    # Define layer ranges for grouping
    layer_ranges = [(0, 9), (10, 24), (25, 38), (39, 53), (52, 66), (67, 80), (81, 94), (95, 109), (110, 123), (124, 140)]

    group_summaries = calculate_group_summaries(gpu_data, layer_ranges)
    add_dla_data_to_groups(group_summaries, dla_data)

    # Print or save the result
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w") as f:
            json.dump(group_summaries, f, indent=4)
        print(f"Results saved to {output_path}")
    else:
        for group_name, data in group_summaries.items():
            print(f"{group_name}: GPU Time = {data['gpu']['total_gpu_time_ms']} ms, Layer Count = {data['gpu']['layer_count']}, DLA Time = {data['dla']['total_time']}, DLA Diff = {data['dla']['diff_from_prev']}")

if __name__ == "__main__":
    main()

