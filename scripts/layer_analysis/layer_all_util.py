import json
import argparse
from pathlib import Path

# print(f"Group {start}-{end}: Processing layer {current_index} name {gpu_data[current_index]['name']}, Time {layer_info['average_time_ms']}, Cumulative Time {group_gpu_time} \n in file {gpu_file}")
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_group_summaries(gpu_json_path, layer_ranges):
    gpu_data = load_json(gpu_json_path)
    group_summaries = {}

    current_range_index = 0
    current_group_layer_count = 0
    current_group_gpu_time = 0.0

    for layer in gpu_data:
        layer_time = layer['average_time_ms']
        layer_count = layer['layer_count']

        if current_group_layer_count + layer_count <= layer_ranges[current_range_index][1] - layer_ranges[current_range_index][0] + 1:
            # Add the layer to the current group
            current_group_gpu_time += layer_time
            current_group_layer_count += layer_count
            print(f"Adding layer {layer['name']}, Time {layer_time}, Cumulative Time {current_group_gpu_time}, Layers in group: {current_group_layer_count}")
        else:
            # Finalize the current group and start a new group
            group_key = f"{layer_ranges[current_range_index][0]}-{layer_ranges[current_range_index][1]}"
            group_summaries[group_key] = {
                "gpu": {
                    "total_gpu_time_ms": current_group_gpu_time,
                    "layer_count": current_group_layer_count
                }
            }

            # Update the range index and reset counters
            current_range_index += 1
            current_group_gpu_time = layer_time
            current_group_layer_count = layer_count
            print(f"Adding first layer of group {layer['name']}, Time {layer_time}, Cumulative Time {current_group_gpu_time}, Layers in group: {current_group_layer_count}")

    # Handle the last group
    if current_group_layer_count > 0:
        group_key = f"{layer_ranges[current_range_index][0]}-{layer_ranges[current_range_index][1]}"
        group_summaries[group_key] = {
            "gpu": {
                "total_gpu_time_ms": current_group_gpu_time,
                "layer_count": current_group_layer_count
            }
        }

    return group_summaries


def add_dla_data_to_groups(group_summaries, dla_data):
    for group_key in group_summaries.keys():
        # Default to zero if no DLA data is found for this group
        dla_info = dla_data.get(group_key, {"total_time": 0, "diff_from_prev": 0})
        group_summaries[group_key]['dla'] = dla_info

    return group_summaries


def align_dla_data(group_summaries, dla_data):
    for group_name in group_summaries:
        # Extract end layer index from the group name
        _, end_str = group_name.split('-')
        end_index = int(end_str)


        # Find the matching DLA data
        dla_key = f"googlenet_dla_transition_at_{end_index + 1}"
        print(f"Adding dla data to {group_name} with key {dla_key}")
        print(dla_data)
        if dla_key in dla_data:
            group_summaries[group_name]['dla'] = dla_data[dla_key]
        else:
            # If no exact match is found in DLA data, handle accordingly
            group_summaries[group_name]['dla'] = {"total_time": 0, "diff_from_prev": 0}

    return group_summaries





def main():
    parser = argparse.ArgumentParser(description="Aggregate layer timing information.")
    parser.add_argument('--gpu_json', help='Path to GPU JSON file', required=True)
    parser.add_argument('--dla_json', help='Path to DLA JSON file', required=True)
    parser.add_argument('--output', help='Output file path (optional)', required=False)
    args = parser.parse_args()

    layer_ranges = [(0, 9), (10, 24), (25, 38), (39, 52), (53, 66), (67, 80), (81, 94), (95, 109), (110, 123), (124, 140)]

    dla_data = load_json(args.dla_json)

    group_summaries = calculate_group_summaries(args.gpu_json, layer_ranges)
    align_dla_data(group_summaries, dla_data)

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w") as f:
            json.dump(group_summaries, f, indent=4)
        print(f"Results saved to {output_path}")
    else:
        print(json.dumps(group_summaries, indent=4))

if __name__ == "__main__":
    main()
