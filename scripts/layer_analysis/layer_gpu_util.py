import json
from pathlib import Path
import argparse

from src.trtexec_profile_parser import *
from src.prototxt_parser import parse_prototxt


def parse_profile(profile_path, real_layers):
    with open(profile_path, "r") as file:
        profile_data = json.load(file)

    layers_info = []
    all_layer_names = set()  # To store individual layer names from profile

    for entry in profile_data:
        if "name" in entry and "averageMs" in entry:
            # Split fused or concatenated layer names
            layer_components = split_layer_names(entry["name"])
            # filtered_layers = filter_real_layers(layer_components, real_layers)
            if not isLayerReal(layer_components, real_layers):
                print(layer_components)
                continue
            layer_count = count_unique_layers(layer_components)

            layers_info.append(
                {
                    "name": entry["name"],
                    "average_time_ms": entry.get("averageMs", 0),
                    "layer_count": layer_count,
                }
            )

            # Add individual layer names to the set
            all_layer_names.update(layer_components)

    return layers_info, all_layer_names


def verify_layers(prototxt_layers, profile_layers):
    missing_layers = prototxt_layers - profile_layers
    if missing_layers:
        print("Missing layers from profile data:", missing_layers)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Parse TRTExec profile JSON for layer timing information."
    )
    parser.add_argument(
        "--profile", help="Path to the TRTExec profile JSON file", required=True
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_path = script_dir.parent.parent
    prototxt_path = root_path / "prototxt_input_files/googlenet.prototxt"
    # profile_dir = root_path/"build/googlenet_transition_plans/profiles/"
    output_dir = root_path / "build/googlenet_transition_plans/layer_times/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    real_layers = parse_prototxt(prototxt_path)
    layers_info, profile_layer_names = parse_profile(args.profile, real_layers)
    if verify_layers(real_layers, profile_layer_names):
        profile_name = Path(args.profile).stem
        output_file_name = f"{profile_name}_filtered.json"
        output_file = output_dir / output_file_name

        # Write the filtered layer information to the output file
        with open(output_file, "w") as file:
            json.dump(layers_info, file, indent=4)

        print(f"Layer timing information saved to {output_file}")
    else:
        print("Layer information from prototxt and profile do not match!")


if __name__ == "__main__":
    main()
