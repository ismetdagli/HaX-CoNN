import json
from pathlib import Path
import sys

import src.trtexec_log_parser as parser


def parse_all_logs(logs_dir):
    results = {}
    for log_file in logs_dir.glob("*.log"):
        mean_time = parser.parse_gpu_compute_mean(log_file)
        if mean_time:
            results[log_file.stem] = mean_time
    return results


def main():
    script_dir = Path(__file__).resolve().parent
    root_path = script_dir.parent.parent
    profiles_dir = root_path / "build/googlenet_transition_plans/profiles/"
    logs_dir = root_path / "build/googlenet_transition_plans/profile_logs/"
    output_dir = root_path / "output/"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = parse_all_logs(logs_dir)
    sorted_results = dict(sorted(results.items()))

    output_file = output_dir / "gpu_compute_times.json"
    with output_file.open("w") as f:
        json.dump(sorted_results, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
