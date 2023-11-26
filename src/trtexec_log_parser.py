import re


def parse_gpu_compute_mean(log_file_path):
    gpu_compute_section = False
    mean_gpu_compute_regex = r"\[I\] mean: ([\d\.]+) ms"

    with open(log_file_path, "r") as file:
        for line in file:
            # Check if we are in the GPU Compute section
            if "GPU Compute" in line:
                gpu_compute_section = True
            elif gpu_compute_section:
                # Once in GPU Compute section, look for the mean time
                match = re.search(mean_gpu_compute_regex, line)
                if match:
                    return match.group(1)

    return None
