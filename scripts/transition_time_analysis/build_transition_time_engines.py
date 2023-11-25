"""
This is the engine building script to find transition cost in calculate_transition_time.py. 
We basically mark the layer as output layer that causes 
a transition between caches/buffers to memory.
The difference between the model having a transition and 
the baseline (no transition cost) model is calculated as
the transition cost of that layer. 
"""
import tensorrt as trt
import sys, os
from natsort import natsorted
import time
from pathlib import Path
import glob
import concurrent.futures

import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class CustomAlgorithmSelector(trt.IAlgorithmSelector):
    def __init__(self):
        trt.IAlgorithmSelector.__init__(self)

    def report_algorithms(self, contexts, choices):
        for choice in choices:
            print(choice.algorithm_variant.implementation)

    def select_algorithms(self, context, choices):
        assert len(choices) > 0
        return list(range(len(choices)))


def build_engine_caffe(
    model_file, deploy_file, trans_layer, runs_on_gpu, batch, verbose=False
):
    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
        config.max_workspace_size = 1 * 1 << 30
        config.flags = 1 << int(trt.BuilderFlag.FP16)
        config.DLA_core = 0
        config.flags = config.flags | 1 << int(trt.BuilderFlag.GPU_FALLBACK)

        builder.max_batch_size = batch

        model_tensors = parser.parse(
            deploy=deploy_file,
            model=None,
            network=network,
            dtype=trt.float16,
        )

        latestLayer = network.num_layers - 1

        if verbose:
            for i, layer in enumerate(network):
                print(layer.name, " & i: ", i)

            print("OUTPUT")
            for i, layer in enumerate(network):
                print(
                    layer.name,
                    " and i:",
                    i,
                    " shape:",
                    network.get_layer(i).get_output(0).shape,
                )

        for i, layer in enumerate(network):
            config.set_device_type(layer, trt.DeviceType.GPU)
            if runs_on_gpu:
                config.set_device_type(layer, trt.DeviceType.GPU)
                if i < trans_layer:
                    config.set_device_type(layer, trt.DeviceType.DLA)  # DLA
                if i == latestLayer:
                    network.mark_output(
                        network.get_layer(network.num_layers - 1).get_output(0)
                    )
                if i == transition:
                    network.mark_output(model_tensors.find(layer.name))
            else:
                config.set_device_type(layer, trt.DeviceType.DLA)  # DLA
                if i < trans_layer:
                    config.set_device_type(layer, trt.DeviceType.GPU)
                if i == latestLayer:
                    network.mark_output(
                        network.get_layer(network.num_layers - 1).get_output(0)
                    )

        return builder.build_engine(network, config)
        # return 0


def save_engine(serialized_engine, save_file):
    with open(save_file, "wb") as f:
        f.write(serialized_engine)


script_dir = Path(__file__).resolve().parent
root_path = script_dir.parent.parent
prototxt = root_path / "prototxt_input_files/googlenet.prototxt"
output_dir_path = root_path / "build/googlenet_transition_plans/"
Path(output_dir_path).mkdir(parents=True, exist_ok=True)


def build_and_save_engine(
    transition, model_file, deploy_file, runs_on_gpu, batch, output_name
):
    serialized_engine = build_engine_caffe(
        model_file, deploy_file, transition, runs_on_gpu, batch
    )
    if serialized_engine is not None:
        save_engine(serialized_engine.serialize(), output_name)


transition = -1


def main(max_workers=4):  # Default to 4, adjust as needed
    script_dir = Path(__file__).resolve().parent
    root_path = script_dir.parent.parent
    prototxt = root_path / "prototxt_input_files" / "googlenet.prototxt"

    transitions = {0, 10, 24, 38, 53, 67, 81, 95, 109, 124, 141}
    batch = 1
    runs_on_gpu = True

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # GPU transitions
            futures_gpu = [
                executor.submit(
                    build_and_save_engine,
                    transition,
                    None,
                    str(prototxt),
                    runs_on_gpu,
                    batch,
                    str(
                        output_dir_path
                        / f"googlenet_gpu_transition_at_{transition}.plan"
                    ),
                )
                for transition in transitions
            ]

            # DLA transitions
            futures_dla = [
                executor.submit(
                    build_and_save_engine,
                    transition,
                    None,
                    str(prototxt),
                    False,
                    batch,
                    str(
                        output_dir_path
                        / f"googlenet_dla_transition_at_{transition}.plan"
                    ),
                )
                for transition in transitions
            ]

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures_gpu + futures_dla):
                future.result()  # This will re-raise any exception raised during execution

    except Exception as e:
        print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
        for transition in transitions:
            # Sequentially build and save engines for GPU transitions
            build_and_save_engine(
                transition,
                None,
                str(prototxt),
                runs_on_gpu,
                batch,
                str(output_dir_path / f"googlenet_gpu_transition_at_{transition}.plan"),
                args.verbose,
            )
            # Sequentially build and save engines for DLA transitions
            build_and_save_engine(
                transition,
                None,
                str(prototxt),
                False,
                batch,
                str(output_dir_path / f"googlenet_dla_transition_at_{transition}.plan"),
                args.verbose,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engine Building Script")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-j", type=int, help="Number of workers")
    args = parser.parse_args()
    if not args.verbose:
        print("For more information run the script with --verbose option.")
    main(max_workers=args.j)
