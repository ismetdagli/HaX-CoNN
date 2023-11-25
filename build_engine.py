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


def build_engine_caffe(model_file, deploy_file, trans_layer, runs_on_gpu, batch):
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
        for i, layer in enumerate(network):
            print(layer.name, " & i: ", i)
            latestLayer = i

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


if __name__ == "__main__":
    count = 0
    batch = 1
    transition = -1
    prototxt = "prototxt_input_files/googlenet.prototxt"
    serialized_engine = build_engine_caffe(None, prototxt, transition, True, batch)
    save_engine(
        serialized_engine.serialize(),
        str("google_only_gpu.plan"),
    )
    serialized_engine = build_engine_caffe(None, prototxt, transition, False, batch)
    save_engine(
        serialized_engine.serialize(),
        str("google_only_dla.plan"),
    )
