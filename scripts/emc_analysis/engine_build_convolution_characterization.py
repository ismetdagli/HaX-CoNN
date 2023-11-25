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


def build_engine_caffe(model_file, deploy_file, trans_layer, starts_gpu, batch):
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
            # print("layer.name: ", layer.name, " & i: ", i )
            latestLayer = i
        # print("Network size:",latestLayer)

        for i, layer in enumerate(network):
            config.set_device_type(layer, trt.DeviceType.GPU)
            if starts_gpu:
                config.set_device_type(layer, trt.DeviceType.GPU)
                if i < trans_layer:
                    config.set_device_type(layer, trt.DeviceType.DLA)  # DLA
                if i == latestLayer:
                    network.mark_output(
                        network.get_layer(network.num_layers - 1).get_output(0)
                    )
                    # network.mark_output(model_tensors.find(layer.name))
                # if i==output_layer_2:
                #     print(output_layer_2)
                #     network.mark_output(model_tensors.find(layer.name))
            else:
                config.set_device_type(layer, trt.DeviceType.DLA)  # DLA
                if i < trans_layer:
                    config.set_device_type(layer, trt.DeviceType.GPU)
                if i == latestLayer:
                    network.mark_output(
                        network.get_layer(network.num_layers - 1).get_output(0)
                    )
            # if i==latestLayer:
            #     network.mark_output(model_tensors.find(layer.name))

        # exit()
        # network.mark_output(model_tensors.find("prob"))
        return builder.build_engine(network, config)
        # return 0


def save_engine(serialized_engine, save_file):
    with open(save_file, "wb") as f:
        f.write(serialized_engine)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root_path = str(script_dir.parent.parent) + "/"

    print(root_path)
    input_dir_path = root_path + "convolution_characterization_prototxts/"
    input_engines = glob.glob(f"{input_dir_path}/*.prototxt")
    output_dir_path = root_path + "build/convolution_characterization_plans/"
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    for engine in input_engines:
        # print(engine)
        count = 0
        batch = 1
        test_network = Path(str(engine))
        network_stem = test_network.stem
        print(network_stem)
        print(engine)
        serialized_engine = build_engine_caffe(None, engine, 0, True, batch)
        if serialized_engine is not None:
            save_engine(
                serialized_engine.serialize(),
                str(output_dir_path + network_stem + ".plan"),
            )
