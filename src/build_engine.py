import tensorrt as trt
import argparse
from pathlib import Path


def build_engine_caffe(deploy_file, transition, starts_gpu, batch, verbose=False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:

        builder.max_batch_size = batch
        configure_builder_config(config)
        parse_model_with_caffe_parser(parser, deploy_file, network)
        configure_layers(network, config, transition, starts_gpu, verbose)
        return build_and_return_engine(builder, network, config)


def configure_builder_config(config):
    config.max_workspace_size = 1 << 30
    config.flags |= 1 << int(trt.BuilderFlag.FP16)
    config.DLA_core = 0
    config.flags |= 1 << int(trt.BuilderFlag.GPU_FALLBACK)


def parse_model_with_caffe_parser(parser, deploy_file, network):
    parser.parse(deploy=deploy_file, model=None, network=network, dtype=trt.float16)


def configure_layers(network, config, transition, starts_gpu, verbose):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if verbose:
            print_layer_info(i, layer)

        if transition < 0:
            device_type = trt.DeviceType.GPU if starts_gpu else trt.DeviceType.DLA
        elif starts_gpu:
            device_type = trt.DeviceType.GPU if i < transition else trt.DeviceType.DLA
        else:
            device_type = trt.DeviceType.DLA if i < transition else trt.DeviceType.GPU

        if verbose:
            print(f"Device is set {device_type}")
        config.set_device_type(layer, device_type)

        if i == network.num_layers - 1 or i == transition:
            network.mark_output(layer.get_output(0))


def print_layer_info(layer_index, layer):
    print(f"Layer {layer_index}: {layer.name}, Type: {layer.type}")


def build_and_return_engine(builder, network, config):
    return builder.build_engine(network, config)


def save_engine(engine, output_file):
    with open(output_file, "wb") as f:
        f.write(engine.serialize())
    print(f"Engine saved to {output_file}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from a Caffe prototxt file."
    )
    parser.add_argument(
        "--prototxt",
        type=str,
        required=True,
        help="Path to the input Caffe prototxt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path to save the output engine",
    )
    parser.add_argument(
        "--starts_gpu",
        type=bool,
        default=True,
        help="Whether the network starts on GPU (True) or DLA (False)",
    )
    parser.add_argument(
        "--transition",
        type=int,
        default=-1,
        help="Layer index where the transition occurs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def print_configuration_summary(prototxt_path, output, starts_gpu, transition, verbose):
    print("Configuration Summary:")
    print(f"Prototxt File: {prototxt_path}")
    print(f"Output Directory: {output}")
    print(f"Starts on GPU: {starts_gpu}")
    if transition == -1:
        print(f"Transition Layer not specified")
    else:
        print(f"Transition Layer Index: {transition}")
    print(f"Verbose Output: {verbose}")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    args = parse_arguments()
    if args.verbose:
        print_configuration_summary(
            prototxt_path=args.prototxt,
            output=args.output,
            starts_gpu=args.starts_gpu,
            transition=args.transition,
            verbose=args.verbose,
        )

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    engine = build_engine_caffe(
        deploy_file=args.prototxt,
        transition=args.transition,
        starts_gpu=args.starts_gpu,
        batch=1,
        verbose=args.verbose,
    )
    if engine:
        save_engine(engine, args.output)
        print(f"Saved Engine: {args.output}")
