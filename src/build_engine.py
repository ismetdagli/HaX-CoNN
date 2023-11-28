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


def build_engine_for_directory(input_dir, output_dir, starts_gpu, verbose):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for prototxt_file in input_dir.glob("*.prototxt"):
        output_file = output_dir / prototxt_file.with_suffix(".plan").name

        print(f"Started build of {prototxt_file}")
        engine = build_engine_caffe(
            deploy_file=str(prototxt_file),
            transition=-1,
            starts_gpu=starts_gpu,
            batch=1,
            verbose=verbose,
        )
        if engine:
            save_engine(engine, str(output_file))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from a Caffe prototxt file."
    )
    parser.add_argument(
        "--prototxt",
        type=str,
        required=True,
        help="Path to the input Caffe prototxt file or a directory for bulk build",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path to save the output engine or directory for bulk build",
    )
    parser.add_argument(
        "--start",
        choices=["gpu", "dla"],
        required=True,
        help="Specify whether to start on GPU or DLA",
    )
    parser.add_argument(
        "--transition",
        type=int,
        default=-1,
        help="Layer index where the transition occurs. Omit the option if a single device will be used.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def print_configuration_summary(prototxt_path, output, starts_at, transition, verbose):
    print("Configuration Summary:")
    print(f"Prototxt File: {prototxt_path}")
    print(f"Output Directory: {output}")
    print(f"Starts on: {starts_at}")
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
            starts_at=args.start,
            transition=args.transition,
            verbose=args.verbose,
        )

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.start == "gpu":
        starts_gpu = True
    elif args.start == "dla":
        starts_gpu = False
    else:
        raise ValueError("Unreachable, set start to gpu or dla {args.start}")

    prototxt_path = Path(args.prototxt)
    output_path = Path(args.output)

    if prototxt_path.is_dir():
        # Handle directory input
        if args.transition != -1:
            raise ValueError("Transition is not supported for bulk build")
        print("Input path is a directory, bulk building initiated")
        build_engine_for_directory(
            input_dir=prototxt_path,
            output_dir=output_path,
            starts_gpu=starts_gpu,
            verbose=args.verbose,
        )
    else:
        # Single file processing
        engine = build_engine_caffe(
            deploy_file=args.prototxt,
            transition=args.transition,
            starts_gpu=starts_gpu,
            batch=1,
            verbose=args.verbose,
        )
        if engine:
            save_engine(engine, args.output)
