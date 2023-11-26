import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_prototxt(deploy_file):
    real_layers = set()
    layer_types_to_exclude = {"Dropout", "Concat", "LayerType.CONCATENATION", "Input",}

    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        parser.parse(deploy=str(deploy_file), model=None, network=network, dtype=trt.float16)

        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if str(layer.type) not in layer_types_to_exclude:
                real_layers.add(layer.name)

    return real_layers
