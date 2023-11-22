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
            print(layer.name, " & i: ", i)
            latestLayer = i
        # print("Network size:",latestLayer)
        # output_layer=48

        print("OUTPUT")
        for i, layer in enumerate(network):
            print(
                layer.name,
                " and i:",
                i,
                " shape:",
                network.get_layer(i).get_output(0).shape,
            )
        # exit()

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
                if i == transition:
                    # print("output layer ", transition)
                    network.mark_output(model_tensors.find(layer.name))
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

        # network.mark_output(model_tensors.find("prob"))
        return builder.build_engine(network, config)
        # return 0


def save_engine(serialized_engine, save_file):
    with open(save_file, "wb") as f:
        f.write(serialized_engine)


if __name__ == "__main__":
    # TODO_ISMET, WRITE A FOR LOOP as the length of transition length.
    prototxt_files = glob.glob(f"prototxt_input_files/Inception_v3.prototxt")
    count = 0
    batch = 1
    transition = -1
    prototxt = "Inception_v3.prototxt"
    # network_stem="deploy_inception-resnet-v2"
    serialized_engine = build_engine_caffe(None, prototxt, 0, True, batch, transition)
    if serialized_engine is not None:
        save_engine(
            serialized_engine.serialize(),
            str("vgg19_gpu_transition_at_" + transition + ".plan"),
        )

    exit()
    serialized_engine = build_engine_caffe(None, prototxt, 1000, True, batch)
    if serialized_engine is not None:
        save_engine(
            serialized_engine.serialize(),
            str("baseline_DNNs/" + network_stem + "_dla.plan"),
        )

    exit()
    prototxt_files = glob.glob(
        f"/home/ismetdagli/Transition-Trt/mobilenet_yolov3_lite_deploy.prototxt"
    )
    prototxt_files = natsorted(prototxt_files)
    count = 0
    batch = 1
    prototxt = "mobilenet_yolov3_lite_deploy.prototxt"
    network_stem = "mobilenet_yolov3_lite_deploy"
    serialized_engine = build_engine_caffe(None, prototxt, 0, True, batch)
    if serialized_engine is not None:
        save_engine(
            serialized_engine.serialize(),
            str("baseline_DNNs/" + network_stem + "_gpu.plan"),
        )
    serialized_engine = build_engine_caffe(None, prototxt, 0, False, batch)
    if serialized_engine is not None:
        save_engine(
            serialized_engine.serialize(),
            str("baseline_DNNs/" + network_stem + "_dla.plan"),
        )

    exit()

    for i in {0, 10, 24, 38, 53, 67, 81, 95, 109, 124, 141}:
        serialized_engine = build_engine_caffe(None, prototxt, i, False, batch)
        if serialized_engine is not None:
            save_engine(
                serialized_engine.serialize(),
                str(
                    "googlenet_batch2/"
                    + network_stem
                    + "_dla"
                    + str(i)
                    + "_batch2.plan"
                ),
            )

    for i in {0, 10, 24, 38, 53, 67, 81, 95, 109, 124, 141}:
        serialized_engine = build_engine_caffe(None, prototxt, i, False, batch)
        if serialized_engine is not None:
            save_engine(
                serialized_engine.serialize(),
                str(
                    "googlenet_batch2/"
                    + network_stem
                    + "_dla"
                    + str(i)
                    + "_batch1.plan"
                ),
            )

    exit()
    for prototxt in prototxt_files:
        if count >= -1:
            starts_gpu = True
            print(str(prototxt))
            network_name = Path(str(prototxt))
            network_stem = network_name.stem
            print(network_stem)
            batch = 2
            serialized_engine = build_engine_caffe(
                None, prototxt, 2000, starts_gpu, batch
            )
            if serialized_engine is not None:
                save_engine(
                    serialized_engine.serialize(),
                    str(
                        "multiple_batch_engines/" + network_stem + "_maxBatch2_dla.plan"
                    ),
                )

            serialized_engine = build_engine_caffe(None, prototxt, 0, starts_gpu, batch)
            if serialized_engine is not None:
                save_engine(
                    serialized_engine.serialize(),
                    str(
                        "multiple_batch_engines/" + network_stem + "_maxBatch2_gpu.plan"
                    ),
                )
        count += 1

    # for current_index in {0,3,4,5,7,8,9,11,12,13,15,16,18,20,24}:
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"Alexnet_cityshape_sd_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("grouped_layer_networks_gpu/alexnet_gpu_"+str(current_index)+".plan"))
    #     # exit()

    # Resnet152 Gpu
    # for current_index in range(31,200,20):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_gpu/resnet152_gpu_"+str(current_index)+".plan"))
    #     # exit()
    # for current_index in range(200,600,20):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_gpu/resnet152_gpu_"+str(current_index)+".plan"))

    # for current_index in range(1,31,2):
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_dla/resnet152_dla_"+str(current_index)+".plan"))

    # for current_index in range(31,669,20):
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_dla/resnet152_dla_"+str(current_index)+".plan"))

    # #resNet18_deploy.prototxt
    # for current_index in {0,5,17,26,38,47,59,68,80,90}:
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resNet18_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("grouped_layer_networks_gpu/resnet18_gpu_"+str(current_index)+".plan"))
    #     # exit()
    # for current_index in {0,5,17,26,38,47,59,68,80,90}:
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resNet18_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("grouped_layer_networks_dla/resnet18_dla_"+str(current_index)+".plan"))

    # #vgg19_modified.prototxt
    # for current_index in {0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}:
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"vgg19_modified.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("grouped_layer_networks_gpu/vgg19_modified_gpu_"+str(current_index)+".plan"))
    #     # exit()
    # for current_index in {0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}:
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"vgg19_modified.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("grouped_layer_networks_dla/vgg19_modified_dla_"+str(current_index)+".plan"))

    # for current_index in range(50,600,10):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_gpu/resnet152_gpu_"+str(current_index)+".plan"))
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_dla/resnet152_dla_"+str(current_index)+".plan"))

    # for current_index in range(654,670,3):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_gpu/resnet152_gpu_"+str(current_index)+".plan"))
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet152.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet152_dla/resnet152_dla_"+str(current_index)+".plan"))

    # for current_index in range(0,92,3):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resNet18_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet18_gpu/resnet18_gpu_"+str(current_index)+".plan"))
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resNet18_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet18_dla/resnet18_dla_"+str(current_index)+".plan"))

    # for current_index in range(36,50,3):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet50_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet50_gpu/resnet50_gpu_"+str(current_index)+".plan"))
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet50_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet50_dla/resnet50_dla_"+str(current_index)+".plan"))

    # for current_index in range(50,180,10):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet50_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet50_gpu/resnet50_gpu_"+str(current_index)+".plan"))
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet50_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet50_dla/resnet50_dla_"+str(current_index)+".plan"))

    # for current_index in range(180,228,3):
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet50_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet50_gpu/resnet50_gpu_"+str(current_index)+".plan"))
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resnet50_deploy.prototxt", int(current_index),starts_gpu)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet50_dla/resnet50_dla_"+str(current_index)+".plan"))

    # for current_index in range(0,1):
    # starts_gpu=True
    # print( str(current_index))
    # serialized_engine = build_engine_caffe(None, f"facenet.prototxt", int(current_index),starts_gpu)
    # # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    # print(current_index)
    # index=str(current_index)
    # if serialized_engine is not None:
    #     save_engine(serialized_engine.serialize(), str("facenet_dla/facenet_dla"+str(current_index)+".plan"))

    # for current_index in (0,4,17,26,38,47,59,68,80,89,90):
    #     # current_index=101
    #     starts_gpu=True
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resNet18_deploy.prototxt", int(current_index),starts_gpu)
    #     # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    #     # print(current_index)
    #     # index=str(current_index)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet18_transition_cost_gpu/resnet18_"+str(current_index)+".plan"))

    # for current_index in (0,4,17,26,38,47,59,68,80,89,90):
    #     # current_index=101
    #     starts_gpu=False
    #     print( str(current_index))
    #     serialized_engine = build_engine_caffe(None, f"resNet18_deploy.prototxt", int(current_index),starts_gpu)
    #     # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    #     # print(current_index)
    #     # index=str(current_index)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("resnet18_transition_cost_dla/resnet18_"+str(current_index)+".plan"))

    # for current_index in {9}:
    #     for i in {15,16,18,20}:
    #         for j in {4,5,7,8,9,11,12,13,15,16,18,20,21}:
    #             starts_gpu=True
    #             if int(current_index) < i:
    #                 if i < j:
    #                     print( str(current_index)+ " and "+  str(i))
    #                     serialized_engine = build_engine_caffe(None, f"Alexnet_cityshape_sd_deploy.prototxt", int(current_index),i,j,starts_gpu)
    #                     # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    #                     # print(current_index)
    #                     # index=str(current_index)
    #                     time.sleep(3)
    #                     if serialized_engine is not None:
    # #                         save_engine(serialized_engine.serialize(), str("alexnet_3transitions/alexnet_gpu_"+str(current_index)+"_"+str(i)+"_"+str(j)+".plan"))
    # for current_index in {7}:
    #     for i in {15,16,18,20}:
    #         for j in {4,5,7,8,9,11,12,13,15,16,18,20,21}:
    #             starts_gpu=True
    #             if int(current_index) < i:
    #                 if i < j:
    #                     print( str(current_index)+ " and "+  str(i))
    #                     serialized_engine = build_engine_caffe(None, f"Alexnet_cityshape_sd_deploy.prototxt", int(current_index),i,j,starts_gpu)
    #                     # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    #                     # print(current_index)
    #                     # index=str(current_index)
    #                     time.sleep(3)
    #                     if serialized_engine is not None:
    #                         save_engine(serialized_engine.serialize(), str("alexnet_3transitions/alexnet_gpu_"+str(current_index)+"_"+str(i)+"_"+str(j)+".plan"))

    # for current_index in {3,4,5,7,8,9,11,12,13,15,16,18}:
    #     for i in {3,4,5,7,8,9,11,12,13,15,16,18,20}:
    #         for j in {4,5,7,8,9,11,12,13,15,16,18,20,21}:
    #             starts_gpu=False
    #             if int(current_index) < i:
    #                 if i < j:
    #                     print( str(current_index)+ " and "+  str(i))
    #                     serialized_engine = build_engine_caffe(None, f"convolution_test/conv1.prototxt", int(current_index),i,j,starts_gpu)
    #                     # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    #                     # print(current_index)
    #                     # index=str(current_index)
    #                     if serialized_engine is not None:
    #                         save_engine(serialized_engine.serialize(), str("convolution_test_plans/conv1.plan"))

    # serialized_engine = build_engine_caffe(None, f"convolution_test/conv1.prototxt", int(1),2,3,True)
    # # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    # # print(current_index)
    # # index=str(current_index)
    # if serialized_engine is not None:
    #     save_engine(serialized_engine.serialize(), str("convolution_test_plans/conv1_gpu.plan"))

    # serialized_engine = build_engine_caffe(None, f"convolution_test/conv1.prototxt", int(1),2,3,False)
    # # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    # # print(current_index)
    # # index=str(current_index)
    # if serialized_engine is not None:
    #     save_engine(serialized_engine.serialize(), str("convolution_test_plans/conv1_dla.plan"))

    # for current_index in {26,59,80}:
    #     for i in {38,47,59,68,80,89,90}:
    #         starts_gpu=True
    #         if int(current_index) < i:
    #             print( str(current_index)+ " and "+  str(i))
    #             serialized_engine = build_engine_caffe(None, f"resNet18_deploy.prototxt", int(current_index),i,starts_gpu)
    #             # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    #             # print(current_index)
    #             # index=str(current_index)
    #             if serialized_engine is not None:
    #                 save_engine(serialized_engine.serialize(), str("alexnet_2transitions/resnet18_gpu_"+str(current_index)+"_"+str(i)+".plan"))

    # current_index = sys.argv[1]
    # i = sys.argv[2]
    # # for current_index in range(39):
    # # for i in range(1,39):
    # starts_gpu=True
    # if int(current_index) < int(i):
    #     print( str(current_index)+ " and "+  str(i))
    #     serialized_engine = build_engine_caffe(None, f"vgg16_deploy.prototxt", int(current_index),int(i),0,starts_gpu)

    #     # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
    #     # print(current_index)
    #     # index=str(current_index)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine.serialize(), str("vgg16_2transition/vgg16_"+str(current_index)+"_"+str(i)+".plan"))
    # exit()


# import tensorrt as trt
# import sys, os
# from natsort import natsorted
# import glob
# from pathlib import Path

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# class CustomAlgorithmSelector(trt.IAlgorithmSelector):
#     def __init__(self):
#         trt.IAlgorithmSelector.__init__(self)

#     def report_algorithms(self, contexts, choices):
#         for choice in choices:
#             print(choice.algorithm_variant.implementation)

#     def select_algorithms(self, context, choices):
#         assert len(choices) > 0
#         return list(range(len(choices)))
# def build_engine_caffe(model_file, deploy_file,output_layer):
#     with trt.Builder(
#         TRT_LOGGER
#     ) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
#         config.max_workspace_size = 1 * 1 << 30
#         config.flags = 1 << int(trt.BuilderFlag.FP16)
#         config.DLA_core = 0
#         config.flags = (config.flags | 1 << int(trt.BuilderFlag.GPU_FALLBACK))

#         builder.max_batch_size = 8
#         #config.algorithm_selector = CustomAlgorithmSelector()

#         model_tensors = parser.parse(
#             deploy=deploy_file,
#             model=model_file,
#             network=network,
#             dtype=trt.float16,
#         )
#         # print("OUTPUT")
#         for i, layer in enumerate(network):
#             print(layer.name)
#         #     # print(network.get_layer(i).get_output(0).shape)


#         # print("\n \n \n INPUT")
#         # for i, layer in enumerate(network):
#         #     # print(layer.name)
#         #     print(network.get_layer(i).get_input(0).shape)
#         latestLayer=0
#         for i, layer in enumerate(network):
#             latestLayer=i


#         for i, layer in enumerate(network):
#             print("Marked as output:", i, layer.name)
#             config.set_device_type(layer, trt.DeviceType.DLA)
#             if i==latestLayer:
#                 network.mark_output(model_tensors.find(layer.name))


#         # network.mark_output(model_tensors.find("prob"))
#         # return builder.build_engine(network, config)
#         return 0


# def build_engine_caffe(model_file, deploy_file,output_layer):
#     with trt.Builder(
#         TRT_LOGGER
#     ) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
#         config.max_workspace_size = 1 * 1 << 31
#         config.flags = 1 << int(trt.BuilderFlag.FP16)
#         config.DLA_core = 0
#         config.flags = (config.flags | 1 << int(trt.BuilderFlag.GPU_FALLBACK))
#         print("test3")
#         builder.max_batch_size = 1
#         #config.algorithm_selector = CustomAlgorithmSelector()
#         print("test4")
#         model_tensors = parser.parse(
#             deploy=deploy_file,
#             model=model_file,
#             network=network,
#             dtype=trt.float16,
#         )
#         # print("OUTPUT")
#         # for i, layer in enumerate(network):
#             # print(layer.name)
#             # print(network.get_layer(i).get_output(0).shape)


#         # print("\n \n \n INPUT")
#         # for i, layer in enumerate(network):
#         #     # print(layer.name)
#         #     print(network.get_layer(i).get_input(0).shape)
#         latestLayer=0
#         for i, layer in enumerate(network):
#             latestLayer=i
#         # print(latestLayer)
#         print("test")

#         for i, layer in enumerate(network):
#             # print("Layer:", i, layer.name)
#             config.set_device_type(layer, trt.DeviceType.GPU)
#             # if (i <= output_layer) or ( i>= last_transition) :
#             #     # if i == current_index:
#             #     #     # print(layer.name)
#             #     #     #builder_config.set_device_type(layer, trt.DeviceType.DLA)
#             #     #     network.mark_output(model_tensors.find(layer.name))
#             #     config.set_device_type(layer, trt.DeviceType.GPU)
#                 # print("Order", i+1, "layer ", layer.name)
#                 # config.set_device_type(layer, trt.DeviceType.DLA)
#             # if i==output_layer or i==latestLayer :
#             if i==latestLayer:
#                 network.mark_output(model_tensors.find(layer.name))
#         print("test2")

#         # network.mark_output(model_tensors.find("temp_pool"))
#         return builder.build_engine(network, config)
#         # return 0


# def save_engine(serialized_engine, save_file):
#     with open(save_file, "wb") as f:
#         f.write(serialized_engine)


# if __name__ == "__main__":
#     current_index = sys.argv[1]

#     i=0
#     print( str(current_index)+ " and "+  str(i))
#     input_dir_path="resnet18_groups_prototxt/"
#     input_engines = glob.glob(f"{input_dir_path}/*.prototxt")
#     for engine in input_engines:

#         # engine = "resnet18_groups_prototxt/group_26_38.prototxt"
#         print(engine)
#         # if engine == "resnet18_groups_prototxt/group_47_59.prototxt":
#         #     continue

#         # engine="resnet18_prototxt/resnet15.prototxt"
#         test_network = Path(str(engine))
#         network_name = test_network.stem
#         print(network_name)
#         serialized_engine = build_engine_caffe(None, engine, int(current_index))
#         # serialized_engine = build_engine_caffe(None, f"vgg19_{current_index}.prototxt",int(current_index))
#         # print(current_index)
#         # index=str(current_index)
#         if serialized_engine is not None:
#             save_engine(serialized_engine.serialize(), "resnet18_batch1_group_layers_gpu/"+network_name+".plan")
#         # if engine == "resnet18_groups_prototxt/group_59_68.prototxt":
#         #     exit()
#         # exit()
