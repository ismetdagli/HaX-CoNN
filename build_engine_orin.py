# THIS IS A SCRIPT FOR ORIN AGX, NOT XAVIER AGX. SOFTWARES ARE NOT INCOMPATIBLE.


import tensorrt as trt
import sys, os
import logging
from pathlib import Path
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    # def set_image_batcher(self, image_batcher: ImageBatcher):
    #     """
    #     Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
    #     to be defined.
    #     :param image_batcher: The ImageBatcher object
    #     """
    #     self.image_batcher = image_batcher
    #     size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
    #     self.batch_allocation = cuda.mem_alloc(size)
    #     self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


def build_engine_caffe(model_file, deploy_file,runs_on_gpu,trans_layer,calibration_files):
    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
        NUM_IMAGES_PER_BATCH = 1
        # batchstream = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files.read)
        # Int8_calibrator = PythonEntropyCalibrator(["data"], batchstream)
        # config.max_workspace_size = 1 * 1 << 30
        # config.flags = 1 << int(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = EngineCalibrator(calibration_files)
        # config.flags = 1 << int(trt.BuilderFlag.FP16)
        # config.int8_calibrator = calibration_files
        config.DLA_core = 0
        config.flags = (config.flags | 1 << int(trt.BuilderFlag.GPU_FALLBACK))
        # config.flags = (config.flags | 1 << int(trt.BuilderFlag.FP16))

        builder.max_batch_size = 1

        model_tensors = parser.parse(
            deploy=deploy_file,
            model=model_file,
            network=network,
            dtype=trt.float32,
        )

        #CHANGE THIS FOR GPU
        if runs_on_gpu:
            device_type=trt.DeviceType.GPU
        else:
            device_type=trt.DeviceType.DLA

        for i, layer in enumerate(network):
            config.set_device_type(layer, trt.DeviceType.GPU)
            if runs_on_gpu :
                config.set_device_type(layer, trt.DeviceType.DLA)  #DLA
                if (i <= trans_layer) :
                    config.set_device_type(layer, trt.DeviceType.GPU)  
                # if i==latestLayer:
                #     network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
            # if i==latestLayer:
            #     network.mark_output(model_tensors.find(layer.name))
            else:
                config.set_device_type(layer, trt.DeviceType.GPU) 
                if (i <= trans_layer) :
                    config.set_device_type(layer, trt.DeviceType.DLA) #DLA
                # if i==latestLayer:
                #     network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
                    # network.mark_output(model_tensors.find(layer.name))  
                # if i==output_layer:
                #     # print(output_layer)
                #     network.mark_output(model_tensors.find(layer.name))

        # network.mark_output(model_tensors.find("prob"))
        # return 0
        network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
        return builder.build_serialized_network(network, config)

def save_engine(serialized_engine, save_file):
    with open(save_file, "wb") as f:
        f.write(serialized_engine) 

if __name__ == "__main__":
    # calibration_files="calibrator_networks/vgg19_calibration"
    # nn_protoxt_path = "prototxts/vgg19_modified.prototxt" # This variable must be modified for the correct PATH.

    # for trans_layer in [0]:
    #     runs_on_gpu=True # True for GPU, False for DLA
    #     serialized_engine = build_engine_caffe(None, nn_protoxt_path,runs_on_gpu,trans_layer,calibration_files)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine, "vgg19_gpu/vgg19_gpu_"+str(trans_layer)+".engine")
    #     # exit()
    #     runs_on_gpu=False # True for GPU, False for DLA
    #     serialized_engine = build_engine_caffe(None, nn_protoxt_path,runs_on_gpu,trans_layer,calibration_files)
    #     if serialized_engine is not None:
    #         save_engine(serialized_engine, "vgg19_dla/vgg19_dla_"+str(trans_layer)+".engine")


    calibration_files="calibrator_networks/googlenet_calibration"
    nn_protoxt_path = "prototxts/resNet-101_deploy.prototxt" # This variable must be modified for the correct PATH.


    count = 0
    batch = 1
    transition = -1
    prototxt = "prototxt_input_files/googlenet.prototxt"
    serialized_engine = serialized_engine = build_engine_caffe(None, nn_protoxt_path,True,transition,calibration_files)
    save_engine(
        serialized_engine,
        str("google_only_gpu.plan"),
    )
    transition = 141
    serialized_engine = serialized_engine = build_engine_caffe(None, nn_protoxt_path,False,transition,calibration_files)
    save_engine(
        serialized_engine,
        str("google_only_dla.plan"),
    )