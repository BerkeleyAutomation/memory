"""
Siamese network implemented in Keras.

Authors: Vishal Satish, Kate Sanders, 
         David Wang, Michael Danielczuk, 
         Matthew Matl
"""
import os
from collections import OrderedDict
import json

import tensorflow as tf
import keras.layers as kl
import keras.models as km

from autolab_core import Logger

from memory.model.utils import (NetworkMode, InputMode, 
                                 get_available_gpus, setup_tf_session, 
                                 l1_distance, l2_distance)
from memory.training.utils import FileTemplates
from memory.model import ResNet50Fused


class SiameseNet(object):
    """ Siamese network implemented in Keras. """

    def __init__(self, config, verbose=True, log_file=None):
        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__, 
                                         log_file=log_file, 
                                         silence=(not verbose), 
                                         global_log_file=verbose)

        #TODO:(vsatish) Figure out why this needs to go before tensorflow.python.client.list_local_devices() in get_available_gpus()
        setup_tf_session()

        # read config
        self._parse_config(config)


    @staticmethod
    def load(model_dir, verbose=True, log_file=None):
        training_config_file = os.path.join(model_dir, FileTemplates.CONFIG_FILENAME)
        with open(training_config_file) as fhandle:    
            training_config = json.load(fhandle, object_pairs_hook=OrderedDict)
        network_config = training_config["siamese_net"]

        network = SiameseNet(network_config, verbose=verbose, log_file=log_file)
        network.initialize_network()
        
        saved_model = os.path.join(model_dir, FileTemplates.FINAL_MODEL_CKPT)
        network.load_trained_weights(saved_model)

        return network


    def load_trained_weights(self, model):
        self._logger.info("Loading pre-trained weights...")
        self._model.load_weights(model)


    def _parse_config(self, config):
        self._network_mode = config["network_mode"]
        assert self._network_mode in [NetworkMode.TRAINING, NetworkMode.INFERENCE], ("Invalid network "
                                                                                     "mode: {}.".format(self._mode))
        self._input_mode = config["input_mode"]
        assert self._input_mode in [InputMode.IMAGE, InputMode.FEATURE], ("Invalid input "
                                                                            "mode {}.".format(self._input_mode))

        self._num_gpus = config["num_gpus"]
        assert self._num_gpus <= 2, "Max 2 GPUs (1 per siamese network) are supported."
        self._avail_gpus = get_available_gpus()
        assert self._num_gpus <= len(self._avail_gpus), "More GPUs requested than available."

        self._input_shape = config["input_shape"]
        if self._input_mode == InputMode.FEATURE:
            assert len(self._input_shape) == 1, "Incorrect input shape dim for feature inputs."
        else:
            assert len(self._input_shape) == 3, "Incorrect input shape dim for image inputs."

        self._architecture = config["architecture"]


    def initialize_network(self):
        self._input_1 = kl.Input(shape=self._input_shape, name="input_1")
        self._input_2 = kl.Input(shape=self._input_shape, name="input_2")
        self._model = self._build_network(self._input_1, self._input_2, name="siamese network")
        self._model.summary() #TODO:(vsatish) Remove eventually.


    def predict(self, input_pairs, bsz=32, verbose=True):
        return self._model.predict(input_pairs, batch_size=bsz, verbose=verbose)


    @property
    def model(self):
        return self._model


    @property
    def input_shape(self):
        return self._input_shape


    @property
    def input_mode(self):
        return self._input_mode


    def _build_input_stream(self, input_node, layer_dict, name=None, layers=None):
        if name is None:
            self._logger.info("Building input stream...")
        else:
            self._logger.info("Building {}...".format(name))

        if layers is None:
            # build layers from scratch
            self._logger.info("Building layers from scratch...")

            layers = []
            prev_layer = "start" # dummy placeholder
            last_index = len(layer_dict.keys()) - 1
            for layer_index, (layer_name, layer_config) in enumerate(layer_dict.items()):
                layer_type = layer_config["type"]

                print(layer_config.keys())

                if layer_type == "conv":
                    self._logger.info("Building convolution layer: {}...".format(layer_name))

                    assert prev_layer != "fc", "Cannot have conv layer after fc layer."
                    assert self._input_mode == InputMode.IMAGE, "Can only have conv layers if inputs are images."

                    if prev_layer == "start":
                        layers.append(kl.Conv2D(layer_config["num_filt"], layer_config["filt_dim"], padding=layer_config["pad"], activation="relu", input_shape=self._input_shape, name=layer_name))
                    else:
                        layers.append(kl.Conv2D(layer_config["num_filt"], layer_config["filt_dim"], padding=layer_config["pad"], activation="relu", name=layer_name))
                elif layer_type == "fc":
                    self._logger.info("Building fully connected layer: {}...".format(layer_name))

                    if prev_layer == "start":
                        assert self._input_mode == InputMode.FEATURE, "Can only have fc layer first if inputs are features."
                    elif prev_layer == "conv":
                        layers.append(kl.Flatten())

                    if layer_index == last_index:
                        # we don't want an activation for the final layer
                        layers.append(kl.Dense(layer_config["out_size"], name=layer_name))
                    else:
                        layers.append(kl.Dense(layer_config["out_size"], activation="relu", name=layer_name))
                elif layer_type == "resnet50f":
                    self._logger.info("Building ResNet50Fused layer: {}...".format(layer_name))

                    assert prev_layer == "start", "ResNet50Fused layer must be first."
                    assert self._input_mode == InputMode.IMAGE, "ResNet50Fused layer inputs must be images."
                    assert layer_index != last_index, "ResNet50Fused layer cannot be last." #NOTE: Ask vishal about this if you're confused.
                    layers.append(ResNet50Fused.load(self._input_shape, weights=layer_config["weights"], trainable=layer_config["trainable"], setup_session=False).model)
                else:
                    raise ValueError("Layer type: {} unsupported in input stream.".format(layer_type))
                prev_layer = layer_type
        else:
             # reuse layers
            self._logger.info("Reusing layers...")

        # build stream
        output_node = input_node
        for layer in layers:
            output_node = layer(output_node)
        return output_node, layers


    def _build_merge_stream(self, input_node_1, input_node_2, layer_dict, name=None):
        if name is None:
            self._logger.info("Building merge stream...")
        else:
            self._logger.info("Building {}...".format(name))

        prev_layer = "start" # dummy placeholder
        last_index = len(layer_dict.keys()) - 1
        output_node = None
        for layer_index, (layer_name, layer_config) in enumerate(layer_dict.items()):
            layer_type = layer_config["type"]

            if layer_type == "fc":
                self._logger.info("Building fully connected layer: {}...".format(layer_name))

                assert prev_layer in ["start", "fc"], "Cannot have fully connected layer after {} layer.".format(prev_layer)

                if prev_layer == "start":
                    # the input streams have no activation, so we add one
                    input_node_1 = kl.Activation("relu")(input_node_1)
                    input_node_2 = kl.Activation("relu")(input_node_2)
                    
                    # concat
                    output_node = kl.Concatenate(axis=1)([input_node_1, input_node_2])

                if layer_index == last_index: 
                    assert layer_config["out_size"] == 1, "Output size of final fully connected layer must be 1."
                    output_node = kl.Dense(layer_config["out_size"], name=layer_name)(output_node)
                else:
                    output_node = kl.Dense(layer_config["out_size"], activation="relu", name=layer_name)(output_node)
            elif layer_type == "l1":
                self._logger.info("Building l1 distance layer: {}...".format(layer_name))

                assert prev_layer == "start", "l1 distance layer must be only layer."
                output_node = kl.Lambda(l1_distance, name=layer_name)([input_node_1, input_node_2])
            elif layer_type == "l2":
                self._logger.info("Building l2 distance layer: {}...".format(layer_name))

                assert prev_layer == "start", "l2 distance layer must be only layer."
                output_node = kl.Lambda(l2_distance, name=layer_name)([input_node_1, input_node_2])
            else:
                raise ValueError("Layer type: {} unsupported in merge stream.".format(layer_type))
            prev_layer = layer_type
        return output_node


    def _build_network(self, input_node_1, input_node_2, name="Siamese Network"):
        self._logger.info("Building {}...".format(name))

        if self._num_gpus == 2:
            # split input streams across GPUs
            with tf.device(self._avail_gpus[0]):
                input_stream_1_out, input_stream_layers = self._build_input_stream(input_node_1, self._architecture["input_stream"], name="input stream 1")
            with tf.device(self._avail_gpus[1]):
                input_stream_2_out, _ = self._build_input_stream(input_node_2, self._architecture["input_stream"], name="input stream 2", layers=input_stream_layers) #TODO:(vsatish) Confirm that this still reuses weights
        else:
            input_stream_1_out, input_stream_layers = self._build_input_stream(input_node_1, self._architecture["input_stream"], name="input stream 1")
            input_stream_2_out, _ = self._build_input_stream(input_node_2, self._architecture["input_stream"], name="input stream 2", layers=input_stream_layers)

        merge_stream_out = self._build_merge_stream(input_stream_1_out, input_stream_2_out, self._architecture["merge_stream"])
        return km.Model(inputs=(input_node_1, input_node_2), outputs=merge_stream_out, name=name)

