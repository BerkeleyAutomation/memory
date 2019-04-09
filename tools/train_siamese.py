"""
Script for training a Siamese Network.

Author: Jeff Mahler, Vishal Satish
"""

import argparse
import os
import time

import autolab_core.utils as utils
from autolab_core import YamlConfig, Logger
from memory import SiameseNet, SiameseTrainer

# setup logger
logger = Logger.get_logger("tools/train_siamese.py")

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Train a Siamese Network in Keras.")
    parser.add_argument("dataset_dir", type=str, default=None,
                        help="path to the base dataset (contains 'originals', 'train', 'validation')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="path to store the trained model")
    parser.add_argument("--tensorboard_port", type=int, default=6006,
                        help="port to launch tensorboard on")
    parser.add_argument("--config_filename", type=str, default=None,
                        help="path to the configuration file to use")
    parser.add_argument("--name", type=str, default=None,
                        help="name for the trained model")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    tensorboard_port = args.tensorboard_port
    config_filename = args.config_filename
    name = args.name
    
    # set default output dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../models")
    
    # set default config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "..",
                                       "cfg/tools/train_siamese.yaml")

    # turn relative paths absolute
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.getcwd(), dataset_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # create output dir if necessary
    utils.mkdir_safe(output_dir)
        
    # open train config
    train_config = YamlConfig(config_filename)
    train_config["tensorboard_port"] = tensorboard_port
    network_params = train_config["siamese_net"]

    if name is None:
        # create a unique model name with a timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        name = "model_{}".format(timestamp)

    # train the network
    start_time = time.time()
    network = SiameseNet(network_params)
    trainer = SiameseTrainer(network,
                             dataset_dir,
                             output_dir,
                             train_config,
                             model_name=name)
    trainer.train()
    logger.info("Total Training Time:" + str(utils.get_elapsed_time(time.time() - start_time)))
