"""
Script for generating ResNet50Fused featurizations of images.

Authors: Vishal Satish, Kate Sanders, 
         David Wang, Michael Danielczuk, 
         Matthew Matl
"""
import argparse
import os

import skimage.io as skio
import numpy as np

from autolab_core import Logger

from memory.model import ResNet50Fused

# setup logger
logger = Logger.get_logger("tools/featurize_images_resnet50fused.py")

DEFAULT_MODEL_WEIGHTS = "/nfs/diskstation/vsatish/dex-net/data/memory/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"  

def get_image_shape(image_dir):
    #NOTE: we do a DF traversal instead of BF because although this script is designed for cases that could have images and directories together at the same level, the datasets we use have images only at the lowest level
    root_dir = image_dir
    item = os.listdir(root_dir)[0]
    full_item_path = os.path.join(root_dir, item)
    while os.path.isdir(full_item_path):
        root_dir = os.path.join(root_dir, item)
        item = os.listdir(root_dir)[0]
        full_item_path = os.path.join(root_dir, item)
    im = skio.imread(full_item_path)
    return im.shape


def featurize_dir(input_dir, output_dir, network):
    logger.info("Featurizing '{}'...".format(input_dir))

    contents = os.listdir(input_dir)
    for item in contents:
        full_item_path = os.path.join(input_dir, item)
        if os.path.isdir(full_item_path):
            # make the corresponding directory in output_dir and recursively featurize it
            full_output_dir = os.path.join(output_dir, item)
            os.mkdir(full_output_dir)
            featurize_dir(full_item_path, full_output_dir, network)
        else:
            # featurize and save the image
            logger.info("Featurizing '{}'.".format(full_item_path))

            im = skio.imread(full_item_path)
            features = np.squeeze(network.predict(im[None, :], bsz=1, verbose=False))
            im_base_name, _ = os.path.splitext(item)
            np.savez_compressed(os.path.join(output_dir, im_base_name), features)


def featurize(image_dir, output_dir, model_weights):
    # get image shape
    logger.info("Finding image shape...")
    im_shape =  get_image_shape(image_dir)

    # load ResNet50Fused
    logger.info("Loading ResNet50Fused model...")
    net = ResNet50Fused.load(im_shape, weights=model_weights)

    # recursively iterate over image_dir and featurize images, storing them in an identical directory structure under output_dir
    featurize_dir(image_dir, output_dir, net)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Generate ResNet50Fused featurizations for images.")
    parser.add_argument("image_dir", type=str, default=None,
                        help="path to the images to featurize")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="path to save the features to")
    parser.add_argument("--model_weights", type=str, default=DEFAULT_MODEL_WEIGHTS,
                        help="path to pre-trained ResNet50Fused model weights")

    args = parser.parse_args()
    model_weights = args.model_weights
    image_dir = args.image_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(image_dir, "featurized_images")
        logger.warning("No output dir specified, saving images to '{}'.".format(output_dir))
        os.mkdir(output_dir)

    featurize(image_dir, output_dir, model_weights)
           
