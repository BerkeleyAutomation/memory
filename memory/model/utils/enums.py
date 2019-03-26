"""
Constants/enums for networks.
Author: Vishal Satish
"""


# enum for training vs. inference
class NetworkMode:
    TRAINING = "training"
    INFERENCE = "inference"


# enum for predicting on features vs. images
class InputMode:
    FEATURE = "feature"
    IMAGE = "image"

