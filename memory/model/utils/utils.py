"""
Utility functions for networks.
Authors: Vishal Satish, Kate Sanders, 
         David Wang, Michael Danielczuk, 
         Matthew Matl
"""
import tensorflow as tf
from tensorflow.python.client import device_lib as tfdl
import keras.backend as kb
from keras.backend.tensorflow_backend import set_session


# distance metrics for use in a Keras Lambda layer
def l2_distance(vects):
    x, y = vects
    return kb.sqrt(kb.sum(kb.square(x - y), axis=1, keepdims=True) + 1e-10)


def l1_distance(vects):
    x, y = vects
    return kb.sum(kb.abs(x - y), axis=1, keepdims=True)

def threshold(vect):
    return kb.sigmoid(vect)

# Tensorflow helper functions
session_has_been_set = False
def setup_tf_session():
    global session_has_been_set
    assert not session_has_been_set, "You are resetting the TF session-are you sure you want to do that?"
    session_has_been_set = True

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    set_session(sess)


def get_available_gpus():
    return [d.name for d in tfdl.list_local_devices() if d.device_type == "GPU"]

