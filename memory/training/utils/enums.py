"""
Constants/enums for training.
Author: Vishal Satish
"""


# general constants
class GeneralConstants:
    JSON_INDENT = 2


# file templates
class FileTemplates:
    # Keras model checkpoints
    MODEL_CKPT = "model_epoch_{epoch:02d}.hdf5"
    FINAL_MODEL_CKPT = "model.hdf5"
    
    TRAIN_HISTORY = "training_history.pkl"
    CONFIG_FILENAME = "config.json"
    

# directory templates
class DirTemplates:
    LOG_DIR = "logs"

