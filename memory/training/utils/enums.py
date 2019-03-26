"""
Constants/enums for training.
Author: Vishal Satish
"""


# constants
"""
class DataConstants:
    ORIG_IM_DATASET_PATH = "/nfs/diskstation/dmwang/mech_search_data/originals"
"""

# file templates
class FileTemplates:
    # Keras model checkpoints
    MODEL_CKPT = "model_{epoch:02d}.hdf5"
    FINAL_MODEL_CKPT = "model.hdf5"
    
    TRAIN_HISTORY = "training_history.pkl"


# directory templates
class DirTemplates:
    LOG_DIR = "logs"

