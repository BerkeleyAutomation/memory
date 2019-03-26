"""
Utility functions for training.
Authors: Vishal Satish, Kate Sanders, 
         David Wang, Michael Danielczuk, 
         Matthew Matl
"""
import keras.backend as kb


# losses
def build_contrastive_loss(margin=1.0):
    def contrastive_loss(y_true, y_pred):
        # 1 is same, 0 is diff
        return kb.mean((1 - y_true) * 0.5 * kb.square(kb.maximum(margin - y_pred, 0)) + y_true * 0.5 * kb.square(y_pred))
    return contrastive_loss

