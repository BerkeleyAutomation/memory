from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
import numpy as np
import os

class NeighborGenerator:
    def __init__(self, dimension=9984, num_neighbors=10, distance_fxn=None):
        self.dimension = dimension
        self.num_neighbors = num_neighbors
        self.distance_fxn = distance_fxn
        if not distance_fxn:
            self.engine = Engine(9984)

    def load_data(self, dir):
        # expects dir to be in same format that siamese network uses, but no view folder
        # takes in npz files produced by featurization script
        for obj in os.listdir(dir):
            obj_dir = os.path.join(os.path.join(dir, obj), 'view_000000')
            for file in os.listdir(obj_dir):
                image = os.path.join(obj_dir, file)
                image = np.load(image)['arr_0']
                self.engine.store_vector(image, obj)

    def predict(self, file_location):
        # takes in npz file from featurization script
        image = np.load(file_location)['arr_0']
        return self.engine.neighbours(image)

    def batch_predict(self, dir):
        neighbor_list = []
        for obj in os.listdir(dir):
            obj_dir = os.path.join(os.path.join(dir, obj), 'view_000000')
            for file in os.listdir(obj_dir):
                image_file = os.path.join(obj_dir, file)
                image = np.load(image_file)['arr_0']
                neighbors = self.engine.neighbours(image)
                neighbor_list.append([image_file, neighbors])
        return neighbor_list
