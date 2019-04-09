import os
import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter


class NeighborGenerator:
    def __init__(self, config, cache_ims, cache_labels):
        self.dimension = config["dimension"]
        self.distance_fxn = config["distance"]
        self.num_neighbors = config["num_neighbors"]
        self.engine = Engine(self.dimension, vector_filters=[NearestFilter(self.num_neighbors)])

        for index in range(len(cache_ims)):
            self.engine.store_vector(cache_ims[index], cache_labels[index])

    def predict(self, imgs):
        neighbor_list = []
        for image in imgs:
            neighbors = self.engine.neighbours(image)
            neighbor_list.append(neighbors)
        return neighbor_list
