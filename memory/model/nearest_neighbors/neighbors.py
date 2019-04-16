"""
Generic nearest neighbors implementation using NearPy.

Authors: Kate Sanders, Vishal Satish
"""
from nearpy import Engine
from nearpy.filters import NearestFilter

from autolab_core import Logger


class Neighbors:
    """ Nearest neighbors. """

    def __init__(self, config, verbose=True, log_file=None):
        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__, 
                                         log_file=log_file, 
                                         silence=(not verbose), 
                                         global_log_file=verbose)

        # read config
        self._parse_config(config)

        self._engine = None


    def _parse_config(self, config):
        self._num_neighbors = config["num_neighbors"]


    def _build_engine(self, dimension):
        # build NearPy engine
        self._logger.info("Building engine...")
        self._engine = Engine(dimension, vector_filters=[NearestFilter(self._num_neighbors)])


    def store(self, vectors, data=None, log_freq=10, verbose=True):
        self._logger.info("Storing vectors...")
        if data is not None:
            assert vectors.shape[0] == len(data), "Dim 0 of vectors and data must match!"

        if self._engine is None:
            self._build_engine(vectors.shape[-1])

        num_vectors = vectors.shape[0]
        for idx in xrange(num_vectors):
            if verbose and idx % log_freq == 0:
                self._logger.info("Storing vector {} of {}...".format(idx, num_vectors))
            if data is not None:
                self._engine.store_vector(vectors[idx], data[idx])
            else:
                self._engine.store_vector(vectors[idx])

    def predict(self, vectors, log_freq=10, verbose=True):
        self._logger.info("Predicting...")

        num_vectors = vectors.shape[0]
        neighbors = []
        for idx in xrange(num_vectors):
            if verbose and idx % log_freq == 0:
                self._logger.info("Predicting vector {} of {}...".format(idx, num_vectors))
            neighbors.append(self._engine.neighbours(vectors[idx]))
        return neighbors

