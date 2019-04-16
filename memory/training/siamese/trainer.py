"""
Trains a Siamese Network. Implemented in Keras.
Authors: Vishal Satish, Kate Sanders, 
         David Wang, Michael Danielczuk, 
         Matthew Matl
"""
import os
import cPickle as pkl
import multiprocessing as mp
from collections import OrderedDict
import json

import keras.callbacks as kc
import keras.optimizers as ko

from autolab_core import Logger

from memory.training.utils import (FileTemplates, DirTemplates, GeneralConstants,
                                   ImageDataset, DataGenerator,
                                   build_contrastive_loss)


class SiameseTrainer(object):
    """ Trains a Siamese Network. Implemented in Keras. """

    def __init__(self, siamese_net, dataset_dir, output_dir, config, model_name=None, progress_dict=None, log_file=None, verbose=True):
        self._network = siamese_net
        self._dataset_dir = dataset_dir
        self._output_dir = output_dir
        self._model_name = model_name
        self._progress_dict = progress_dict

        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__, 
                                         log_file=log_file, 
                                         silence=(not verbose), 
                                         global_log_file=verbose)

        # read config
        self._parse_config(config)


    def _parse_config(self, config):
        self._cfg = config

        # training
        self._num_epochs = config["num_epochs"]
        self._loss_margin = config["loss_margin"]
        self._optimizer_config = config["optimizer"]
        self._reg_coeff = config["reg_coeff"]
        self._drop_rate = config["drop_rate"]
        self._tensorboard_port = config["tensorboard_port"]

        # data
        self._bsz = config["bsz"]
        self._shuffle_training_inputs = config["shuffle_training_inputs"]

        self._num_prefetch_workers = config["num_prefetch_workers"]
        assert self._num_prefetch_workers <= mp.cpu_count(), "More workers than available logical cores."
        
        self._num_train_pairs = config["num_train_pairs"]
        self._num_val_pairs = config["num_val_pairs"]
        self._data_augmentation_suffixes = config["data_augmentation_suffixes"]
        self._allow_different_views = config["allow_different_views"]


    def _launch_tensorboard(self):
        """ Launches Tensorboard to visualize training. """
        FNULL = open(os.devnull, "w")
        self.logger.info(
            "Launching Tensorboard, Please navigate to localhost:{} in your favorite web browser to view summaries".format(self._tensorboard_port))
        self._tensorboard_proc = subprocess.Popen(["tensorboard", "--port", str(self._tensorboard_port),"--logdir", os.path.join(self._model_dir, DirTemplates.LOG_DIR)], stdout=FNULL)


    def _close_tensorboard(self):
        """ Closes Tensorboard process. """
        self._logger.info("Closing Tensorboard...")
        self._tensorboard_proc.terminate()                        


    def _build_loss(self):
        return build_contrastive_loss(margin=self._loss_margin)


    def _build_optimizer(self):
        if self._optimizer_config["type"] == "Adam":
            return ko.Adam(lr=self._optimizer_config["lr"])


    def train(self):
        # set up for training
        self._setup()

        # build the network
        self._network.reg_coeff = self._reg_coeff # set the network l2 weight regularization coefficient
        self._network.drop_rate = self._drop_rate # set the network dropout rate
        self._network.initialize_network()

        # optimize
        self._network.model.compile(loss=self._build_loss(),
                                    optimizer=self._build_optimizer(),
                                    metrics=["acc"])

        callbacks = [
            kc.TensorBoard(log_dir=os.path.join(self._model_dir, DirTemplates.LOG_DIR), histogram_freq=0, write_graph=True, write_images=False),
            kc.ModelCheckpoint(os.path.join(self._model_dir, FileTemplates.MODEL_CKPT), verbose=0, save_weights_only=True)
        ]
        #TODO:(vsatish) Automate tensorboard launch.

        history = self._network.model.fit_generator(generator=self._train_gen,
                                                    validation_data=self._val_gen,
                                                    epochs=self._num_epochs,
                                                    use_multiprocessing=True,
                                                    callbacks=callbacks,
                                                    workers=self._num_prefetch_workers)

        # save the training history
        with open(os.path.join(self._model_dir, FileTemplates.TRAIN_HISTORY), "wb") as fhandle:
            pkl.dump(history.history, fhandle)

        # save the final model
        self._network.model.save_weights(os.path.join(self._model_dir, FileTemplates.FINAL_MODEL_CKPT))


    def _prepare_datasets(self):
        self._logger.info("Preparing datasets...")

        train_dataset = ImageDataset(self._dataset_dir, "train", self._data_augmentation_suffixes, self._allow_different_views)
        train_dataset.prepare(self._num_train_pairs)

        val_dataset = ImageDataset(self._dataset_dir, "validation")
        val_dataset.prepare(self._num_val_pairs)

        return train_dataset, val_dataset


    def _build_data_generators(self, train_dataset, val_dataset):
        self._logger.info("Building data generators...")

        train_generator = DataGenerator(train_dataset, batch_size=self._bsz,
                                                       dim=self._network.input_shape,
                                                       shuffle=self._shuffle_training_inputs,
                                                       dataset_type=self._network.input_mode)
        val_generator = DataGenerator(val_dataset, batch_size=self._bsz,
                                                   dim=self._network.input_shape,
                                                   shuffle=self._shuffle_training_inputs,
                                                   dataset_type=self._network.input_mode)

        return train_generator, val_generator


    def _create_output_dir(self):
        self._logger.info("Creating output dir...")

        # create the output dir
        self._model_dir = os.path.join(self._output_dir, self._model_name)
        os.mkdir(self._model_dir)

        # save the training config to the output dir
        self._logger.info("Saving training config...")

        # copy some extra metadata to the config
        self._cfg["dataset_dir"] = self._dataset_dir

        # save
        cfg_save_fname = os.path.join(self._model_dir, FileTemplates.CONFIG_FILENAME)
        cfg_save_dict = OrderedDict()
        for key in self._cfg.keys():
            cfg_save_dict[key] = self._cfg[key]
        with open(cfg_save_fname, "w") as fhandle:
            json.dump(cfg_save_dict,
                      fhandle,
                      indent=GeneralConstants.JSON_INDENT)


    def _setup(self):
        self._logger.info("Setting up for training...")

        # create output dir
        self._create_output_dir()

        # prepare train and val datasets
        train_dataset, val_dataset = self._prepare_datasets()

        # build train and val data generators
        self._train_gen, self._val_gen = self._build_data_generators(train_dataset, val_dataset)

