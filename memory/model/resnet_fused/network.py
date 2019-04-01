"""
ResNet50 model in Keras with fused intermediate layers. Backend
is assumed to be TensorFlow and top fully connected layer is removed.

Adapted from: https://arxiv.org/pdf/1604.00133.pdf.

Author: Vishal Satish
"""
import keras.backend as kb
import keras.layers as kl
import keras.models as km

from autolab_core import Logger

from memory.model.utils import setup_tf_session


class ResNet50Fused(object):
    """ ResNet50 model in Keras with fused intermediate layers.  """

    def __init__(self, config, verbose=True, log_file=None):
        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__, 
                                         log_file=log_file, 
                                         silence=(not verbose), 
                                         global_log_file=verbose)

        # read config
        self._parse_config(config)


    def _parse_config(self, config):
        self._input_im_shape = config["input_im_shape"]
        self._weights = config["weights"]


    @staticmethod
    def load(input_im_shape, weights="random", trainable=True, setup_session=True, verbose=True, log_file=None):
        network = ResNet50Fused({"input_im_shape": input_im_shape, "weights": weights}, verbose=verbose, log_file=log_file)
        network.initialize_network(setup_session=setup_session)

        """       
        #TODO:(vsatish) Stricter check for valid arg
        if weights != "random":
            # load pre-trained weights
            network.load_trained_weights(weights)
        """       

        if not trainable:
            network.freeze_network() 

        return network


    def freeze_network(self):
        self._logger.info("Freezing network...")

        assert self._weights != "random", "Cannot freeze random weights."
        for layer in self._model.layers:
            layer.trainable = False #TODO:(vsatish) Confirm that the compile() before training actually takes this into account


    def initialize_network(self, setup_session=True):
        if setup_session:
            # do any TF session setup stuff here before we start building the network
            setup_tf_session()

        self._input_im = kl.Input(shape=self._input_im_shape, name="input_im")
        self._model = self._build_network(self._input_im, name="ResNet50Fused")


    def predict(self, images, bsz=32, verbose=True):
        return self._model.predict(images, batch_size=bsz, verbose=verbose)


    @property
    def model(self):
        return self._model


    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        self._logger.info("Building identity block: Stage {}, Block {}...".format(stage, block))

        filters1, filters2, filters3 = filters
        if kb.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = kl.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = kl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = kl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = kl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = kl.add([x, input_tensor])
        x = kl.Activation('relu')(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.

        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        self._logger.info("Building conv block: Stage {}, Block {}...".format(stage, block))

        filters1, filters2, filters3 = filters
        if kb.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = kl.Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = kl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = kl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = kl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = kl.Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = kl.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = kl.add([x, shortcut])
        x = kl.Activation('relu')(x)
        return x


    def _build_network(self, input_im, name="ResNet50Fused Network"):
        self._logger.info("Building {}...".format(name))

        if kb.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1

        x = kl.ZeroPadding2D((3, 3))(input_im)
        x = kl.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = kl.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x2 = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x2, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x1 = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x1, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x0 = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        #out1 = GlobalAveragePooling2D(name='out2')(x)
        #out1 = Flatten(name='out2')(out1)

        x = self.conv_block(x0, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = kl.AveragePooling2D((7, 7), name='avg_pool')(x)

        # last layer is flattened
        x_fl = kl.Flatten()(x)

        # Create basic model.
        model = km.Model(input_im, x_fl, name='resnet50')

        # pooling for intermediate layers
        flatten2 = kl.GlobalAveragePooling2D(name='flatten02')(x2)
        flatten1 = kl.GlobalAveragePooling2D(name='flatten01')(x1)
        flatten0 = kl.GlobalAveragePooling2D(name='flatten00')(x0)

        #TODO:(vsatish) Stricter check for valid arg
        if self._weights != "random":
            # load weights for basic model
            model.load_weights(self._weights)
            if kb.image_data_format() == "channels_first" and kb.backend() == "tensorflow":
                self._logger.warning("You are using the TensorFlow backend, yet you "
                                     "are using the Theano "
                                     "image data format convention "
                                     "(`image_data_format='channels_first'`). "
                                     "For best performance, set "
                                     "`image_data_format='channels_last'` in "
                                     "your Keras config "
                                     "at ~/.keras/keras.json.")

        # new output with concatenated intermediate layers
        conc = kl.Concatenate(name='ress')([flatten2, flatten1, flatten0, x_fl])

        # new model
        model_concatenated = km.Model(inputs=model.input, outputs=conc, name=name)
        return model_concatenated

