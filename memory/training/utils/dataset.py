"""
Dataset and generators for training a Siamese Network in Keras. Based on the 
Matterport dataset.
Authors: Vishal Satish, Kate Sanders, 
         David Wang, Michael Danielczuk, 
         Matthew Matl
"""
import os
import random

import skimage
import numpy as np
import keras.utils as ku


class ImageDataset(object):
    """
    ImageDataset creates a Matterport dataset for a directory of
    images in order to ensure compatibility with benchmarking tools
    and image resizing for networks.

    Directory structure must be as follows:

    $base_path/
        test_indices.npy
        train_indices.npy
        depth_ims/ (Depth images here)
            image_000000.png
            image_000001.png
            ...
        color_ims/ (Color images here)
            image_000000.png
            image_000001.png
            ...
        modal_segmasks/ (GT segmasks here, one channel)
            image_000000.png
            image_000001.png
            ...
    """

    def __init__(self, base_path, imset, data_augmentation_suffixes=[''], allow_different_views=False):
        assert base_path != "", "You must provide the path to a dataset!"

        self.data_path = os.path.join(base_path, imset)
        self._orig_im_data_path = os.path.join(base_path, "originals")
        self._class_labels = os.listdir(self.data_path)
        self._num_classes = len(self._class_labels)
        self._num_images = len(os.listdir(os.path.join(self.data_path, self._class_labels[0])))
        self.triple_info = []
        self.data_augmentation_suffixes = data_augmentation_suffixes
        self.allow_different_views = allow_different_views

    def generate_triple_from_ind(self, index):
        class_ind = index % self._num_images**2
        rem_ind = index - class_ind
        image1_ind = int((index - class_ind)/self._num_images)

    # Full image of class1, partial images of class 2
    def generate_triple_specific(self, pos, class1, class2):

        folder1 = os.path.join(self.data_path + random.choice(self.data_augmentation_suffixes), class1)
        # Get folder for original images
        orig_folder = os.path.join(self._orig_im_data_path + random.choice(self.data_augmentation_suffixes), class1)

        # Choose random view and get corresponding image:
        views = os.listdir(orig_folder)
        view = random.choice(views)
        view_str = view.split('.')[0]
        im1 = os.path.join(orig_folder, view)

        if pos:
            # Pick random slice of original image in that view
            im2_views = os.path.join(folder1, view_str)
            im2 = os.path.join(im2_views, random.choice(os.listdir(im2_views))  )

            # print ("im2: " + str(im2) )
            label = 1
        else:
            folder2 = os.path.join(self.data_path + random.choice(self.data_augmentation_suffixes), class2)

            # Choose random view
            new_views = os.listdir(folder2)
            new_view = random.choice(new_views)

            im2_views = os.path.join(folder2, new_view)
            im2 = os.path.join(im2_views, random.choice(os.listdir(im2_views)))

            # print ("im2: " + str(im2) )
            label = 0

        return (im1, im2, label, class1, class2)


    def generate_triple(self, pos=False):
        class1 = random.choice(self._class_labels)

        folder1 = os.path.join(self.data_path + random.choice(self.data_augmentation_suffixes), class1)
        # Get folder for original images
        orig_folder = os.path.join(self._orig_im_data_path + random.choice(self.data_augmentation_suffixes), class1)

        # Choose random view and get corresponding image:
        views = os.listdir(orig_folder)
        view = random.choice(views)
        view_str = view.split('.')[0]
        #view_str = 'view_000000'
        im1 = os.path.join(orig_folder, view)

        if pos:
            # Pick random slice of original image.
            if self.allow_different_views:
                view = random.choice(views)
                view_str = view.split('.')[0]
            im2_views = os.path.join(folder1, view_str)
            #im2_views = os.path.join(folder1, "{}.npz".format(view_str))
            #im2_views = os.path.join(folder1, "view_000000")
            im2 = os.path.join(im2_views, random.choice(os.listdir(im2_views))  )
            #im2 = os.path.join(im2_views, "img.npz")
            # print ("im2: " + str(im2) )
            #im2 = im2_views
            label = 1
            class2 = class1
        else:
            # Make sure you get a different class
            class2 = random.choice(self._class_labels)
            while class2 == class1:
                class2 = random.choice(self._class_labels)

            folder2 = os.path.join(self.data_path, class2)

            # Choose random view
            new_views = os.listdir(folder2)
            new_view = random.choice(new_views)
            #new_view = 'view_000000'
            im2_views = os.path.join(folder2, new_view)
            #im2 = im2_views
            #im2 = os.path.join(im2_views, "img.npz")
            im2 = os.path.join(im2_views, random.choice(os.listdir(im2_views))  )
            # print ("im2: " + str(im2) )

            label = 0

        return (im1, im2, label, class1, class2)

    def add_triple(self, path1, path2, label, class1, class2):
        self.triple_info.append({
            "im1": path1,
            "im2": path2,
            "label": label,
            "class1": class1,
            "class2": class2
        })

    def prepare_specific(self, size, class1, class2):
        for i in range(size):
            self.add_triple(*self.generate_triple_specific(True, class1, class2))
            self.add_triple(*self.generate_triple_specific(False, class1, class2))

    def prepare(self, size):
        for i in range(size):
            self.add_triple(*self.generate_triple(True))
            self.add_triple(*self.generate_triple(False))

    def load_im(self, image_id, key):
        # loads image from path
        image = skimage.io.imread(self.triple_info[image_id][key])

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_feat(self, image_id, key):
        return np.load(self.triple_info[image_id][key])['arr_0']

    def load_label(self, image_id):
       return self.triple_info[image_id]['label']

    def load_classes(self, image_id):
        return (self.triple_info[image_id]['class1'], self.triple_info[image_id]['class2'])

    @property
    def triples(self):
        return self.triple_info


class DataGenerator(ku.Sequence):
    def __init__(self, im_dataset, batch_size=32, dim=(32,32,32), shuffle=True, dataset_type="image", output=False):
        """Initialization"""
        self.im_dataset = im_dataset
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dataset_type = dataset_type
        self.output = output

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y, z, c1, c2 = self.__data_generation(indices)
        if self.output:
            f = open("ground.txt", "a")
            for index in range(len(z)):
                f.write('{} {} {} '.format(z[index], c1[index], c2[index]))
            f.close()
        return [X, Y], z

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.im_dataset.triples))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.im_dataset.triples) / self.batch_size))

    def __data_generation(self, indices):
        """Generates data containing batch_size samples,  X : (n_samples, *dim, n_channels) """
        # Initialization
        shape = [self.batch_size]
        for i in self.dim:
            shape.append(i)
        X = np.empty(shape, dtype=np.uint8)
        Y = np.empty(shape, dtype=np.uint8)
        z = np.empty((self.batch_size), dtype=np.uint8)
        c1 = []
        c2 = []

        # Generate data
        for i, ind in enumerate(indices):
            # Store sample
            if self.dataset_type == "image":
                X[i,] = self.im_dataset.load_im(ind, "im1")
                Y[i,] = self.im_dataset.load_im(ind, "im2")
            elif self.dataset_type == "feature":
                X[i,] = self.im_dataset.load_feat(ind, "im1")
                Y[i,] = self.im_dataset.load_feat(ind, "im2")
            else:
                raise ValueError("Dataset type {0} not supported.".format(self.dataset_type))


            # Store class
            z[i] = self.im_dataset.load_label(ind)

            ca1, ca2 = self.im_dataset.load_classes(ind)
            c1 += [ca1]
            c2 += [ca2]
        return X, Y, z, c1, c2

