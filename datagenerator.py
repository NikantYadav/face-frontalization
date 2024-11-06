import numpy as np
import os
from glob import glob
import keras
import PIL.Image as pilimg
from keras.utils import Sequence

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset_dir, batch_size=32, dim=(128, 128), n_channels=3, shuffle=True):
        """
        Args:
            dataset_dir: Directory containing subject folders (e.g., 'Dataset/').
            batch_size: Number of samples per batch.
            dim: Dimensions to which images will be resized (height, width).
            n_channels: Number of channels in the images (3 for RGB).
            shuffle: Whether to shuffle the data after every epoch.
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle

        # Collect subject folders (e.g., 001, 002, etc.)
        self.subject_folders = sorted(glob(os.path.join(self.dataset_dir, '*/')))
        
        # Initialize lists for side and frontal images
        self.sideslist = []
        self.frontslist = []

        # For each subject folder, get the side images and the target frontal image
        for folder in self.subject_folders:
            all_images = sorted(glob(os.path.join(folder, '*.png'))) # Exclude target image
            
            print(len(all_images))
            if len(all_images) < 77:
                print(f"Warning: Folder '{folder}' contains fewer than 77 images. Skipping this folder.")
                continue  # Skip folders with fewer than 77 images
            side_images = all_images[:76] + all_images[77:]
            frontal_image = all_images[76]  # Assume the target image has "target" in its name
            
            self.sideslist.extend(side_images)
            self.frontslist.append(frontal_image)  # Each subject has one frontal image

        # Shuffle indices after each epoch
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.sideslist) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Get corresponding side and frontal images
        sideslist_temp = [self.sideslist[k] for k in indexes]
        frontslist_temp = [self.frontslist[index % len(self.frontslist)]] * len(sideslist_temp) # All side images in a batch will share the same target frontal image
        

        # Generate data for the batch
        sides, fronts = self.__data_generation(sideslist_temp, frontslist_temp)

        return sides, fronts

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        self.indexes = np.arange(len(self.sideslist))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sideslist, frontslist):
        """
        Generates data for a batch: loads and processes the images.
        """
        # Initialize arrays for batch
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, (sidename, frontname) in enumerate(zip(sideslist, frontslist)):
            # Load and process side image
            side = pilimg.open(sidename)
            side = side.resize(self.dim)
            side = np.array(side)
            X[i] = side

            # Load and process frontal image (target)
            front = pilimg.open(frontname)
            front = front.resize(self.dim)
            front = np.array(front)
            Y[i] = front

        # Apply preprocessing (normalization)
        return self.preprocessing(X), self.preprocessing(Y)

    def preprocessing(self, img):
        """
        Preprocess images (normalize to [-1, 1] range)
        """
        return (img / 255.0) * 2 - 1  # Normalize image to [-1, 1]



class DataGenerator_predict(keras.utils.Sequence):
    def __init__(self, sideslist, batch_size=32, dim=(128, 128), n_channels=3, shuffle=True):
        """
        Args:
            sideslist: List of side image file paths.
            batch_size: Number of samples per batch.
            dim: Dimensions to which images will be resized (height, width).
            n_channels: Number of channels in the images (3 for RGB).
            shuffle: Whether to shuffle the data after every epoch.
        """
        self.sideslist = sideslist
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle

        # Initialize indices and shuffle if required
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.sideslist) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        sideslist_temp = [self.sideslist[k] for k in indexes]
        sides = self.__data_generation(sideslist_temp)
        return sides

    def on_epoch_end(self):
        # Shuffle indices after each epoch
        self.indexes = np.arange(len(self.sideslist))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sideslist):
        """
        Generates data for a batch: loads and processes the side images.
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, sidename in enumerate(sideslist):
            # Load and process side image
            side = pilimg.open(sidename)
            side = side.resize(self.dim)
            side = np.array(side)
            X[i] = side

        # Apply preprocessing (normalization)
        return self.preprocessing(X)

    def preprocessing(self, img):
        """
        Preprocess images (normalize to [-1, 1] range)
        """
        return (img / 255.0) * 2 - 1  # Normalize image to [-1, 1]