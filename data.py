import os
from os.path import join
import numpy as np
from random import shuffle
from PIL import Image

def is_jpeg(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg"])

def get_subdirs(directory):
    subdirs = sorted([join(directory, name) for name in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, name))])
    return subdirs

flatten = lambda l: [item for sublist in l for item in sublist]

class ExternalInputIterator(object):
    
    def __init__(self, imageset_dir, batch_size, random_shuffle=False):
        self.images_dir = imageset_dir
        self.batch_size = batch_size

        # First, figure out what are the inputs and what are the targets in your directory structure:
        # Get a list of filenames for the target (frontal) images
        self.frontals = np.array([join(imageset_dir, frontal_file) for frontal_file in sorted(os.listdir(imageset_dir)) if is_jpeg(frontal_file)])
        
        # Get a list of lists of filenames for the input (profile) images for each person
        profile_files = [[join(person_dir, profile_file) for profile_file in sorted(os.listdir(person_dir)) if is_jpeg(profile_file)] for person_dir in get_subdirs(imageset_dir)]
        
        # Build a flat list of frontal indices, corresponding to the *flattened* profile_files
        frontal_ind = []
        for ind, profiles in enumerate(profile_files):
            frontal_ind += [ind] * len(profiles)
        self.frontal_indices = np.array(frontal_ind)
        
        # Now that we have built frontal_indices, we can flatten profile_files
        self.profiles = np.array(flatten(profile_files))

        # Shuffle the (input, target) pairs if necessary
        if random_shuffle:
            ind = np.array(range(len(self.frontal_indices)))
            shuffle(ind)
            self.profiles = self.profiles[ind]
            self.frontal_indices = self.frontal_indices[ind]

                
    def __iter__(self):
        self.i = 0
        self.n = len(self.frontal_indices)
        return self

    def __next__(self):
        profiles = []
        frontals = []
        
        for _ in range(self.batch_size):
            profile_filename = self.profiles[self.i]
            frontal_filename = self.frontals[self.frontal_indices[self.i]]

            # Open and read the images using PIL
            profile_image = Image.open(profile_filename)
            frontal_image = Image.open(frontal_filename)

            profiles.append(np.array(profile_image))
            frontals.append(np.array(frontal_image))

            self.i = (self.i + 1) % self.n
        
        return (profiles, frontals)

    next = __next__


class ImagePipeline:
    '''
    Constructor arguments:  
    - imageset_dir: directory containing the dataset
    - image_size = 128: length of the square that the images will be resized to
    - random_shuffle = False
    - batch_size = 64
    - num_threads = 2
    '''
    
    def __init__(self, imageset_dir, image_size=128, random_shuffle=False, batch_size=64):
        self.iterator = ExternalInputIterator(imageset_dir, batch_size, random_shuffle)
        self.num_inputs = len(self.iterator.frontal_indices)
        self.image_size = image_size
    
    def epoch_size(self):
        return self.num_inputs
    
    def resize_and_normalize(self, image):
        # Resize the image
        image = image.resize((self.image_size, self.image_size))
        
        # Normalize the image: subtract 128 and divide by 128 (common image preprocessing)
        image = np.array(image).astype(np.float32)
        image = (image - 128) / 128.0
        return image

    def define_graph(self):
        # This function is now redundant, as we don't need to build a DALI graph.
        pass
    
    def iter_setup(self):
        # Get the next batch of profile and frontal images
        images, targets = self.iterator.next()
        
        # Preprocess the images: resize and normalize
        processed_profiles = [self.resize_and_normalize(img) for img in images]
        processed_frontals = [self.resize_and_normalize(img) for img in targets]
        
        return np.array(processed_profiles), np.array(processed_frontals)


