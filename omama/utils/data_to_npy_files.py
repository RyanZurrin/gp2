# class to convert data to npy files
from pathlib import Path

import numpy as np
from PIL import Image
import os
import tqdm
from tqdm import tqdm
from matplotlib import pyplot as plt


class DataToNpyFiles:
    """
    Class to convert segmentation datasets containing images and masks to
    single images and masks in npy files for use in deep learning models.
    """

    def __init__(self,
                 image_dir: Path,
                 mask_dir: Path,
                 output_dir: Path,
                 images_file_name: str,
                 masks_file_name: str,
                 image_extension: str,
                 force=False):
        """
        Args:
            image_dir: Path to the directory containing the images
            mask_dir: Path to the directory containing the masks
            output_dir: Path to the directory where the npy files will be
            image_extension: The extension of the images
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.image_extension = image_extension
        self.image_paths = self._get_paths(image_dir)
        self.mask_paths = self._get_paths(mask_dir)
        self.image_name = images_file_name
        self.mask_name = masks_file_name
        self.force = force
        self._convert_data()

    def __str__(self):
        return 'DataToNpyFiles(image_dir={}, ' \
               'mask_dir={}, output_dir={}, ' \
               'image_extension={})'.format(self.image_dir,
                                            self.mask_dir,
                                            self.output_dir,
                                            self.image_extension)

    def _already_converted(self):
        """
        Check if the data has already been converted to npy files
        """
        if self.image_name + '.npy' in os.listdir(self.output_dir) and \
                self.mask_name + '.npy' in os.listdir(self.output_dir):
            return True
        else:
            return False

    def _get_paths(self, root_dir):
        image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(self.image_extension):
                    image_paths.append(os.path.join(root, file))
        # sort the paths so that they are in the same order as the masks
        image_paths.sort()
        return image_paths

    def _convert_data(self):
        """
        Convert the data to npy files
        """
        # check if the data has already been converted, unless force is True
        if self._already_converted() and not self.force:
            print('Data has already been converted to npy files')
            return
        images = []
        masks = []
        for image_path in self.image_paths:
            image = Image.open(image_path)
            images.append(np.array(image))
        for mask_path in self.mask_paths:
            mask = Image.open(mask_path)
            masks.append(np.array(mask))
        images = np.array(images)
        masks = np.array(masks)
        # sort the images and masks by the image name so that they are in
        # the same order
        images = images[np.argsort(self.image_paths)]
        masks = masks[np.argsort(self.mask_paths)]
        np.save(str(self.output_dir / (self.image_name + '.npy')), images)
        np.save(str(self.output_dir / (self.mask_name + '.npy')), masks)
        print(f'Images and masks saved to {self.output_dir}')

    def display_image_and_mask(self, index):
        """
        Display the image and mask at the given index
        Args:
            index: The index of the image and mask to display
        """
        image, mask = self.load_data()
        plt.subplot(1, 2, 1)
        plt.imshow(image[index])
        plt.subplot(1, 2, 2)
        plt.imshow(mask[index])
        plt.show()

    def load_data(self):
        """
        Load the saved .npy files
        """
        images = np.load(str(self.output_dir / (self.image_name + '.npy')))
        masks = np.load(str(self.output_dir / (self.mask_name + '.npy')))
        return images, masks

    def get_data_with_masks(self):
        """
        Iterate through the images dircetory and return a list of paths
        to the images that also have a mask file
        """
        data_with_masks = []
        for image_path in self.image_paths:
            if Path(image_path).stem in self.mask_paths:
                data_with_masks.append(image_path)
        return data_with_masks

    def seperate_masks_and_images(self, root_dir, output_img, output_mask,
                                  image_extension, mask_keyword):
        """
        Seperate the masks and images into seperate folders
        """
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(image_extension):
                    if mask_keyword in file:
                        os.rename(os.path.join(root, file),
                                  os.path.join(output_mask, file))
                    else:
                        os.rename(os.path.join(root, file),
                                  os.path.join(output_img, file))

    def print_images_in_dir(self, root_dir, image_extension):
        """
        Print the images in the images directory using Image and plt
        """
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(image_extension):
                    image = Image.open(os.path.join(root, file))
                    plt.imshow(image)
                    plt.show()

    def reshape_all_images_and_masks(self, new_shape=(512, 512, 1)):
        """
        Reshape all images and masks in the given directory
        """
        from skimage.transform import resize

        try:
            images, masks = self.load_data()
            for i in tqdm(range(len(images))):
                images[i] = resize(images[i], new_shape, mode='constant',
                                   preserve_range=True)
                masks[i] = resize(masks[i], new_shape, mode='constant',
                                  preserve_range=True)
            np.save(str(self.output_dir / (self.image_name + '.npy')), images)
            np.save(str(self.output_dir / (self.mask_name + '.npy')), masks)
        except:
            print('Could not use tqdm to show progress')
            images, masks = self.load_data()
            images = np.array([resize(image, new_shape, mode='constant',
                                      preserve_range=True)
                               for image in images])
            masks = np.array([resize(mask, new_shape, mode='constant',
                                     preserve_range=True)
                              for mask in masks])
            np.save(str(self.output_dir / (self.image_name + '.npy')), images)
            np.save(str(self.output_dir / (self.mask_name + '.npy')), masks)

        # use tqdm to show progress
