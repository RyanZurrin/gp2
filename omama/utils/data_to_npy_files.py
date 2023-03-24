# class to convert data to npy files
from pathlib import Path

import numpy as np
from PIL import Image
import os
import tqdm
from tqdm import tqdm
from matplotlib import pyplot as plt
import tempfile


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
                 mask_extension: str = None,
                 image_shape: tuple = (512, 512, 1),
                 force=False):
        """
        Create a DataToNpyFiles object

        :param image_dir: Path to the directory containing the images
        :param mask_dir: Path to the directory containing the masks
        :param output_dir: Path to the directory to save the npy files
        :param images_file_name: Name of the npy file to save the images
        :param masks_file_name: Name of the npy file to save the masks
        :param image_extension: Extension of the images
        :param mask_extension: Extension of the masks
        :param image_shape: Shape of the images
        :param force: Force the conversion of the data to npy files even if
        the data has already been converted
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.image_extension = image_extension
        if mask_extension is None:
            self.mask_extension = image_extension
        else:
            self.mask_extension = mask_extension
        self.image_shape = image_shape
        self.image_paths = self._get_img_paths(image_dir)
        self.mask_paths = self._get_mask_paths(mask_dir)
        self.image_name = images_file_name
        self.mask_name = masks_file_name
        self.force = force
        # self._resize_images()
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

    def _get_img_paths(self, root_dir):
        image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(self.image_extension):
                    image_paths.append(os.path.join(root, file))
        # sort the paths so that they are in the same order as the masks
        image_paths.sort()
        return image_paths

    def _get_mask_paths(self, root_dir):
        mask_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(self.mask_extension):
                    mask_paths.append(os.path.join(root, file))
        # sort the paths so that they are in the same order as the images
        mask_paths.sort()
        return mask_paths

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
        # if the type is not numpy array use Image.open otherwise use np.load
        try:
            for image_path in tqdm(self.image_paths):
                image = Image.open(image_path)
                images.append(np.array(image))
            for mask_path in tqdm(self.mask_paths):
                mask = Image.open(mask_path)
                masks.append(np.array(mask))
        except:
            for image_path in tqdm(self.image_paths):
                image = np.load(image_path)
                images.append(image)
            for mask_path in tqdm(self.mask_paths):
                mask = np.load(mask_path)
                masks.append(mask)
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

    @staticmethod
    def show_image_and_mask(image, mask):
        """
        Display the image and mask together side by side
        Args:
            index: The index of the image and mask to display
        """
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()

    def load_data(self):
        """
        Load the saved .npy files
        """
        try:
            images = np.load(str(self.output_dir / (self.image_name + '.npy')))
            masks = np.load(str(self.output_dir / (self.mask_name + '.npy')))
        except ValueError:
            # add allow_pickle=True to fix error
            images = np.load(str(self.output_dir / (self.image_name + '.npy')),
                             allow_pickle=True)
            masks = np.load(str(self.output_dir / (self.mask_name + '.npy')),
                            allow_pickle=True)
        except FileNotFoundError:
            print('Data has not been converted to npy files yet')
            return

        return images, masks

    def remove_certain_images(self, substring_in_filename, image_type='masks'):
        """
        Remove images that have a certain substring in their filename
        Args:
            image_type: The type of image to remove. Can be 'images' or 'masks'
            substring_in_filename: The substring that the filename must contain
        """
        if image_type == 'images':
            image_paths = self.image_paths
        elif image_type == 'masks':
            image_paths = self.mask_paths
        else:
            raise ValueError('image_type must be "images" or "masks"')
        for image_path in image_paths:
            if substring_in_filename in image_path:
                os.remove(image_path)
        print(f'Removed images with {substring_in_filename} in their filename')

    def resave_npy_files(self):
        """
        Resave the npy files
        """
        images, masks = self.load_data()
        np.save(str(self.output_dir / (self.image_name + '.npy')), images)
        np.save(str(self.output_dir / (self.mask_name + '.npy')), masks)
        print(f'Images and masks saved to {self.output_dir}')

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

    def fix_dtype(self):
        """
        If images originally loaded where not all the same size the dypte will
        be Object. After reshaping all the images and masks the dtype of the
        npy files will still be Object and the shpae will be (n,). This function
        fixes that by loading the npy files and saving them again as npy files
        with a dypte of flaot64 and a shape of (n, x, y, z), where n is the
        number of images, x, y, are the dimensions of the images and z is 1 which
        is the number of channels
        """
        # save each of the images and masks from the npy file into temp directories
        images, masks = self.load_data()
        # use tempfiles to create temp directories
        temp_image_dir = tempfile.TemporaryDirectory()
        print(f'Temp image directory: {temp_image_dir.name}')
        temp_mask_dir = tempfile.TemporaryDirectory()
        print(f'Temp mask directory: {temp_mask_dir.name}')
        for i in range(len(images)):
            np.save(str(Path(temp_image_dir.name) / f'image_{i}.npy'),
                    images[i])
            np.save(str(Path(temp_mask_dir.name) / f'mask_{i}.npy'), masks[i])

        # print the ist of images and masks in the temp directories
        # print(f'Images in temp image directory: {os.listdir(temp_image_dir.name)}')
        # print(f'Masks in temp mask directory: {os.listdir(temp_mask_dir.name)}')
        # print out the shape of the images and masks in the temp directories
        print(
            f'Shape of images in temp image directory: {np.load(str(Path(temp_image_dir.name) / "image_0.npy")).shape}')
        print(
            f'Shape of masks in temp mask directory: {np.load(str(Path(temp_mask_dir.name) / "mask_0.npy")).shape}')

        #
        # # remove the original npy files
        # os.remove(str(self.output_dir / (self.image_name + '.npy')))
        # os.remove(str(self.output_dir / (self.mask_name + '.npy')))
        #
        # create a new instance of the class with the temp directories
        print(f'Output directory: {self.output_dir}')
        print(f'Image name: {self.image_name}')
        print(f'Mask name: {self.mask_name}')
        print(f'Image extension: {self.image_extension}')
        print(f'Mask extension: {self.mask_extension}')
        print(f'temp image directory: {temp_image_dir.name}')
        print(f'temp mask directory: {temp_mask_dir.name}')
        temp_data = DataToNpyFiles(temp_image_dir.name,
                                   temp_mask_dir.name,
                                   self.output_dir,
                                   self.image_name,
                                   self.mask_name,
                                   'npy',
                                   force=True
                                   )
        # # reshape the images and masks in the temp directories
        # temp_data.reshape_all_images_and_masks()
        # clear the temp directories and delete the temp directories
        temp_image_dir.cleanup()
        temp_mask_dir.cleanup()

    def _resize_images(self):
        """
        Reshape all images and masks in the masks and images directory, use
        similar code to the reshape_all_images_and_masks function just instead
        of loading npy files it loads the images and masks from the folders
        """
        from skimage.transform import resize

        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(self.image_extension):
                    image = Image.open(os.path.join(root, file))
                    image = np.array(image)
                    image = resize(image, self.image_shape, mode='constant',
                                   preserve_range=True)
                    image = Image.fromarray(image)
                    image.save(os.path.join(self.image_dir, file))
        for root, dirs, files in os.walk(self.mask_dir):
            for file in files:
                if file.endswith(self.mask_extension):
                    mask = Image.open(os.path.join(root, file))
                    mask = np.array(mask)
                    mask = resize(mask, self.image_shape, mode='constant',
                                  preserve_range=True)
                    mask = Image.fromarray(mask)
                    mask.save(os.path.join(self.mask_dir, file))
        print('Images and masks resized')

    @staticmethod
    def seperate_masks_and_images(root_dir,
                                  output_img,
                                  output_mask,
                                  image_extension,
                                  mask_keyword,
                                  mask_extension=None):
        """
        Seperate the masks and images into seperate folders
        """
        import os
        if mask_extension is None:
            mask_extension = image_extension
        if not os.path.exists(output_img):
            os.mkdir(output_img)
        if not os.path.exists(output_mask):
            os.mkdir(output_mask)

        if mask_extension != image_extension:
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(image_extension):
                        if mask_keyword in file:
                            os.rename(os.path.join(root, file),
                                      os.path.join(output_mask, file))
                        else:
                            os.rename(os.path.join(root, file),
                                      os.path.join(output_img, file))
                    elif file.endswith(mask_extension):
                        os.rename(os.path.join(root, file),
                                  os.path.join(output_mask, file))
        else:
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(image_extension):
                        if mask_keyword in file:
                            os.rename(os.path.join(root, file),
                                      os.path.join(output_mask, file))
                        else:
                            os.rename(os.path.join(root, file),
                                      os.path.join(output_img, file))
