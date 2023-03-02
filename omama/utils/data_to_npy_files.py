# class to convert data to npy files
from pathlib import Path

import numpy as np
from PIL import Image
import glob

from matplotlib import pyplot as plt


class DataToNpyFiles:
    def __init__(self,
                 images_path: str,
                 masks_path: str,
                 output_dir: str,
                 image_type: str):
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.output_dir = Path(output_dir)
        self.image_type = image_type
        self.img_npy_path = self.output_dir / 'images.npy'
        self.masks_npy_path = self.output_dir / 'masks.npy'
        self.convert()

    def convert(self):
        print('Converting data to npy files...')
        if self.images_path.is_file():
            print("Single image file found: {}".format(self.images_path))
            self.image_to_npy(self.images_path, self.img_npy_path)
        else:
            print("Image directory found: {}".format(self.images_path))
            self.image_collection_to_npy(self.images_path, self.image_type)
        if self.masks_path.is_file():
            print("Single mask file found: {}".format(self.masks_path))
            self.mask_to_npy(self.masks_path, self.masks_npy_path)
        else:
            print("Mask directory found: {}".format(self.masks_path))
            self.mask_collection_to_npy(self.masks_path, self.image_type)

    def image_to_npy(self, image_path: Path, npy_path: Path):
        print('Converting image to npy file...')
        image = Image.open(image_path)
        image = np.array(image)
        np.save(str(npy_path), image)

    def image_collection_to_npy(self, image_dir: Path, image_type: str):
        print('Converting image collection to npy file...')
        image_paths = glob.glob(str(image_dir) / ('*.' + image_type))
        print(f'path is {str(image_dir) / ("*." + image_type)}')
        print('Image paths: {}'.format(image_paths))
        images = []
        for image_path in image_paths:
            print('Converting image: {}'.format(image_path))
            image = Image.open(image_path)
            image = np.array(image)
            print('Image shape: {}'.format(image.shape))
            images.append(image)
        np.save(str(self.img_npy_path), images)

    def mask_to_npy(self, mask_path: Path, npy_path: Path):
        print('Converting mask to npy file...')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        np.save(str(npy_path), mask)

    def mask_collection_to_npy(self, mask_dir: Path, image_type: str):
        print('Converting mask collection to npy file...')
        mask_paths = glob.glob(str(mask_dir / ('*.' + image_type)))
        print('Mask paths: {}'.format(mask_paths))
        masks = []
        for mask_path in mask_paths:
            print('Converting mask: {}'.format(mask_path))
            mask = Image.open(mask_path)
            mask = np.array(mask)
            print('Mask shape: {}'.format(mask.shape))
            masks.append(mask)
        np.save(str(self.masks_npy_path), masks)

    def load_images(self):
        print('Loading images from npy file...')
        images = np.load(str(self.img_npy_path))
        return images

    def load_masks(self):
        print('Loading masks from npy file...')
        masks = np.load(str(self.masks_npy_path))
        return masks

    def load_data(self):
        print('Loading data from npy files...')
        images = self.load_images()
        masks = self.load_masks()
        return images, masks

    def view_image_and_mask(self, image_index):
        print('Viewing image and mask...')
        images, masks = self.load_data()
        image = images[image_index]
        mask = masks[image_index]
        plt.imshow(image)
        plt.show()
        plt.imshow(mask)
        plt.show()

