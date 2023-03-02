# class to convert data to npy files

import numpy as np
from PIL import Image
import glob


class DataToNpyFiles:
    def __init__(self, imgs, masks, output_dir):
        self.imgs = imgs
        self.masks = masks
        self.output_dir = output_dir
        self.imgs_npy = self.imgs + '.npy'
        self.masks_npy = self.masks + '.npy'
        self.convert()

    def convert(self):
        self.imgae_collection_to_npy(self.imgs, 'png', 'npy')
        self.mask_collection_to_npy(self.masks, 'png', 'npy')

    @staticmethod
    def read(file_name):
        img = np.array(Image.open(file_name))
        return img

    @staticmethod
    def get_files_in_collection(collection_path, infile_type):
        files = glob.glob(collection_path + '/*.' + infile_type)
        return files

    def imgae_collection_to_npy(self, collection_path, infile_type,
                                outfile_type):
        files = self.get_files_in_collection(collection_path, infile_type)
        imgages = []
        for file in files:
            img = self.read(file)
            imgages.append(img)
        images = np.array(imgages)
        np.save(collection_path + '.' + outfile_type, images)

    def mask_collection_to_npy(self, collection_path, infile_type,
                               outfile_type):
        files = self.get_files_in_collection(collection_path, infile_type)
        masks = []
        for file in files:
            mask = self.read(file)
            masks.append(mask)
        masks = np.array(masks)
        np.save(collection_path + '.' + outfile_type, masks)
