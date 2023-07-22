from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from tqdm import tqdm

import os
import glob
import argparse
import numpy as np
import skimage.io as mh
import skimage.transform as skt


def split_image(img):
    left = img[0:512,0:512]
    right = img[0:512,512:]
    return left, right

def resize_image(side, normalize):
    side_resized = skt.resize(side, (512, 512), preserve_range=True)
    side_resized = side_resized.astype(np.float32)
    if normalize:
        side_resized /= 255.0
    return side_resized

def resize_and_cast(side, target_dtype):
    side_resized = skt.resize(side, (512, 512), preserve_range=True)
    side_resized = side_resized.astype(target_dtype)
    return side_resized

def process_image(img, normalize):
    # Check if the image has more than one channel
    if len(img.shape) > 2 and img.shape[2] > 1:
        # If image has alpha channel, convert it to RGB
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        # Convert image to grayscale
        img = rgb2gray(img)
    left, right = split_image(img)
    images = []
    left_resized = resize_image(left, normalize)
    if np.prod(left_resized.shape) > 0:
        images.append(left_resized)
    if right.size != 0:  # Add this line
        right_resized = resize_image(right, normalize)
        if np.prod(right_resized.shape) > 0:
            images.append(right_resized)
    return np.array(images)

def process_mask(img):
    # Check if the image has more than one channel
    if len(img.shape) > 2 and img.shape[2] > 1:
        # If image has alpha channel, convert it to RGB
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        # Convert image to grayscale
        img = rgb2gray(img)
    left, right = split_image(img)
    masks = []
    left_resized = resize_and_cast(left, np.bool)
    if np.prod(left_resized.shape) > 0:
        masks.append(left_resized)
    if right.size != 0:  # Add this line
        right_resized = resize_and_cast(right, np.bool)
        if np.prod(right_resized.shape) > 0:
            masks.append(right_resized)
    return np.array(masks)


def load_files(datafolder, file_type, normalize):
    all_files = sorted(glob.glob(os.path.join(datafolder, '*.*')))
    image_files = [file for file in all_files if
                   'mask' not in file] if file_type == 'image' else []
    mask_files = [file for file in all_files if
                  'mask' in file] if file_type == 'mask' else []

    if file_type == 'image':
        images = []
        for a in tqdm(image_files):
            img = mh.imread(a)
            images.extend(process_image(img, normalize))
        return np.array(images)

    elif file_type == 'mask':
        masks = []
        for a in tqdm(mask_files):
            img = mh.imread(a)
            masks.extend(process_mask(img))
        return np.array(masks)


# other functions remain the same...

def process_images_and_masks(image_folder, mask_folder, normalize, output_dir,
                             image_filename, mask_filename):
    images = load_files(image_folder, 'image', normalize)

    if mask_folder is None:
        masks = load_files(image_folder, 'mask', normalize)
    else:
        masks = load_files(mask_folder, 'mask', normalize)

    if images.shape[0] != masks.shape[0]:
        raise ValueError('Number of images and masks do not match')

    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    print('Image dtype:', images.dtype)
    print('Mask dtype:', masks.dtype)
    print('Image shape:', images.shape)
    print('Mask shape:', masks.shape)

    np.save(os.path.join(output_dir, image_filename), images)
    np.save(os.path.join(output_dir, mask_filename), masks)


def main():
    parser = argparse.ArgumentParser(
        description="Load and process image and mask data.")
    parser.add_argument("-i", "--image_folder", type=str,
                        help="Root path to the images.")
    parser.add_argument("-m", "--mask_folder", type=str, default=None,
                        help="Root path to the masks. If not specified, it is assumed that masks are in the image folder.")
    parser.add_argument("--normalize", action='store_true',
                        help="Whether to normalize the images.")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Output directory for the processed arrays.")
    parser.add_argument("-if", "--image_filename", type=str,
                        help="Filename for the processed image array.")
    parser.add_argument("-mf", "--mask_filename", type=str,
                        help="Filename for the processed mask array.")
    args = parser.parse_args()

    process_images_and_masks(args.image_folder, args.mask_folder, args.normalize,
                             args.output_dir, args.image_filename,
                             args.mask_filename)


if __name__ == "__main__":
    main()
