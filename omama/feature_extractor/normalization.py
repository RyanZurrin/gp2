import time
import types

import pydicom as dicom
import numpy as np
import mahotas as mh

DEBUG = False


# class to represent different normalization techniques

class Normalize:
    @staticmethod
    def extract_pixels(images, timing=False):
        """Extract pixels from images
        Parameters
        ----------
        images : numpy.ndarray | list of numpy.ndarray
            Array of images to be normalized
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.
        """
        t0 = time.time()
        pixels = []
        if isinstance(images, list):
            if isinstance(images[0], np.ndarray):
                return images
            elif isinstance(images[0], types.SimpleNamespace):
                for image in images:
                    pixels.append(image.pixels)
            elif isinstance(images[0], dicom.dataset.FileDataset):
                for image in images:
                    pixels.append(image.pixel_array)
            else:
                raise TypeError("Unknown type of images")
        elif isinstance(images, np.ndarray):
            pixels = images  # was returning this as list before
        elif isinstance(images, types.SimpleNamespace):
            pixels = images.pixels
        else:
            raise TypeError("Unknown type of images")
        if timing:
            print("Extract pixels: {}".format(time.time() - t0))
        return pixels

    @staticmethod
    def _minmax_helper(pixels):
        """Helper function to normalize data using minmax method
        """
        max_val = np.max(pixels)
        min_val = np.min(pixels)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= min_val
        normalized_pixels /= (max_val - min_val)
        normalized_pixels *= 255

        return normalized_pixels

    @staticmethod
    def minmax(pixels, downsample=False, output_shape=None, timing=False):
        """The min-max approach (often called normalization) rescales the
        feature to a fixed range of [0,1] by subtracting the minimum value
        of the feature and then dividing by the range, which is then multiplied
        by 255 to bring the value into the range [0,255].
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        if isinstance(pixels, list):
            if DEBUG:
                print("minmax: list")
            normalized_pixels = []
            if downsample:
                normalized_pixels = Normalize.downsample(images=pixels,
                                                         output_shape=output_shape,
                                                         normalize=True)
            else:
                for p in pixels:
                    normalized_pixels.append(Normalize._minmax_helper(p))
        else:
            if DEBUG:
                print("minmax: not list")
            if downsample:
                normalized_pixels = Normalize.downsample(images=pixels,
                                                         output_shape=output_shape,
                                                         normalize=True)
            else:
                normalized_pixels = Normalize._minmax_helper(pixels)

        if timing:
            print("minmax: ", time.time() - t0)
        return normalized_pixels

    @staticmethod
    def _max_helper(pixels):
        """Helper function to normalize data using max method
        """
        max_val = np.max(pixels)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels /= abs(max_val)
        normalized_pixels *= 255

        return normalized_pixels

    @staticmethod
    def max(pixels, downsample=False, output_shape=None, timing=False):
        """The maximum absolute scaling rescales each feature between -1 and 1
        by dividing every observation by its maximum absolute value.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = []
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            for p in temp_pixels:
                normalized_pixels.append(Normalize._max_helper(p))
        else:
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            normalized_pixels = Normalize._max_helper(temp_pixels)

        if timing:
            print("max: ", time.time() - t0)
        return normalized_pixels

    @staticmethod
    def _gaussian_helper(pixels):
        """Helper function to normalize data using gaussian blur
        """
        normalized_pixels = mh.gaussian_filter(pixels, sigma=1)
        normalized_pixels /= normalized_pixels.max()
        return normalized_pixels

    @staticmethod
    def gaussian(pixels, downsample=False, output_shape=None, timing=False):
        """Normalize by gaussian
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = []
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            for p in temp_pixels:
                normalized_pixels.append(Normalize._gaussian_helper(p))
        else:
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            normalized_pixels = Normalize._gaussian_helper(temp_pixels)

        if timing:
            print("gaussian: ", time.time() - t0)
        return normalized_pixels

    @staticmethod
    def _zscore_helper(pixels):
        """Helper function to normalize data using zscore method
        """
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= np.mean(normalized_pixels)
        normalized_pixels /= np.std(normalized_pixels)
        normalized_pixels *= 255

        return normalized_pixels

    @staticmethod
    def z_score(pixels, downsample=False, output_shape=None, timing=False):
        """The z-score method (often called standardization) transforms the data
        into a distribution with a mean of 0 and a standard deviation of 1.
        Each standardized value is computed by subtracting the mean of the
        corresponding feature and then dividing by the standard deviation.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = []
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            for p in temp_pixels:
                normalized_pixels.append(Normalize._zscore_helper(p))
        else:
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            normalized_pixels = Normalize._zscore_helper(temp_pixels)

        if timing:
            print("z_score: ", time.time() - t0)
        return normalized_pixels

    @staticmethod
    def _robust_helper(pixels):
        """Helper function to normalize data using robust method
        """
        median = np.median(pixels)
        iqr = np.percentile(pixels, 75) - np.percentile(pixels, 25)
        normalized_pixels = (pixels - median) / iqr

        return normalized_pixels

    @staticmethod
    def robust(pixels, downsample=False, output_shape=None, timing=False):
        """In robust scaling, we scale each feature of the data set by subtracting
        the median and then dividing by the interquartile range. The interquartile
        range (IQR) is defined as the difference between the third and the first
        quartile and represents the central 50% of the data.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = []
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            for p in temp_pixels:
                normalized_pixels.append(Normalize._robust_helper(p))
        else:
            if downsample:
                temp_pixels = Normalize.downsample(pixels, output_shape)
            normalized_pixels = Normalize._robust_helper(temp_pixels)

        if timing:
            print("robust: ", time.time() - t0)
        return normalized_pixels

    @staticmethod
    def downsample(images, output_shape=None, flatten=False,
                   normalize=None, timing=False):
        """Downsample images to a given shape.
        Parameters
        ----------
        images : numpy.ndarray, list of numpy.ndarray
            Array of images to be downsampled
        output_shape : tuple
            Shape of the output images
        flatten : bool
            If true, the images are flattened to a 1D array
        normalize : str | bool
            If not None, the images are normalized using the given method
        timing : bool
            If true, the time needed to perform the downsampling is printed
        Returns
        -------
        numpy.ndarray | list of numpy.ndarray
            Downsampled images
        """
        t0 = time.time()
        if DEBUG:
            print("Downsampling images to shape: {}".format(output_shape))
        from skimage.transform import resize
        if output_shape is None:
            output_shape = (128, 128)
        images_copy = Normalize.extract_pixels(images)
        if isinstance(images_copy, list):
            resized = []
            if DEBUG:
                print("Downsampling {} images".format(len(images_copy)))
            for img in images_copy:
                resized.append(resize(img, output_shape))
                if timing:
                    print("downsample: ", time.time() - t0)
        else:
            if DEBUG:
                print("Downsampling 1 image")
            resized = resize(images_copy, output_shape)

        if normalize is not None:
            if DEBUG:
                print("Normalizing images in downsampling")
            if isinstance(normalize, bool):
                normalize = "minmax"
            if isinstance(resized, list):
                for i in range(len(resized)):
                    resized[i] = Normalize.get_norm(resized[i],
                                                    normalize)
            else:
                resized = Normalize.get_norm(resized, normalize)

        if flatten:
            if DEBUG:
                print("Flattening images in downsampling")
            resized = [img.flatten() for img in resized]

        if timing:
            print("downsample: ", time.time() - t0)
        return resized

    @staticmethod
    def get_norm(pixels,
                 norm_type='min-max',
                 downsample=False,
                 output_shape=None,
                 timing=False):
        """Normalize pixels
        Parameters
        ----------
        pixels : numpy.ndarray
            Array of pixels to be normalized
        norm_type : str, optional
            Type of normalization. The default is 'min-max'.
            options -> attributes possible:
                'min-max',
                'max',
                'gaussian',
                'z-score',
                'robust',
                'downsample' -> output_shape
        downsample : bool, optional
            Whether to downsample the images. The default is False.
        output_shape : tuple, optional
            Shape of the output images. The default is None.
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        if norm_type.lower() == 'max':
            normalized_pixels = Normalize.max(pixels, downsample, output_shape)
        elif norm_type.lower() == 'minmax' or norm_type.lower() == 'min-max':
            normalized_pixels = Normalize.minmax(pixels, downsample,
                                                 output_shape)
        elif norm_type.lower() == 'gaussian':
            normalized_pixels = Normalize.gaussian(pixels, downsample,
                                                   output_shape)
        elif norm_type.lower() == 'zscore' or norm_type.lower() == 'z-score':
            normalized_pixels = Normalize.z_score(pixels, downsample,
                                                  output_shape)
        elif norm_type.lower() == 'robust':
            normalized_pixels = Normalize.robust(pixels, downsample,
                                                 output_shape)
        elif norm_type.lower() == 'downsample':
            normalized_pixels = Normalize.downsample(pixels, output_shape)
        else:
            raise ValueError('Invalid normalization type')

        if timing:
            print("get_norm: ", time.time() - t0)

        return normalized_pixels
