# class for extracting features from data
import time
from types import SimpleNamespace
from .normalization import Normalize
from .keypoints import KeyPoints
import mahotas as mh
import numpy as np


class Features:
    @staticmethod
    def histogram(pixels, norm_type=None, timing=False):
        """Create histogram of data
        Returns
        -------
        np.ndarray
            List of histograms
        """
        t0 = time.time()
        if isinstance(pixels, list):
            histograms = []
            for i in range(len(pixels)):
                if isinstance(pixels[i], SimpleNamespace):
                    tmp_pixels = pixels[i].pixels.copy()
                else:
                    tmp_pixels = pixels[i].copy()
                if norm_type is not None:
                    tmp_pixels = Normalize.get_norm(tmp_pixels, norm_type)
                histograms.append(mh.fullhistogram(tmp_pixels.astype(np.uint8)))
        # if pixels is a single image get histogram
        else:
            tmp_pixels = pixels.copy()
            if norm_type is not None:
                tmp_pixels = Normalize.get_norm(tmp_pixels, norm_type)
            histograms = mh.fullhistogram(tmp_pixels.astype(np.uint8))

        if timing:
            print('Histogram time: {}'.format(time.time() - t0))
        return histograms

    @staticmethod
    def sift(pixels,
             norm_type='minmax',
             timing=False,
             **kwargs
             ):
        """Create SIFT features of data
        Parameters:
        ----------
        pixels : np.ndarray | list of np.ndarray
            array of pixels
        norm_type : str, optional
            Type of normalization. The default is minmax.
        downsample : bool, optional
            Downsample image. The default is False.
        timing : bool, optional
            Print timing. The default is False.
        **kwargs : dict
            Additional arguments for sift

        Parameters passable through kwargs:
        ----------
        upsampling: int, optional
            Prior to the feature detection the image is upscaled by a factor of
            1 (no upscaling), 2 or 4. Method: Bi-cubic interpolation.
        n_octaves: int, optional
            Maximum number of octaves. With every octave the image size is
            halved and the sigma doubled. The number of octaves will be
            reduced as needed to keep at least 12 pixels along each dimension
            at the smallest scale.
        n_scales: int, optional
            Maximum number of scales in every octave.
        sigma_min: float, optional
            The blur level of the seed image. If upsampling is enabled
            sigma_min is scaled by factor 1/upsampling
        sigma_in: float, optional
            The assumed blur level of the input image.
        c_dog: float, optional
            Threshold to discard low contrast extrema in the DoG. It’s final
            value is dependent on n_scales by the relation:
            final_c_dog = (2^(1/n_scales)-1) / (2^(1/3)-1) * c_dog
        c_edge: float, optional
            Threshold to discard extrema that lie in edges. If H is the Hessian
            of an extremum, its “edgeness” is described by tr(H)²/det(H).
            If the edgeness is higher than (c_edge + 1)²/c_edge, the extremum
            is discarded.
        n_bins: int, optional
            Number of bins in the histogram that describes the gradient
            orientations around keypoint.
        lambda_ori: float, optional
            The window used to find the reference orientation of a keypoint has
            a width of 6 * lambda_ori * sigma and is weighted by a standard
            deviation of 2 * lambda_ori * sigma.
        c_max: float, optional
            The threshold at which a secondary peak in the orientation histogram
            is accepted as orientation
        lambda_descr: float, optional
            The window used to define the descriptor of a keypoint has a width
            of 2 * lambda_descr * sigma * (n_hist+1)/n_hist and is weighted by
            a standard deviation of lambda_descr * sigma.
        n_hist: int, optional
            The window used to define the descriptor of a keypoint consists of
            n_hist * n_hist histograms.
        n_ori: int, optional
            The number of bins in the histograms of the descriptor patch.
        timing: bool, optional
            Whether to time the function. The default is False.

        Returns
        -------
        ski.feature.sift.Sift
            SIFT descriptor
        """
        t0 = time.time()
        from skimage.feature import SIFT
        tmp_pixels = pixels.copy()
        if norm_type is not None:
            if isinstance(tmp_pixels, list):
                for i in range(len(pixels)):
                    tmp_pixels[i] = Normalize.get_norm(pixels[i], norm_type)
            else:
                tmp_pixels = Normalize.get_norm(pixels, norm_type)

        upsampling = kwargs.get('upsampling', 1)
        n_octaves = kwargs.get('n_octaves', 1)
        n_scales = kwargs.get('n_scales', 1)
        sigma_min = kwargs.get('sigma_min', 1.3)
        sigma_in = kwargs.get('sigma_in', .5)
        c_dog = kwargs.get('c_dog', .7)
        c_edge = kwargs.get('c_edge', .05)
        n_bins = kwargs.get('n_bins', 10)
        lambda_ori = kwargs.get('lambda_ori', .5)
        c_max = kwargs.get('c_max', 1.5)
        lambda_descr = kwargs.get('lambda_descr', .5)
        n_hist = kwargs.get('n_hist', 1)
        n_ori = kwargs.get('n_ori', 1)

        descriptor_extractor = SIFT(upsampling=upsampling,
                                    n_octaves=n_octaves,
                                    n_scales=n_scales,
                                    sigma_min=sigma_min,
                                    sigma_in=sigma_in,
                                    c_dog=c_dog,
                                    c_edge=c_edge,
                                    n_bins=n_bins,
                                    lambda_ori=lambda_ori,
                                    c_max=c_max,
                                    lambda_descr=lambda_descr,
                                    n_hist=n_hist,
                                    n_ori=n_ori)

        if isinstance(tmp_pixels, list):
            descriptors = []
            for i in range(len(tmp_pixels)):
                descriptors.append(descriptor_extractor.detect_and_extract(
                    tmp_pixels[i]))

            if timing:
                print('SIFT: ', time.time() - t0)
            return descriptors
        else:
            descriptor_extractor.detect_and_extract(tmp_pixels)
            if timing:
                print('SIFT: ', time.time() - t0)
            return descriptor_extractor

    @staticmethod
    def sift_keypoint_intensities(imgs,
                                  norm_type='minmax',
                                  timing=False,
                                  **kwargs):
        """Create SIFT keypoints of data
        Parameters
        ----------
        imgs : np.ndarray | list of np.ndarray | SimpleNamespace
            array of imgages
        norm_type : str, optional
            Type of normalization. The default is minmax.
        timing: bool, optional
            Whether to time the function. The default is False.
        **kwargs : dict
            See Features.sift for arguments possible to pass through kwargs
        Returns
        -------
        np.ndarray
            SIFT keypoints
        """
        t0 = time.time()
        pixels = Normalize.extract_pixels(imgs)
        dis_ext = Features.sift(pixels,
                                norm_type=norm_type,
                                timing=timing,
                                **kwargs)

        descriptors = []

        for item in pixels:
            dis_ext.detect_and_extract(item)
            kp = dis_ext.keypoints
            descriptors.append(kp)

        kp = KeyPoints(images=imgs,
                       dis_ext=dis_ext,
                       keypoints=descriptors,
                       norm_type=norm_type,
                       timing=timing,
                       **kwargs)

        if timing:
            print('SIFT: ', time.time() - t0)

        return kp.intensities()

    @staticmethod
    def orb(pixels,
            norm_type='minmax',
            timing=False,
            **kwargs):
        """Create ORB features of data

        Parameters:
        ----------
        pixels: np.ndarray | list of np.ndarray
            array of pixels
        norm_type : str, optional
            Type of normalization. The default is minmax.
        downsample: bool, optional
            Whether to downsample the image to 128x128. The default is True.
        output_shape: tuple, optional
            Shape of the output image. The default is (128, 128).
        timing: bool, optional
            Whether to time the function. The default is False.
        **kwargs : dict
            Additional arguments for ORB

        Parameters passable through kwargs:
        ----------
        n_keypoints: int, optional
            Number of keypoints to be returned. The function will return the
            best n_keypoints according to the Harris corner response if more
            than n_keypoints are detected. If not, then all the detected
            keypoints are returned.
        fast_n: int, optional
            The n parameter in skimage.feature.corner_fast. Minimum number of
            consecutive pixels out of 16 pixels on the circle that should all be
            either brighter or darker w.r.t test-pixel. A point c on the circle
            is darker w.r.t test pixel p if Ic < Ip - threshold and brighter
            if Ic > Ip + threshold. Also stands for the n in FAST-n corner
            detector.
        fast_threshold: float, optional
            The threshold parameter in feature.corner_fast. Threshold used to
            decide whether the pixels on the circle are brighter, darker or
            similar w.r.t. the test pixel. Decrease the threshold when more
            corners are desired and vice-versa.
        harris_k: float, optional
            The k parameter in skimage.feature.corner_harris. Sensitivity factor
            to separate corners from edges, typically in range [0, 0.2]. Small
            values of k result in detection of sharp corners.
        downscale: float, optional
            Downscale factor for the image pyramid. Default value 1.2 is chosen
            so that there are more dense scales which enable robust scale
            invariance for a subsequent feature description.
        n_scales: int, optional
            Maximum number of scales from the bottom of the image pyramid to
            extract the features from.
        timing: bool, optional
            Whether to time the function. The default is False.

        Returns
        -------
        ski.feature.orb.ORB
            ORB descriptor
        """
        t0 = time.time()
        from skimage.feature import (match_descriptors, corner_harris,
                                     corner_peaks, ORB, plot_matches)
        tmp_pixels = pixels.copy()
        tmp_pixels = Normalize.downsample(tmp_pixels)
        if norm_type is not None:
            if isinstance(tmp_pixels, list):
                for i in range(len(tmp_pixels)):
                    tmp_pixels[i] = Normalize.get_norm(pixels=tmp_pixels[i],
                                                       norm_type=norm_type,
                                                       timing=timing)
            else:
                tmp_pixels = Normalize.get_norm(pixels=tmp_pixels,
                                                norm_type=norm_type,
                                                timing=timing)

        n_keypoints = kwargs.get('n_keypoints', 50)
        fast_n = kwargs.get('fast_n', 9)
        fast_threshold = kwargs.get('fast_threshold', 0.08)
        harris_k = kwargs.get('harris_k', 0.04)
        downscale = kwargs.get('downscale', 1.2)
        n_scales = kwargs.get('n_scales', 8)

        descriptor_extractor = ORB(n_keypoints=n_keypoints,
                                   fast_n=fast_n,
                                   fast_threshold=fast_threshold,
                                   harris_k=harris_k,
                                   downscale=downscale,
                                   n_scales=n_scales)

        if isinstance(tmp_pixels, list):
            keypoints = []
            for i in range(len(tmp_pixels)):
                descriptor_extractor.detect(tmp_pixels[i])
                kp = descriptor_extractor.keypoints
                keypoints.append(kp)
            if timing:
                print('ORB: ', time.time() - t0)
            return keypoints
        else:
            descriptor_extractor.detect_and_extract(tmp_pixels)
            keypoints = descriptor_extractor.keypoints
            if timing:
                print('ORB: ', time.time() - t0)
            return keypoints

    @staticmethod
    def orb_keypoint_intensities(imgs,
                                 timing=False,
                                 **kwargs):
        """Create ORB keypoints of data
        Parameters
        ----------
        imgs : np.ndarray | list of np.ndarray | SimpleNamespace
            array of imgages
        norm_type : str, optional
            Type of normalization. The default is minmax.
        timing: bool, optional
            Whether to time the function. The default is False.
        **kwargs : dict
            See Features.orb for arguments possible to pass through kwargs
        Returns
        -------
        np.ndarray
            ORB keypoints
        """
        t0 = time.time()
        from skimage.feature import (match_descriptors, corner_harris,
                                     corner_peaks, ORB, plot_matches)

        n_keypoints = kwargs.get('n_keypoints', 50)
        fast_n = kwargs.get('fast_n', 9)
        fast_threshold = kwargs.get('fast_threshold', 0.08)
        harris_k = kwargs.get('harris_k', 0.04)
        downscale = kwargs.get('downscale', 1.2)
        n_scales = kwargs.get('n_scales', 8)
        output_shape = kwargs.get('output_shape', (128, 128))

        descriptor_extractor = ORB(n_keypoints=n_keypoints,
                                   fast_n=fast_n,
                                   fast_threshold=fast_threshold,
                                   harris_k=harris_k,
                                   downscale=downscale,
                                   n_scales=n_scales)

        pixels = []
        for img in imgs:
            pixels.append(img.pixels)

        downsized_imgs = Normalize.downsample(imgs,
                                              output_shape=output_shape,
                                              normalize=True)

        descriptors = []

        for item in downsized_imgs:
            descriptor_extractor.detect_and_extract(item)
            kp = descriptor_extractor.keypoints
            descriptors.append(kp)

        kp = KeyPoints(images=imgs,
                       dis_ext=descriptor_extractor,
                       keypoints=descriptors,
                       timing=timing,
                       **kwargs)

        if timing:
            print('ORB: ', time.time() - t0)

        return kp.intensities()

    @staticmethod
    def get_features(data,
                     feature_type="hist",
                     norm_type=None,
                     timing=False,
                     **kwargs):
        """Get features of data
        Parameters
        ----------
        data : SimpleNamespace, np.ndarray, list of np.ndarray, any
            array of pixels
        feature_type : str, optional
            Type of feature to extract. The default is "histogram".
        norm_type : str, optional
            Type of normalization. The default is None.
        timing: bool, optional
            Whether to time the function. The default is False.
        Returns
        -------
        np.ndarray, ski.feature.FeatureDetector
        """
        t0 = time.time()
        if feature_type == "hist" or feature_type == "histogram":
            features = Features.histogram(data, norm_type)
        elif feature_type == "sift":
            features = Features.sift_keypoint_intensities(data,
                                                          norm_type,
                                                          timing,
                                                          **kwargs)
        elif feature_type == "orb":
            features = Features.orb_keypoint_intensities(data,
                                                         norm_type,
                                                         **kwargs)
        elif feature_type == "downsample":
            output_shape = kwargs.get('output_shape', (256, 256))
            features = Normalize.downsample(data,
                                            output_shape=output_shape,
                                            flatten=True,
                                            normalize=norm_type,
                                            timing=timing)
        else:
            raise ValueError("Feature type not supported")
        if timing:
            print('Features: ', time.time() - t0)
        return features
