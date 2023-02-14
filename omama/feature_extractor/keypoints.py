import time

import numpy as np
# from .features import Features
from .normalization import Normalize


class KeyPoints:
    imageID_to_keypoints = {}
    imageID_to_intensities = {}
    descriptors = []

    def __init__(self,
                 dis_ext,
                 images,
                 keypoints=None,
                 norm_type=None,
                 timing=False,
                 **kwargs):
        t0 = time.time()
        self._dis_ext = dis_ext
        self._norm_type = norm_type
        self._images = images
        self._keypoints = keypoints
        self._timing = timing
        self._options = kwargs
        self._init()
        if timing:
            print("KeyPointss: {}".format(time.time() - t0))

    def __len__(self):
        return len(self._images)

    def _init(self):
        if self._keypoints is None:
            for image in self._images:
                if self._norm_type is not None:
                    pixels = Normalize.get_norm(image.pixels, self._norm_type)
                else:
                    pixels = image.pixels
                kp = self._dis_ext.keypoints
                kp_intensities = self._extract_intensities(pixels, kp)
                self.imageID_to_intensities[
                    image.SOPInstanceUID] = kp_intensities
                self.imageID_to_keypoints[image.SOPInstanceUID] = kp

        else:
            for image, kp in zip(self._images, self._keypoints):
                if self._norm_type is not None:
                    pixels = Normalize.get_norm(image.pixels, self._norm_type)
                else:
                    pixels = image.pixels

                kp_intensities = self._extract_intensities(pixels, kp)
                self.imageID_to_intensities[
                    image.SOPInstanceUID] = kp_intensities
                self.imageID_to_keypoints[image.SOPInstanceUID] = kp
        self._normalize_keypoint_lengths()

    def get_data(self, index):
        sop_uid = self._images[index].SOPInstanceUID
        pixels = self._images[index].pixels
        kp = self.imageID_to_keypoints[sop_uid]
        kp_intensities = self.imageID_to_intensities[sop_uid]
        return sop_uid, pixels, kp, kp_intensities

    @staticmethod
    def _extract_intensities(pixels, keypoints):
        """Extract intensities of pixel under keypoints
        Parameters
        ----------
        pixels : np.ndarray
            array of pixels
        keypoints : list
            keypoints to extract intensities from
        Returns
        -------
        list
            list of intensities
        """
        intensities = []
        for kp in keypoints:
            intensities.append(pixels[int(kp[0]), int(kp[1])])
        return np.array(intensities, dtype=np.float32)

    def intensities(self):
        """returns the intensities as np.ndarray"""
        intensities = []
        for image in self._images:
            kp_array = self.imageID_to_intensities[image.SOPInstanceUID]

            intensities.append(kp_array)
        return intensities

    def _normalize_keypoint_lengths(self):
        """ Normalize keypoint lengths by finding the smallest length of
        keypoints and then selecting a random distribution of keypoints of
        similar length in the rest of the keypoints.
        """
        min_length = min(
            [len(kp) for kp in self.imageID_to_intensities.values()])

        for k, v in self.imageID_to_intensities.items():
            keypoint_used_list = np.random.choice(v,
                                                  min_length,
                                                  replace=False
                                                  )
            self.imageID_to_intensities[k] = keypoint_used_list
