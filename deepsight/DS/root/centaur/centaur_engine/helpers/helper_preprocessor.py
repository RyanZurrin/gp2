import numpy as np
import os
import tempfile



def get_orientation_change(image_lat, patient_orientation):
    """
    get the orientation change the image needs before passed into Centaur
    Args:
        image_lat: str. 'L' or 'R' expected
        patient_orientation: str (A|R format expected) or list-str (same format)

    Returns: str specifying the orientation change needed
    One of the following: "rotate_180", "up_down_flip", "left_right_flip", "identity", "unknown"

    """
    if isinstance(patient_orientation, str):
        patient_orientation = patient_orientation.split('|')
    patient_orientation_x = patient_orientation[0]
    patient_orientation_y = patient_orientation[1]

    valid_image_laterality =  ["L", "R", "B"]
    valid_patient_orientation = ["A", "P", "L", "R", "H", "F", "HR", "HL", "FL", "FR"]

    # make sure image_lat value is expected:
    assert image_lat in valid_image_laterality, \
        "invalid value for image_validation:{}".format(image_lat)
    assert all(x in valid_patient_orientation for x in patient_orientation), \
        "invalid value for patient orientation: {}".format(patient_orientation)


    # if Left breast:
    if image_lat == "L":

        # X-axis should point from P to A
        x_flip = patient_orientation_x != "A"

        # y-axis should point from L to R or HL to FR
        y_flip = patient_orientation_y not in ["R", "FR", "F"]

    # if Right breast:
    elif image_lat == "R":

        # X-axis should point from A to P
        x_flip = patient_orientation_x != "P"

        # y-axis should point from R to L or HR to FL
        y_flip = patient_orientation_y not in ["L", "FL", "F"]

    # if B
    else:
        return "unknown"


    # If both axis are flipped then we need to rotate 180
    if x_flip and y_flip:
        return "rotate_180"

    # only y  axis is flipped
    elif y_flip:
        return "up_down_flip"

    # only X axis is flipped
    elif x_flip:
        return "left_right_flip"

    # neither are flipped
    else:
        return "identity"


def implement_orientation(pixel_array, orientation_change):
    """
     Implement the pixel orientation change according to orientation_change
    Args:
        pixel_array:numpy array representing pixel values
        orientation_change: One of the following: "rotate_180", "up_down_flip", "left_right_flip", "identity"

    Returns: numpy array of pixel values that is after transformation

    """
    old_pixel_shape = pixel_array.shape
    if orientation_change == "rotate_180":
        if pixel_array.ndim == 2:
            pixel_array = np.rot90(pixel_array, 2)
        else:
            pixel_array = np.rot90(pixel_array, 2, (1, 2))

    elif orientation_change == "left_right_flip":
        if pixel_array.ndim == 2:
            pixel_array = np.fliplr(pixel_array)
        else:
            pixel_list = []

            for frame in pixel_array:
                slice_array = np.fliplr(frame)
                pixel_list.append(slice_array)
            pixel_array = np.array(pixel_list)

    elif orientation_change == "up_down_flip":
        if pixel_array.ndim == 2:
            pixel_array = np.flipud(pixel_array)

        else:
            pixel_list = []

            for frame in pixel_array:
                slice_array = np.flipud(frame)
                pixel_list.append(slice_array)
            pixel_array = np.array(pixel_list)

    # orientation change one of ["rotate_180", "left_right_flip', "up_down_flip", "identity", None]
    elif orientation_change not in [None, "identity"]:
        raise ValueError("orientation change change value of {} is unexpected".format(orientation_change))
    assert old_pixel_shape == pixel_array.shape, "orientation changes should not change pixel shape, original: {} new:{}".format(old_pixel_shape, pixel_array.shape)
    return pixel_array


def set_pixel_array(args, return_dict=None):
    dr, idx, save_to_ram = args

    # get orientation change and process the changes
    orientation_change = get_orientation_change(dr.metadata['ImageLaterality'], dr.metadata['PatientOrientation'])
    pixel_array = dr.get_ds().pixel_array
    pixel_array = implement_orientation(pixel_array, orientation_change)

    dr.metadata['pixel_array_shape'] = list(pixel_array.shape)
    data = dr.pxl_to_np(pixel_array)
    if not save_to_ram:
        dr.np_to_file(data)
    if return_dict is not None:
        return_dict[idx] = data
    return data


def make_tmp_dir(tmp_dir):
    found_unique = False
    while not found_unique:
        tmp_dir = os.path.join(tmp_dir, next(tempfile._get_candidate_names()))
        if not os.path.exists(tmp_dir):
            found_unique = True
            os.makedirs(tmp_dir)
    return tmp_dir


def nested_index(data, t):
    """Accesses values from a list given nested indicies.

    Note: Example includes accessing val[0][1][0].

    Args:
        data (list): A list to be accessed according to indices in t
        t (int or list): An index or a list of indices.

    Returns:
        str
    """
    if not isinstance(data, list):
        raise TypeError("'data' needs to be a list")
    if not isinstance(t, list) and not isinstance(t, int):
        raise TypeError("'t' needs to be a list or int")

    def _is_dicom_code(arr):

        # unicode is only in python2
        #return any([type(el) == unicode for el in arr]) and len(arr) == 2

        return any([isinstance(el, str) for el in arr]) and len(arr) == 2



    if not isinstance(t, int) and not _is_dicom_code(t):
        v = data[t[0]]
        for tt in t[1:]:
            v = v[tt]
    else:
        v = data[t]
    return v
