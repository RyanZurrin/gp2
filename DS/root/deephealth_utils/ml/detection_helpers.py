import numpy as np
from scipy.ndimage import zoom
from deephealth_utils.misc.processing_helpers import transform_im_from_proc_info


def box_coords_to_original(box_coords, proc_info):
    '''
    Take box coordinates that are outputted by model and transform them back to respect to the original image.
    '''
    crop_info = proc_info['crop_info']
    original_shape = proc_info['original_shape']
    input_shape = proc_info['transformed_shape']
    # crop_info, original_shape, input_shape = dim_info

    if len(box_coords) == 0:
        return box_coords
    transformed_box_coords = np.zeros_like(box_coords)
    for b_num, box in enumerate(box_coords):
        c_im = np.zeros(input_shape)
        box = [int(np.round(b)) for b in box]
        assert box[2] > box[0], 'Invalid box coordinates, box[2] <= box[0]'
        assert box[3] > box[1], 'Invalid box coordinates, box[3] <= box[1]'

        c_im[box[1]:box[3], box[0]:box[2]] = 1

        if crop_info is not None:
            pad_width = [[0, 0], [0, 0]]
            if isinstance(crop_info, list) and len(crop_info) == 1:
                crop_info = crop_info[0]
            orig_size = crop_info[1]
            for axis in [0, 1]:
                if crop_info[0][axis][0] > 0:
                    pad_width[axis][0] = crop_info[0][axis][0]
                if orig_size[axis] - crop_info[0][axis][1] > 0:
                    pad_width[axis][1] = orig_size[axis] - crop_info[0][axis][1]
            if np.sum(pad_width):
                init_target_size = (crop_info[0][0][1] - crop_info[0][0][0], crop_info[0][1][1] - crop_info[0][1][0])
            else:
                init_target_size = orig_size

            if c_im.shape != init_target_size:
                v = [float(init_target_size[0]) / c_im.shape[0], float(init_target_size[1]) / c_im.shape[1]]
                c_im = zoom(c_im, v, order=0)

            if np.sum(pad_width):
                c_im = np.pad(c_im, pad_width, mode='constant')

        if c_im.shape != original_shape:
            v = [float(original_shape[0]) / c_im.shape[0], float(original_shape[1]) / c_im.shape[1]]
            c_im = zoom(c_im, v, order=0)

        b_idx = np.nonzero(c_im >= 1)
        transformed_box_coords[b_num] = [b_idx[1].min(), b_idx[0].min(), b_idx[1].max(), b_idx[0].max()]

    return transformed_box_coords


def box_coords_to_transformed(box_coords, proc_info, orig_shape=None):
    '''
    Take box coordinates that are with respect to original image and transform them according to how the image was transformed.
    :param box_coords: list of float (num_bbxs X 4-5)
    :param proc_info: Image processing info
    :param orig_shape: 2-int-list. Image original shape. If not present, use proc_info to extract it
    :return:
           numpy array with num_bbx x 4-5
    '''
    if orig_shape is None:
        orig_shape = proc_info['original_shape']
    if len(box_coords) == 0:
        return box_coords
    box_coords = np.array(box_coords)

    transformed_box_coords = np.zeros((0, box_coords.shape[-1]))
    for b_num, box in enumerate(box_coords):
        c_im = np.zeros(orig_shape)
        box_list = [int(np.round(b)) for b in box]
        c_im[box_list[1]:box_list[3], box_list[0]:box_list[2]] = 1
        c_im = transform_im_from_proc_info(c_im, proc_info)

        b_idx = np.nonzero(c_im >= 1)
        assert len(b_idx[0]) > 0, "Bbx {} with coordinates '{}' could not be transformed".format(b_num, box)
        b_coords = [b_idx[1].min(), b_idx[0].min(), b_idx[1].max(), b_idx[0].max()]
        for v in range(4, box_coords.shape[-1]):  # in case, for instance, box_coords also contain labels at the end
            b_coords.append(box[v])
        b_coords = np.array(b_coords).reshape((1, len(b_coords)))
        transformed_box_coords = np.vstack((transformed_box_coords, b_coords))

    return transformed_box_coords.astype(np.int)
