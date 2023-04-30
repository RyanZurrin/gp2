import glob
import os
import numpy as np
import hashlib
import centaur_engine.constants as const
from centaur_engine.helpers.helper_category import CategoryHelper


def transform_input(x, transform_type):
    if transform_type == 'none':
        pass

    elif transform_type == 'lr_flips':
        x = np.fliplr(x)

    elif transform_type == 'vert_flips':
        x = np.rot90(x, 2)

    elif transform_type == 'lr_and_vert_flips':
        x = np.fliplr(np.rot90(x, 2))

    else:
        raise ValueError("transform_type not recognized")

    return x

#
# def zoom_to_match_size(im, target_im, **kwargs):
#     if isinstance(target_im, tuple):
#         t_shape = target_im
#     else:
#         t_shape = target_im.shape
#     v = [float(t_shape[0]) / im.shape[0], float(t_shape[1]) / im.shape[1]]
#     if im.ndim == 3:
#         v.append(1)
#     return zoom(im, v, **kwargs)


def empirical_proportion(x, x_all_sorted):
    if x <= x_all_sorted[0]:
        y = 0.01
    elif x >= x_all_sorted[-1]:
        y = 0.99
    else:
        y = np.mean(x >= x_all_sorted)  # assumes x_all_sorted is large
        # low_idx = x >= x_all_sorted
        # high_idx = x <= x_all_sorted
        # low_val = x_all_sorted[low_idx][-1]
        # high_val = x_all_sorted[high_idx][0]
        # if high_val == low_val:
        #     y = np.mean(low_idx)
        # else:
        #     low_prop = np.mean(low_idx)
        #     y = low_prop + 1. / len(x_all_sorted) * (x - low_val) / (high_val - low_val)

    y = max(y, 0.01)
    y = min(y, 0.99)
    return y


def threshold_and_max_detections(boxes, scores, labels, score_threshold=0.05, max_detections=100):
    """Thresholds the bboxes based on score and caps the max detections

    :return: boxes, scores, labels
    """

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes = boxes[0, indices[scores_sort], :]
    image_scores = scores[scores_sort]
    image_labels = labels[0, indices[scores_sort]]
    # image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    return image_boxes, image_scores, image_labels


# def post_process_im(im, extra_info, map_config, original_size):
#     # Extract 'trim_image' from the extra info
#     crop_info = []
#     for values in extra_info:
#         for k in values:
#             if 'trim_image' in k:
#                 crop_info.append(values[k])
#
#     if im.ndim == 3:
#         im = im.mean(axis=-1)
#     if crop_info is not None:
#         pad_width = [[0, 0], [0, 0]]
#         if isinstance(crop_info, list) and len(crop_info) == 1:
#             crop_info = crop_info[0]
#         orig_size = crop_info[1]
#         for axis in [0, 1]:
#             if crop_info[0][axis][0] > 0:
#                 pad_width[axis][0] = crop_info[0][axis][0]
#             if orig_size[axis] - crop_info[0][axis][1] > 0:
#                 pad_width[axis][1] = orig_size[axis] - crop_info[0][axis][1]
#         if np.sum(pad_width):
#             init_target_size = (crop_info[0][0][1] - crop_info[0][0][0], crop_info[0][1][1] - crop_info[0][1][0])
#         else:
#             init_target_size = orig_size
#         if im.shape != init_target_size:
#             im = zoom_to_match_size(im, init_target_size, order=map_config['zoom_order'])
#         if np.sum(pad_width):
#             if map_config['pad_mode'] == 'median':
#                 pad_val = np.median(im)
#             elif map_config['pad_mode'] == 'zero':
#                 pad_val = 0.
#             im = np.pad(im, pad_width, mode='constant', constant_values=pad_val)
#     if im.shape != original_size:
#         im = zoom_to_match_size(im, original_size, order=map_config['zoom_order'])
#     if map_config['convert_to_uint8']:
#         im[im < 0] = 0
#         im[im > 255] = 255
#         im = im.astype(np.uint8)
#
#         return im
#
#     else:
#         return im

def get_crop_info(proc_info):
    crop_info = []
    for values in proc_info:
        for k in values:
            if 'trim_image' in k:
                crop_info.append(values[k])
    return crop_info


def get_fourway_category(value, thresholds):
    if thresholds[0] > thresholds[1] or thresholds[1] > thresholds[2]:
        raise ValueError("Thresholds not in ascending order.")
    if value < thresholds[0]:
        return 0
    elif value < thresholds[1]:
        return 1
    elif value < thresholds[2]:
        return 2
    else:
        return 3


def get_binary_category(score, cutoff):
    """
    Gets whether a study is suspicious based on its score and on a score cutoff. If the study score is strictly
    smaller than the cutoff, the study is not suspicious. Otherwise, the study is suspicious.
    :param score: float. The study score.
    :param cutoff: float. The suspicious score threshold.
    :return: int. The category number for suspicious or not suspicious based on the study score.
    """
    assert 0 <= cutoff <= 1, f"Cutoff outside of expected range ([0,1]): {cutoff}"
    assert 0 <= score <= 1, f"Score outside of expected range ([0,1]): {score}"
    if score < cutoff:
        return CategoryHelper.NOT_SUSPICIOUS
    return CategoryHelper.SUSPICIOUS


def get_actual_version_number(version_type, version='current'):
    """
    Get model or threshold version.
    If version is 'current', get the actual version.
    Ensure that there is one and only one folder that identifies the version
    Args:
        version_type: str one of  ['model' , 'cadt' , 'cadx']
        version: which version of model or cadt/cadx thresholds

    Returns: version of model or threshold

    """
    paths = {
        'model': const.MODEL_PATH,
        'cadt': const.CADT_THRESHOLD_PATH,
        'cadx': const.CADX_THRESHOLD_PATH
    }
    prefix = ''
    if version_type == 'model':
        prefix = "agg_"
    if version == 'current':
        versions = glob.glob(os.path.join(paths[version_type], prefix+'*'))
    else:
        versions = glob.glob(os.path.join(paths[version_type], version))
        
    if len(versions) == 0:
        raise IOError("{} version could not be found: {}".format(version_type, version))
    if len(versions) > 1:
        raise IOError("More than one version in {}. Please specifyd".format(paths[version_type]))
    else:
        return versions[0].split('/')[-1].split('.')[0]


def get_model_file_hashes():
    hash_dict = {}
    for root, dirs, files in os.walk(const.MODEL_PATH):
        for fn in files:
            full_fn = os.path.join(root, fn)
            file_hash = hashlib.sha1()
            BLOCK_SIZE = 2**13
            with open(full_fn, 'rb') as f:  # Open the file to read its bytes
                fb = f.read(BLOCK_SIZE)  # Read from the file. Take in the amount declared above
                while len(fb) > 0:  # While there is still data being read from the file
                    file_hash.update(fb)  # Update the hash
                    fb = f.read(BLOCK_SIZE)  # Read the next block from the file
            hash_dict[full_fn] = file_hash.hexdigest()
    return hash_dict
