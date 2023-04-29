import sys

import copy
import imageio
import numpy as np
from scipy.misc import imresize
from skimage import transform

if sys.version[0] == '3':  # Python 3.X
    unicode = type(None)


class ImageInputProcessor(object):
    def __init__(self, resize_targets=[None], rand_resize_ranges=[None],
                 flip_params=['rand'], rot_params=[None], norm_factor='auto',
                 trim_background=False, is_contra_pairs=False, convert_to_rgb=False,
                 imagenet_preprocess=False, divisible_factor=None, use_skimage=False):

        verify_init(
            resize_targets, rand_resize_ranges, flip_params,rot_params,
            norm_factor, trim_background, is_contra_pairs,convert_to_rgb,
            imagenet_preprocess, divisible_factor,use_skimage)


        self.n_scales = len(resize_targets)
        self.is_contra_pairs = is_contra_pairs
        self.convert_to_rgb = convert_to_rgb
        self.imagenet_preprocess = imagenet_preprocess
        self.concat_axis = -1

        if len(rot_params) < self.n_scales:
            rot_params += [rot_params[-1]] * (self.n_scales - len(rot_params))

        self.process_funcs = [[target_and_rand_resize, flip_im, norm_array, orth_rotate] for _ in range(self.n_scales)]
        self.process_params = [
            [{'resize_target': resize_targets[i], 'rand_range': rand_resize_ranges[i], 'use_skimage': use_skimage},
             {'flip_param': flip_params[i]}, {'norm_factor': norm_factor},
             {'rot_params': rot_params[i]}] for i in range(self.n_scales)]
        if trim_background:
            for i in range(self.n_scales):
                self.process_funcs[i].insert(2,
                                             trim_image_background)  # put it before norm_array in case that messes up with background calculation
                self.process_params[i].insert(2, {'divisible_factor': divisible_factor})

        if divisible_factor is not None:
            for i in range(self.n_scales):
                self.process_funcs[i].insert(-2, make_size_divisible)
                self.process_params[i].insert(-2, {'divisible_factor': divisible_factor})

    def create_input(self, im_paths, return_extra_info=False, rotate_180=False, view=None):
        if type(im_paths) not in [list, tuple]:
            im_paths = [im_paths]
        if type(rotate_180) not in [list, tuple]:
            rotate_180 = [rotate_180] * len(im_paths)

        orig_ims = []
        for im_n, im_path in enumerate(im_paths):
            # Read Images
            if (not type(im_path) is np.ndarray) and (not type(im_path) is imageio.core.util.Array):
                orig_im = imageio.imread(im_path)
            else:
                orig_im = im_path

            # convert to float since this will be the output type of many of the functions
            # orig_im = np.array(orig_im, float)

            if im_n == 1 and self.is_contra_pairs:
                orig_im = np.fliplr(orig_im)

            if rotate_180[im_n]:
                orig_im = np.rot90(orig_im, 2)

            orig_ims.append(orig_im)

        # If more than one image, resize the height of original images to match the first.
        if len(orig_ims) > 1:
            target_h = orig_ims[0].shape[0]
            for i in range(1, len(orig_ims)):
                if orig_ims[i].shape[0] != target_h:
                    z = float(target_h) / orig_ims[i].shape[0]
                    orig_ims[i] = transform.resize(orig_ims[i], (target_h, int(np.round(orig_ims[i].shape[1] * z))),
                                                   preserve_range=True)

        if self.is_contra_pairs:
            inputs = [None] * self.n_scales * 2
        else:
            inputs = []
        extra_info = []

        for i in range(self.n_scales):
            ims = process_im(orig_ims, process_funcs=self.process_funcs[i], process_params=self.process_params[i],
                             return_extra_info=return_extra_info)

            if return_extra_info:
                extra_info.append(ims[1])
                ims = ims[0]

            # pad images if they are not of equal size
            if len(ims) > 1:
                max_h = 0
                max_w = 0
                for im in ims:
                    max_h = max(max_h, im.shape[0])
                    max_w = max(max_w, im.shape[1])
                for im_i, im in enumerate(ims):
                    nh = max_h - im.shape[0]
                    nw = max_w - im.shape[1]
                    if nh > 0 or nw > 0:
                        n_pad = [[0, nh], [0, 0]]
                        if nw:
                            mid = int(im.shape[1] / 2)
                            mean_l = np.mean(im[:, :mid])
                            mean_r = np.mean(im[:, mid:])
                            if mean_l > mean_r:
                                n_pad[1][1] = nw
                            else:
                                n_pad[1][0] = nw
                        im = np.pad(im, n_pad, mode='constant', constant_values=np.min(im))
                        ims[im_i] = im

            # make so input is like (batch size,) +  im.shape
            for j, im in enumerate(ims):
                while im.ndim < 4:
                    im = im.reshape((1,) + im.shape)
                im = validate_channel_axis(im)
                ims[j] = im

            n_ims = len(ims)
            if n_ims == 1:
                inp = ims[0]
            else:
                if self.is_contra_pairs:
                    if n_ims == 2:
                        inputs[i] = ims[0]
                        inputs[i + self.n_scales] = ims[1]
                    else:  # for having multiple channels/slices
                        inputs[i] = np.concatenate(ims[:n_ims / 2], self.concat_axis)
                        inputs[i + self.n_scales] = np.concatenate(ims[n_ims / 2:], self.concat_axis)
                    continue
                else:
                    inp = np.concatenate(ims, self.concat_axis)
            inputs.append(inp)

        if self.convert_to_rgb:
            for k in range(len(inputs)):
                if inputs[k].shape[-1] == 1:
                    inputs[k] = np.tile(inputs[k], 3)

        if self.imagenet_preprocess:
            if isinstance(self.imagenet_preprocess, (str, unicode)) and self.imagenet_preprocess == '-1_1':
                for k in range(len(inputs)):
                    inputs[k] -= 0.5
                    inputs[k] *= 2
            else:
                for k in range(len(inputs)):
                    inputs[k] *= 255.
                    inputs[k] -= 255. / 2

        if return_extra_info:
            return inputs, extra_info
        else:
            return inputs

def verify_init(
    resize_targets,
    rand_resize_ranges,
    flip_params,
    rot_params,
    norm_factor,
    trim_background,
    is_contra_pairs,
    convert_to_rgb,
    imagenet_preprocess,
    divisible_factor,
    use_skimage,
):

    # copy to avoid mutating lists:
    resize_targets = copy.deepcopy(resize_targets)
    rot_params = copy.deepcopy(rot_params)
    flip_params = copy.deepcopy(flip_params)
    rand_resize_ranges = copy.deepcopy(rand_resize_ranges)

    list_names = ["resize_targets", "rot_params", "flip_params", "rand_resize_ranges"]
    list_params = [resize_targets, rot_params, flip_params, rand_resize_ranges]

    # Check if len of arrays match:
    for i, list_param in enumerate(list_params):

        if type(list_param) not in [list, tuple]:

            message = "{} should be list or tuple, it is type {}".format(
                list_names[i], type(list_params[i])
            )
            raise ValueError(message)

        # list length doesnt match
        if len(list_param) != len(resize_targets):

            # rot_params can extend its length with the last element
            if list_names[i] == "rot_params":
                if len(rot_params) > len(resize_targets):
                    message = format_error_messages("len_mismatch", ("rot_params",))
                    raise ValueError(message)
                else:
                    rot_params += [rot_params[-1]] * (len(resize_targets) - len(rot_params))


            else:
                message = format_error_messages("len_mismatch", (list_names[i],))
                raise ValueError(message)



    # Check each element of list parameters
    for i in range(len(resize_targets)):
        rt = resize_targets[i]
        rsr = rand_resize_ranges[i]
        fp = flip_params[i]
        rp = rot_params[i]

        # resize_targets[i] must be number, None or list tuple
        if not (type(rt) in [float, int, tuple, list, type(None)]):
            message = format_error_messages(
                "wrong datatype", (str(i), "resize_targets")
            )
            raise ValueError(message)

        # resize_targets[i]  with length 2 or 3
        if type(rt) in [list, tuple]:
            if len(rt) < 2:
                message = "invalide size at index {} of resize_targets".format(i)
                raise ValueError(message)
            elif len(rt) > 3:
                message = "resize_targets[i] got length greater than 3 at index {}".format(
                    i
                )
                raise ValueError(message)


        # random_size_range[i] should be list, tuple, bool:
        if type(rsr) not in [list, tuple, type(None), bool]:
            message = format_error_messages("wrong datatype", (i, "random_size_range"))
            raise ValueError(message)

        # flip_params [i] is None , bool or float <= 1 >=0.
        if type(fp) not in [float, int, bool, type(None)]:
            message = format_error_messages("wrong datatype", (str(i), "flip_params"))
            raise ValueError(message)

        # when flip_params[i] is a number
        if type(fp) in [float, int] and( fp > 1 or fp <0):
            message = "flip probability shouldn't be greater than 1 or smaller than 0: index {} of flip_params".format(
                i
            )
            raise ValueError(message)

        # rot_params[i]: one of  [NoneType, 'rand','rand_vert']
        if not (rp is None or rp == "rand" or rp == "rand_vert"):
            message = "rot_params[i] has to be one of  [NoneType, 'rand','rand_vert']"
            raise ValueError(message)

    # norm_factors: one of ['auto','0_1',float]
    if not (
        norm_factor == "auto"
        or norm_factor == "0_1"
        or type(norm_factor) in [float, int]
    ):
        message = "norm_factor should be one of ('auto','0_1',float)"
        raise ValueError(message)

    # We do not use skimage anymore:
    if use_skimage:
        raise ValueError("use_skimage should be False")

        # No longer processing contra_pairs
    if is_contra_pairs:
        raise ValueError("is_contra_pairs should be false")

    # functions that are :bool  or None
    bool_params = [
        trim_background,
        convert_to_rgb,
    ]

    bool_param_names = [
        "trim_background",
        "convert_to_rgb",
    ]


    # wrong types but would not throw Errors
    for i, param in enumerate(bool_params):
        if not (isinstance(param, bool) or param is None):
            message = "{} should be bool or NoneType".format(bool_param_names[i])
            raise ValueError(message)

    # divisible_factor has to be None
    if not (divisible_factor is None):
        message = "divisible_factor has to be None"
        raise ValueError(message)

    # imagenet_preprocess should one of [True, False, '-1_1']
    if imagenet_preprocess not in [True, False, '-1_1']:
        raise ValueError("imagenet_preprocess should one of [True, False, '-1_1'] ")



def format_error_messages(message_type, args=("",)):
    """
    Helper Function to format error and warning messages
    """
    message_dict = {
        "len_mismatch": "length of {} does not match len(resize_targets)",
        "len too small": "length of {} is less than len(resize_targets) will result in IndexError",
        "wrong datatype": "wrong parameter type at index {} of {}",
        "target size and random": "at index {}, resize_target is specified but random_resize_range is not None. When randomly resizing cant specify target_resize ",
    }
    return message_dict[message_type].format(*args)

def process_im(orig_im, process_funcs=[], process_params=[], return_extra_info=False):
    im = copy.deepcopy(orig_im)
    process_params = copy.deepcopy(process_params)
    extra_info = {}
    for i in range(len(process_funcs)):
        if return_extra_info:
            if process_params[i] is None:
                process_params[i] = {'return_extra_info': True}
            else:
                process_params[i]['return_extra_info'] = True
        if process_params[i] is None:
            im = process_funcs[i](im)
        else:
            im = process_funcs[i](im, **process_params[i])
        if return_extra_info:
            extra_info[process_funcs[i].__name__] = im[1]
            im = im[0]

    if return_extra_info:
        return im, extra_info
    else:
        return im


def target_and_rand_resize(im, resize_target=None, rand_range=None, return_extra_info=False, use_skimage=False):
    mult_ims = type(im) in (list, tuple)
    if not mult_ims:
        im = [im]

    z = calc_resize_factor(im[0], resize_target, rand_range)
    if z != 1:
        for k in range(len(im)):
            if use_skimage:
                output_shape = [int(np.round(im[k].shape[i] * z)) for i in [0, 1]]
                im[k] = transform.resize(im[k], output_shape, preserve_range=True)
            else:
                im[k] = imresize(im[k], z)
    if not mult_ims:
        im = im[0]

    if return_extra_info:
        return im, z
    else:
        return im


def flip_im(im, flip_param=None, return_extra_info=False):
    do_flip = False
    if flip_param is not None:
        if isinstance(flip_param, bool) and flip_param:
            do_flip = True
        elif np.random.rand() < flip_param:
            do_flip = True

    if do_flip:
        if isinstance(im, list):
            im = [np.fliplr(i) for i in im]
        else:
            im = np.fliplr(im)

    if return_extra_info:
        return im, do_flip
    else:
        return im


def norm_array(X, norm_factor=None, return_extra_info=False):
    mult_ims = type(X) in (list, tuple)
    if not mult_ims:
        X = [X]

    for i, im in enumerate(X):
        if norm_factor is not None:
            if norm_factor == '0_1':
                if 'float' not in str(im.dtype):
                    im = im.astype(np.float32)
                im = (im - im.min()) / (im.max() - im.min())
            else:
                # hot fix, need to change
                im = im.astype(np.float32)
                if norm_factor == 'auto':
                    max_val = np.max(im)
                    did_renorm = False
                    for v in [2 ** 8 - 1, 2 ** 10 - 1, 2 ** 12 - 1, 2 ** 14 - 1, 2 ** 16 - 1]:
                        if max_val <= v and not did_renorm:
                            im = im / v
                            did_renorm = True
                else:
                    im = im / norm_factor

        X[i] = im

    if not mult_ims:
        X = X[0]

    if return_extra_info:
        return X, norm_factor
    else:
        return X


def rand_im_resize(im, low, high):
    im = imresize(im, np.random.uniform(low, high))
    return im


def orth_rotate(im, rot_params=None, return_extra_info=False):
    n = 0
    if rot_params is not None:
        if rot_params == 'rand':
            n = np.random.randint(4)
        elif isinstance(rot_params, int):
            n = rot_params
        elif rot_params == 'rand_vert':
            n = np.random.choice([0, 2])

    if n > 0:
        mult_ims = type(im) in (list, tuple)
        if not mult_ims:
            im = [im]

        im = [np.rot90(i, n) for i in im]

        if not mult_ims:
            im = im[0]

    if return_extra_info:
        return im, n
    else:
        return im


def calc_resize_factor(im, resize_target=None, rand_range=None):
    z = 1.
    # pdb.set_trace()
    if resize_target is not None:
        if hasattr(resize_target, '__len__'):
            r = [float(resize_target[k]) / im.shape[k] for k in [0, 1]]
            r = [v for v in r if v > 0]
            min_r = np.min(r)
            max_r = np.max(r)
            if len(resize_target) == 3:
                z = min_r + (max_r - min_r) * resize_target[2]
            else:
                z = np.random.uniform(min_r, max_r)
        else:
            z = resize_target

    if rand_range is not None:
        z *= np.random.uniform(rand_range[0], rand_range[1])
    z = float(z)

    return z


def trim_image_background(ims, background_lvl=0, n_buffer=20, crops=None, divisible_factor=None,
                          return_extra_info=False):
    mult_ims = type(ims) in (list, tuple)
    if not mult_ims:
        ims = [ims]

    if crops is None:
        all_crops = []
        for im in ims:
            if np.max(im) <= background_lvl:
                crops = None
            else:
                crops = []
                for a in [1, 0]:
                    vals = np.max(im, axis=a)
                    idx = np.nonzero(vals > background_lvl)[0]
                    if a == 0:
                        idx_back = np.nonzero(vals <= background_lvl)[0]
                        if len(idx_back):
                            if idx_back[0] >= len(vals) - 1 - idx_back[-1]:  # breast on left side of image
                                i0 = 0
                                i1 = idx_back[0]
                            else:
                                i0 = idx_back[-1]
                                i1 = len(vals) - 1
                        else:
                            i0 = idx[0]
                            i1 = idx[-1]
                    else:
                        i0 = idx[0]
                        i1 = idx[-1]
                    v0 = int(max(0, i0 - n_buffer))
                    v1 = int(min(len(vals), i1 + n_buffer))
                    crops.append([v0, v1])
            all_crops.append(crops)

        all_crops = [crop for crop in all_crops if crop is not None]
        if len(all_crops):
            crops = all_crops[0]
            if len(all_crops) > 1:
                for j in range(1, len(all_crops)):
                    for k in range(2):
                        crops[k][0] = min(crops[k][0], all_crops[j][k][0])
                        crops[k][1] = max(crops[k][1], all_crops[j][k][1])
        else:
            crops = None

    if crops is not None:
        if divisible_factor is not None:  # make image size divisible by divisible_factor
            for axis in [0, 1]:
                sz = crops[axis][1] - crops[axis][0]
                delta = sz % divisible_factor
                if delta:
                    delta = divisible_factor - delta
                    avail = [crops[axis][0], ims[0].shape[axis] - crops[axis][1]]
                    to_pad = [0, 0]
                    min_axis = np.argmin(avail)
                    to_pad[min_axis] = min(avail[min_axis], delta / 2)
                    other_axis = 1 if min_axis == 0 else 0
                    to_pad[other_axis] = min(delta - to_pad[min_axis], avail[other_axis])
                    crops[axis][0] -= to_pad[0]
                    crops[axis][1] += to_pad[1]

        for i, im in enumerate(ims):
            ims[i] = im[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]

    if not mult_ims:
        ims = ims[0]

    if return_extra_info:
        return ims, (crops, im.shape)
    else:
        return ims


def make_size_divisible(ims, divisible_factor, pad_value=0, return_extra_info=False):
    mult_ims = type(ims) in (list, tuple)
    if not mult_ims:
        ims = [ims]

    im = ims[0]
    to_pad = [0, 0]
    pad_direction = [0, 0]
    for axis in [0, 1]:
        delta = im.shape[axis] % divisible_factor
        if delta:
            to_pad[axis] = divisible_factor - delta
        if to_pad[axis]:
            if axis == 0:
                if np.mean(im[0, :]) > np.mean(im[-1, :]):
                    pad_direction[axis] = 1
            elif np.mean(im[:, 0]) > np.mean(im[:, -1]):
                pad_direction[axis] = 1

    pad_width = [[0, 0], [0, 0]]
    if np.sum(to_pad):
        for axis in [0, 1]:
            pad_width[axis][pad_direction[axis]] = to_pad[axis]
        for i in range(len(ims)):
            ims[i] = np.pad(ims[i], pad_width, mode='constant', constant_values=pad_value)

    if not mult_ims:
        ims = ims[0]

    if return_extra_info:
        return ims, pad_width
    else:
        return ims


def validate_channel_axis(X, data_format='channels_last'):
    if X.ndim == 3:
        if X.shape[0] in [1, 3] and data_format == 'channels_last':
            X = X.transpose((1, 2, 0))
        elif X.shape[2] in [1, 3] and data_format == 'channels_first':
            X = X.transpose((2, 0, 1))
    elif X.ndim == 4:
        if X.shape[1] in [1, 3] and data_format == 'channels_last':
            X = X.transpose((0, 2, 3, 1))
        elif X.shape[3] in [1, 3] and data_format == 'channels_first':
            X = X.transpose((0, 3, 1, 2))
    return X


class FileToInputsProcessor(object):
    def __init__(self, dcm_process_fxn, array_process_fxn):
        self.dcm_process_fxn = dcm_process_fxn
        self.array_process_fxn = array_process_fxn

    def create_input(self, file_path, **kwargs):
        is_image_file = False
        for s in ['.png', '.jpg', '.jpeg', '.tiff']:
            if s in file_path:
                is_image_file = True
                break

        if is_image_file:
            x = imageio.imread(file_path)
        else:
            x = self.dcm_process_fxn(file_path)

        return self.array_process_fxn(x, **kwargs)

# should use this from keras
# def to_categorical(y, num_classes=None):
