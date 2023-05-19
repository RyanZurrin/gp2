import multiprocessing as mp
import signal
import numpy as np
import tqdm
from scipy.ndimage import zoom


def run_pool_unordered(fxn, inputs, n=None):
    if n is None:
        n = len(inputs)

    orig_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool()
    signal.signal(signal.SIGINT, orig_sigint_handler)
    try:
        results = []
        for z in tqdm.tqdm(pool.imap_unordered(fxn, inputs), total=n):
            results.append(z)
    except KeyboardInterrupt:
        pool.terminate()
        raise Exception("Caught KeyboardInterrupt, terminated workers")
    else:
        print("Normal termination")
        pool.close()
    pool.join()

    return results


def zoom_to_match_size(im, target_im, **kwargs):
    if isinstance(target_im, (tuple, list)):
        t_shape = target_im
    else:
        t_shape = target_im.shape
    v = [float(t_shape[0]) / im.shape[0], float(t_shape[1]) / im.shape[1]]
    if im.ndim == 3:
        v.append(1)
    return zoom(im, v, **kwargs)


# for instance, for post-processing a heatmap 
def post_process_im(im, map_config, proc_info):
    crop_info = proc_info['crop_info']
    orig_size = proc_info['original_shape']
    #transformed_size = proc_info['transformed_size']
    # crop_info, orig_size, transformed_size = dim_info
    # Extract 'trim_image' from the extra info
    # crop_info = []
    # for values in extra_info:
    #     for k in values:
    #         if 'trim_image' in k:
    #             crop_info.append(values[k])

    if im.ndim == 3:
        im = im.mean(axis=-1)
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
        if im.shape != init_target_size:
            im = zoom_to_match_size(im, init_target_size, order=map_config['zoom_order'])
        if np.sum(pad_width):
            if map_config['pad_mode'] == 'median':
                pad_val = np.median(im)
            elif map_config['pad_mode'] == 'zero':
                pad_val = 0.
            im = np.pad(im, pad_width, mode='constant', constant_values=pad_val)
    if im.shape != orig_size:
        im = zoom_to_match_size(im, orig_size, order=map_config['zoom_order'])
    if map_config['convert_to_uint8']:
        im[im < 0] = 0
        im[im > 255] = 255
        im = im.astype(np.uint8)

        return im

    else:
        return im
