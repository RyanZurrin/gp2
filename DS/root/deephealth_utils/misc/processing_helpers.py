from deephealth_utils.data.input_processing import target_and_rand_resize, flip_im, norm_array, orth_rotate, trim_image_background


def transform_im_from_proc_info(im, proc_info):
    for f in [target_and_rand_resize, flip_im, norm_array, orth_rotate, trim_image_background]:
        fname = f.__name__
        vals = proc_info['proc_funcs'][fname] if 'proc_funcs' in proc_info else proc_info[fname]
        if fname in ['target_and_rand_resize', 'flip_im', 'orth_rotate']:
            im = f(im, vals)
        elif fname == 'norm_array':
            im = f(im, norm_factor=vals)
        elif fname == 'trim_image_background':
            im = f(im, crops=vals[0])
    return im
