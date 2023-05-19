import pydicom
import pdb
import imageio
import numpy as np

from deephealth_utils.data.input_processing import target_and_rand_resize, norm_array

TEST_DXM_IM = './tests/output/test_window_im.png'

def test_target_and_rand_resize():
    X = imageio.imread(TEST_DXM_IM)
    X = np.array(X, float)
    X_out, z = target_and_rand_resize(X, resize_target=(1750, -1), return_extra_info=True)
    pdb.set_trace()

def test_flip_im():
    pass

def test_norm_im():
    X = imageio.imread(TEST_DXM_IM)
    X = np.array(X, float)
    X_out = norm_array(X, 'auto')
    assert X_out.max() == 1.0, 'max is not 1.0 after norm'

def test_orth_rotate():
    pass

if __name__ == '__main__':
    #test_target_and_rand_resize()
    test_norm_im()
