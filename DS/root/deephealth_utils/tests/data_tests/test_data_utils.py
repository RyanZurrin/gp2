import pydicom
import pdb
import imageio

from deephealth_utils.data.utils import window_im

TEST_DXM_DCM = '../../project_data/core-mammo-ml/testing/example_DH0.new_study/DXm.1.2.826.0.1.3680043.9.7134.1.4.0.99943.1505908964.913616'
TEST_DXM_IM = './tests/output/test_window_im.png'

def test_window_im():
    ds = pydicom.dcmread(TEST_DXM_DCM)
    X = window_im(ds.pixel_array, float(ds.WindowCenter), float(ds.WindowWidth))
    imageio.imwrite(TEST_DXM_IM, X)
    X_in = imageio.imread(TEST_DXM_IM)
    pdb.set_trace()


if __name__ == '__main__':
    test_window_im()
