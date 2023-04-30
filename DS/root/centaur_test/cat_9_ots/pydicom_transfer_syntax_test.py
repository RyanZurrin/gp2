import os
from pydicom.data import get_testdata_file
import pydicom
import PIL
import numpy as np
import shutil
import subprocess
# from centaur_test.datasets.data_manager import DataManager
import tempfile

UID_map = {
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2": 'Implicit VR Little Endian',
    "1.2.840.10008.1.2.2": 'Explicit VR Big Endian',
    '1.2.840.10008.1.2.1.99': 'Deflated Explicit VR Little Endian',
    '1.2.840.10008.1.2.5': 'RLE Lossless',
    '1.2.840.10008.1.2.4.57': 'JPEG Lossless (Process 14)',
    '1.2.840.10008.1.2.4.70': 'JPEG Lossless (Process 14, SV1)',
    '1.2.840.10008.1.2.4.80': 'JPEG LS Lossless',
    '1.2.840.10008.1.2.4.90': 'JPEG2000 Lossless'

}

for (k, v) in list(UID_map.items()):
    UID_map[v] = k


class DCMTKError(Exception):

    def __int__(self, message):
        """
        specific error for errors resulting from when encoding a dicom in a specific transfer syntax
        """
        super().__init__(message)


def get_test_files():
    """
    # get all files for DS_01 for testing pydicom_transfer_syntax
    # Returns: list of file paths
    We are going to use pydicom's own test image

    """
    # dm = DataManager()
    # file_df = dm.get_filelist_df()
    # base_dir = dm._data_dir
    # paths = [os.path.join(base_dir, x) for x in file_df.file_path.values]
    # return paths
    fpath = get_testdata_file('CT_small.dcm')
    return [fpath]


def get_pixels_from_raw(raw_data, image_shape):
    """
    convert raw_data in little endian into pixels values
    Args:
        raw_data (bytes): raw bytes in little endian, e.g. b'\xff\xf8\x00\x00\x00\x00\x00\x00'
        image_shape: the shape of the resulting image

    Returns: numpy array of pixel values

    """

    # total number of pixels
    num_pixels = 1
    for s in image_shape:
        num_pixels *= s

    # how many bytes specify one pixel value
    num_bytes_per_pixel = int(len(raw_data) / num_pixels)

    # making sure that the number of bytes make sense for the image size.
    assert len(raw_data) % num_pixels == 0, "size of data doesnt make sense"

    # loop through the raw data and append the pixel values
    pixels = []
    for i in range(num_bytes_per_pixel, len(raw_data) + 1, num_bytes_per_pixel):
        byte = raw_data[i - num_bytes_per_pixel: i]
        pixel = int.from_bytes(byte, byteorder='little')
        pixels.append(pixel)
    pixels = np.array(pixels).reshape(image_shape)

    return pixels


def print_difference(array1, array2):
    """
    given two arrays of the same shape prints the location of mismatch, and the
    values at the mismatch location
    Args:
        array1: numpy array
        array2: numpy array

    Returns: None

    """
    # no need to print anything
    if np.array_equal(array1, array2):
        return

    match_array = array1 == array2

    unique_elements = np.unique(match_array, return_counts=True)

    print("locations of missmatch", np.where(match_array == False)[0])
    print("number of matches (True) and number of mismatches (False)", unique_elements)

    print("array1", array1[match_array == False])
    print("array2", array2[match_array == False])

    diff = array1 - array2
    diff = diff[match_array == False]

    print('array1 - array2', diff)


def compare_dcmtk_pydicom(original_file, file_encoded, output_folder, language):
    """
    compare if the pydicom reading of the files encoded in a transfer syntax
    result in matching pixels of DCMTK reading of the original file
    Args:
        original_file: the original uncompressed file
        file_encoded: encoded or compressed file
        output_folder: where is the intermediate files will be stored

    Returns: bool, whether or not the pydicom reading of pixels match that of DCMTK

    """
    command_read = "dcmdump +W {} {}".format(output_folder, original_file)
    val = subprocess.call(command_read, shell=True)
    if val != 0:
        raise DCMTKError("DCMTK dcmdump for {} produced an error".format(original_file))

    ds = pydicom.dcmread(file_encoded)
    pydicom_pixels = ds.pixel_array
    assert ds.file_meta.TransferSyntaxUID == language, "wrong transfer syntax language expected {} got {}".format(
        UID_map[language], UID_map[str(ds.file_meta.TransferSyntaxUID)])

    # dcmtk sometimes dump the pixel data onto multiple raw files therefore
    # we need to index and concatenate
    raw_index = 0
    raw_file_name = os.path.basename(original_file) + '.{}.raw'.format(raw_index)
    raw_file_path = os.path.join(output_folder, raw_file_name)
    dcmtk_data = b''

    while os.path.exists(raw_file_path):
        with open(raw_file_path, "rb") as file:
            data = file.read()
        dcmtk_data += data
        raw_index += 1
        raw_file_name = os.path.basename(original_file) + '.{}.raw'.format(raw_index)
        raw_file_path = os.path.join(output_folder, raw_file_name)

    if len(dcmtk_data) == 0:
        raise DCMTKError("dcmdump did not find any pixel data")

    dcmtk_pixels = get_pixels_from_raw(dcmtk_data, pydicom_pixels.shape)

    if np.array_equal(dcmtk_pixels, pydicom_pixels):
        return True
    else:
        print_difference(dcmtk_pixels, pydicom_pixels)


def dcm_conv_function(ts_dicom, dcmfile_out, option):
    """
    calls the DCKTK's dcmconv function
    Args:
        dcmfile_in: dicom file to convert
        dcmfile_out: dicome file to write the converted data
        option: specifies the type of encoding

    Returns: None

    """
    command_encode = "dcmconv {} {} {}".format(option, ts_dicom, dcmfile_out)
    val = subprocess.call(command_encode, shell=True)
    if val != 0:
        raise DCMTKError("DCMTK encoding failed for {}".format(ts_dicom))


def dcmcjpeg_function(ts_dicom, dcmfile_out, option):
    """
    calls the DCKTK's dcmcjpeg function

    dcmfile_in: dicom file to convert
    dcmfile_out: dicome file to write the converted data
    option: specifies the type of encoding

    Returns: None

    """
    command_encode = "dcmcjpeg {} {} {}".format(option, ts_dicom, dcmfile_out)
    val = subprocess.call(command_encode, shell=True)
    if val != 0:
        raise DCMTKError("DCMTK encoding failed for {}".format(ts_dicom))


def helper_explicit_vr_little_enidan(ts_dicom, output_folder):
    """
    creates a explicit  vr little endian encoded file and compare pydicom reading of the
    encoded file and the DCMTK reading of the original file
    Args:
        ts_dicom: dicom for which the test is done on
        output_folder: temporary folder that stores files needed for computation and
        should be deleted after the test is finished

    Returns: None

    """

    option = "+te"
    dcmfile_out = "{}/explicit_vr_little_enidan.dcm".format(output_folder)
    dcm_conv_function(ts_dicom, dcmfile_out, option)
    match = compare_dcmtk_pydicom(ts_dicom, dcmfile_out, output_folder, UID_map['Explicit VR Little Endian'])
    assert match, "pixels values from pydicom does not match that of DCMTK"


def helper_implicit_vr_little_endian(ts_dicom, output_folder):
    """
    creates a implicit vr little endian encoded file and compare pydicom reading of the
    encoded file and the DCMTK reading of the original file
    Args:
        ts_dicom: dicom for which the test is done on
        output_folder: temporary folder that stores files needed for computation and
        should be deleted after the test is finished

    Returns: None

    """

    option = "+ti"
    dcmfile_out = "{}/implicit_vr_little_endian.dcm".format(output_folder)
    dcm_conv_function(ts_dicom, dcmfile_out, option)
    match = compare_dcmtk_pydicom(ts_dicom, dcmfile_out, output_folder, UID_map['Implicit VR Little Endian'])
    assert match, "pixels values from pydicom does not match that of DCMTK"


def helper_explicit_vr_big_endian(ts_dicom, output_folder):
    """
    creates a explicit VR big endian encoded file and compare pydicom reading of the
    encoded file and the DCMTK reading of the original file
    Args:
        ts_dicom: dicom for which the test is done on
        output_folder: temporary folder that stores files needed for computation and
        should be deleted after the test is finished

    Returns: None

    """

    option = "+tb"
    dcmfile_out = "{}/explicit_vr_big_endian.dcm".format(output_folder)
    dcm_conv_function(ts_dicom, dcmfile_out, option)
    match = compare_dcmtk_pydicom(ts_dicom, dcmfile_out, output_folder, UID_map['Explicit VR Big Endian'])
    assert match, "pixels values from pydicom does not match that of DCMTK"


def helper_deflated_explicit_vr_little_endian(ts_dicom, output_folder):
    """
        creates a deflated explicit vr little endian encoded file and compare pydicom reading of the
        encoded file and the DCMTK reading of the original file
        Args:
            ts_dicom: dicom for which the test is done on
            output_folder: temporary folder that stores files needed for computation and
            should be  deleted after the test is finished

        Returns: None

    """

    option = "+td"
    dcmfile_out = "{}/deflated_explicit_vr_little_endian.dcm".format(output_folder)
    dcm_conv_function(ts_dicom, dcmfile_out, option)
    match = compare_dcmtk_pydicom(ts_dicom, dcmfile_out, output_folder, UID_map['Deflated Explicit VR Little Endian'])
    assert match, "pixels values from pydicom does not match that of DCMTK"


def helper_rle_lossless(ts_dicom, output_folder):
    """
        creates a RLE Lossless encoded file and compare pydicom reading of the
        encoded file and the DCMTK reading of the original file
        Args:
            ts_dicom: dicom for which the test is done on
            output_folder: temporary folder that stores files needed for computation and
            should be deleted after the test is finished

        Returns: None

    """
    dcmfile_out = "{}/rle_lossless.dcm".format(output_folder)

    command = "dcmcrle {} {}".format(ts_dicom, dcmfile_out)
    val = subprocess.call(command, shell=True)

    if val != 0:
        raise DCMTKError("DCMTK encoding failed for {}".format(ts_dicom))

    match = compare_dcmtk_pydicom(ts_dicom, dcmfile_out, output_folder, UID_map['RLE Lossless'])
    assert match, "pixels values from pydicom does not match that of DCMTK"


# JPEG Lossless (Process 14, SV1)
def helper_jpeg_lossless_process_14_sv1(ts_dicom, output_folder):
    """
        creates a jpeg lossless process 14 SV1 and compare pydicom reading of the
        encoded file and the DCMTK reading of the original file
        Args:
            ts_dicom: dicom for which the test is done on
            output_folder: temporary folder that stores files needed for computation and
            should be deleted after the test is finished

        Returns: None

    """

    option = '+e1'
    dcmfile_out = "{}/jpeg_lossless_process_14_sv1.dcm".format(output_folder)
    dcmcjpeg_function(ts_dicom, dcmfile_out, option)
    match = compare_dcmtk_pydicom(ts_dicom, dcmfile_out, output_folder, UID_map['JPEG Lossless (Process 14, SV1)'])
    assert match, "pixels values from pydicom does not match that of DCMTK"


def helper_j2k(ts_dicom, output_folder):
    """
    creates a jpeg2000  encoded file and compare pydicom reading of the
    encoded file and the DCMTK reading of the original file
    Args:
        ts_dicom: dicom for which the test is done on
        output_folder: temporary folder that stores files needed for computation and
        should be deleted after the test is finished

    Returns: None

    """
    encoded_file = "{}/j2k.dcm".format(output_folder)
    command = "gdcmconv --j2k {} {}".format(ts_dicom, encoded_file)
    val = subprocess.call(command, shell=True)

    if val != 0:
        raise DCMTKError("DCMTK encoding failed for {}".format(ts_dicom))

    ds_orig = pydicom.dcmread(ts_dicom)
    ds_encoded = pydicom.dcmread(encoded_file)

    assert ds_encoded.file_meta.TransferSyntaxUID == UID_map[
        "JPEG2000 Lossless"], "wrong Transfer syntax expected {} got {}".format("JPEG2000 Lossless", UID_map[
        str(ds_encoded.file_meta.TransferSyntaxUID)])
    pix_orig = ds_orig.pixel_array
    pix_encoded = ds_encoded.pixel_array

    np.array_equal(pix_orig, pix_encoded), "pixel values did not match"


# JPEG LS Lossless
def helper_jpeg_ls_lossless(ts_dicom, output_folder):
    """
    creates a jpeg ls lossless encoded file and compare pydicom reading of the
    encoded file and the DCMTK reading of the original file
    Args:
        ts_dicom: dicom for which the test is done on
        output_folder: temporary folder that stores files needed for computation and
        should be deleted after the test is finished

    Returns: None

    """
    encoded_file = "{}/j2k.dcm".format(output_folder)
    command = "gdcmconv --jpegls  {} {}".format(ts_dicom, encoded_file)
    val = subprocess.call(command, shell=True)

    if val != 0:
        raise DCMTKError("DCMTK encoding failed for {}".format(ts_dicom))

    ds_orig = pydicom.dcmread(ts_dicom)
    ds_encoded = pydicom.dcmread(encoded_file)

    assert ds_encoded.file_meta.TransferSyntaxUID == UID_map[
        'JPEG LS Lossless'], "wrong Transfer syntax expected {} got {}".format("JPEG2000 Lossless", UID_map[
        str(ds_encoded.file_meta.TransferSyntaxUID)])
    pix_orig = ds_orig.pixel_array
    pix_encoded = ds_encoded.pixel_array

    np.array_equal(pix_orig, pix_encoded), "pixel values did not match"


def helper_run_test(test_func):
    """
    helper function that runs transfer syntax tests on DS_01
    Args:
        test_func: function performs testing for specific transfer syntax

    Returns: None
    """

    # booleans to keep tract if we have test hologic and ge files


    files = get_test_files()
    non_testable_files = []

    # for each file in DS_01 run the test_function
    for file in files:
        output_dir = tempfile.mkdtemp(prefix="temp_output_folder_")
        try:
            test_func(file, output_folder=output_dir)

        # error from DCMTK converting or reading the file. This case the test should not fail and just skips this file
        except DCMTKError:
            non_testable_files.append(file)

        # clean the temp folder
        finally:
             shutil.rmtree(output_dir)
    assert len(non_testable_files) != len(files), "no testable files"


# start of tests
def test_T_207_1():
    """
    Test Explicit VR Little Endian
    """
    helper_run_test(helper_explicit_vr_little_enidan)


def test_T_207_2():
    """
    Test Implicit VR Little Endian
    """
    helper_run_test(helper_implicit_vr_little_endian)


def test_T_207_3():
    """
    Test Explicit VR Big Endian
    """
    helper_run_test(helper_explicit_vr_big_endian)


def test_T_207_4():
    """
    Test Deflated Explicit VR Little Endian
    """
    helper_run_test(helper_deflated_explicit_vr_little_endian)


def test_T_207_5():
    """
    Test RLE Lossless
    """
    helper_run_test(helper_rle_lossless)


def test_T_207_6():
    """
    Test JPEG lossless 14 sv1
    """
    helper_run_test(helper_jpeg_lossless_process_14_sv1)


def test_T_207_7():
    """
    Test JPEG ls lossless
    """

    helper_run_test(helper_jpeg_ls_lossless)


def test_T_207_8():
    """
    Test Jpeg 2000
    """

    helper_run_test(helper_j2k)

