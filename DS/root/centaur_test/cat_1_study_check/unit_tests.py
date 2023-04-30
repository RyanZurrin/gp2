import datetime
import logging
import traceback
import os
import random
import string
import tempfile
import warnings

import numpy as np
import pytest
import pydicom
import pandas as pd
import pydicom.uid
from dateutil.relativedelta import relativedelta
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset

from centaur_engine.preprocessor import Preprocessor
from centaur_deploy.deploys.config import Config
from centaur_test.config_factory import ConfigFactory
from deephealth_utils.data.checker import Checker
import deephealth_utils
import centaur_test.utils as utils
from centaur_test.data_manager import DataManager
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap

###################################################
# AUX FUNCTIONS
###################################################
@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    '''
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    '''
    record_xml_attribute("classname", "1_Unit")

def _get_random_dicom_value(dicom_data_element):
    '''
    Get a random value for a DICOM attribute based on its type
    :param dicom_data_element: DicomElement
    :return: random value
    '''
    multiplicity = dicom_data_element.VM
    representation = dicom_data_element.VR
    if multiplicity == 1:
        return __get_random_dicom_value_repr(representation)
    # Multiple
    val = []
    for i in range(multiplicity):
        val.append(__get_random_dicom_value_repr(representation))
    return val


def __get_random_dicom_value_repr(representation):
    '''Get a random value for different DICOM representations, as defined in
    http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html'''
    if representation == "UI":
        length = random.randint(1, 32) * 2  # Force even number <= 64
        return ''.join([random.choice(string.digits + '.') for n in range(length)])
    if representation == "SH":
        # Short string (max 16 chars)
        return ''.join([random.choice(string.ascii_letters + ' ') for n in range(random.randint(0, 16))])
    if representation == "LO":
        # Long string (max 64 chars)
        return ''.join([random.choice(string.ascii_letters + ' ') for n in range(random.randint(0, 64))])
    if representation == "ST":
        # Short text (max 1024 chars)
        return ''.join([random.choice(string.ascii_letters + string.whitespace + string.punctuation) for n in
                        range(random.randint(0, 1024))])
    if representation == "LT":
        # Long text (max 10240 chars)
        return ''.join([random.choice(string.ascii_letters + string.whitespace + string.punctuation) for n in
                        range(random.randint(0, 10240))])

    if representation == 'DS':
        # Decimal
        return "{:.2}".format(random.random())
    if representation == 'AS':
        # Age
        return "{:03}{}".format(random.randint(0, 110), random.choice(('D', 'W', 'M', 'Y')))
    if representation == 'DA':
        # Date
        start = datetime.datetime(1900, 1, 1)
        years = 150
        end = start + datetime.timedelta(days=365 * years)
        date_ = start + (end - start) * random.random()
        return date_.strftime("%Y%m%d")
    if representation == 'DT':
        # Datetime
        start = datetime.datetime(1900, 1, 1)
        years = 150
        end = start + datetime.timedelta(days=365 * years)
        date_ = start + (end - start) * random.random()
        return date_.strftime("%Y%m%d") + "{:02}{:02}{:02}".format(random.randint(0, 23), random.randint(0, 59),
                                                                   random.randint(0, 60))
    if representation == "IS":
        return str(random.randint(0, 10000))

    # Default: just random string
    return ''.join([random.choice(string.ascii_letters + ' ') for n in range(random.randint(0, 16))])


def _get_temp_file_path():
    return os.path.join(tempfile.gettempdir(), "0.dcm")


def _remove_temp_file():
    """
    If it exists, remove the temp file that is created for most of the tests
    """
    if os.path.exists(_get_temp_file_path()):
        os.remove(_get_temp_file_path())


def teardown_module(module):
    """
    Method invoked after the execution of any test.
    Just remove the temp file if it exists
    :param module:
    :return:
    """
    _remove_temp_file()


def _replace_dicom_attribute(dicom_ds, attributes, values, output_file_path=None):
    '''
    Replace the value in a DICOM attribute and save the new dataset in a temp file
    :param dicom_ds: pydicom dataset.
    :param attributes: list of str or str. Attribute to modify
    :param values: list of str or str. Value to set. If None, remove attribute
    :param output_file_path: str. Path to the new generated file. When None, use the default temp file
    :return: str. Path to the temp file that was generated

    '''
    if not isinstance(attributes, list):
        attributes = [attributes]
        values = [values]

    for attribute, value in zip(attributes, values):
        if value is None:
            if attribute in dicom_ds:
                del dicom_ds[attribute]
        else:
            setattr(dicom_ds, attribute, value)
    # Save file
    temp_file = _get_temp_file_path() if output_file_path is None else output_file_path
    dicom_ds.save_as(temp_file)
    return temp_file

def _get_config(skip_study_checks=True, process_pixel_data=False):
    """
    Generate a Config object that can be reused for most of the tests (Production config with no pixels reading)
    Returns:
        Config object
    """
    config_factory = ConfigFactory.VerificationInternalConfigFactory(optional_params_dict={
                                                                        'process_pixel_data': process_pixel_data,
                                                                        'skip_study_checks': skip_study_checks
                                                                        }
                                                                    )
    return Config(overriden_params=config_factory.params_dict)

def _test_attribute_exists(file_path, attribute, expected_flag):
    '''
    Test if a DICOM attribute exists in a file and making sure the required flags are raised
    :param file_path: str. Test file path
    :param attribute: str. Attribute tag/description
    :param expected_flag: str. Expected flag in case of a wrong value
    '''

    config = _get_config()

    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    ds = pydicom.dcmread(file_path)

    # Blank attribute (empty string)
    ds[attribute].value = ""
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    os.remove(temp_file)
    assert expected_flag in reasons, "{} flag expected for attribute '{}' missing. I got: {}".format(expected_flag,
                                                                                                     attribute, reasons)
    # Blank attribute (whitespace)
    ds[attribute].value = " "
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    os.remove(temp_file)
    assert expected_flag in reasons, "{} flag expected for attribute '{}' missing. I got: {}".format(expected_flag,
                                                                                                     attribute, reasons)

    # Remove attribute
    del ds[attribute]
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    os.remove(temp_file)
    assert expected_flag in reasons, "{} flag expected for attribute '{}' missing. I got: {}".format(expected_flag,
                                                                                                     attribute, reasons)


def _test_attribute_equals(file_path, attribute, valid_values, expected_flag, invalid_values=None, allow_missing=False):
    '''
    Make sure a certain DICOM file attribute has a valid value and making sure the required flags are raised.
    :param file_path: str. Test file path
    :param attribute: str.Attribute tag/description
    :param valid_values: list. List of valid values
    :param expected_flag: str. Expected flag in case of a wrong value
    :param invalid_values: list (optional). Any value that is not valid for the field. If no value is specified, try to generate
                        a random one based on the DICOM field type and also the blank value when allow_missing==false
    :param allow_missing: bool. Allow the attribute not to be present in the dataset. If allow_missing is None, the
                                'None' validation will be skipped
    '''

    config = _get_config()
    
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    ds = pydicom.dcmread(file_path)

    if allow_missing:
        valid_values.append(None)
    elif allow_missing == False:
        # Blanks/missing not allowed
        if invalid_values is None:
            invalid_values = [None]
        else:
            invalid_values.append(None)

    for valid_value in valid_values:

        ds = pydicom.dcmread(file_path)
        temp_file = _replace_dicom_attribute(ds, attribute, valid_value)
        passed = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        assert passed, "Unexpected failures when testing attribute '{}': {}. Valid value: {}".format(attribute, reasons,
                                                                                               valid_value)
    if invalid_values is None:
        if attribute not in ds:
            raise Exception(
                "Random invalid value cannot be generated because the attribute '{}' is not present in the file."
                " Please specify an invalid value".format(attribute))
        # Generate a random value that should be wrong
        invalid_values = [_get_random_dicom_value(ds[attribute])]

    for invalid_value in invalid_values:
        ds = pydicom.dcmread(file_path)
        # Ensure there is a flag for each one of the wrong values
        temp_file = _replace_dicom_attribute(ds, attribute, invalid_value)
        _ = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()

        assert expected_flag in reasons, "{} flag expected for attribute '{}'. Attribute value: {}. Actual flags: {}. ".format(
            expected_flag, attribute, invalid_value, reasons)


def _test_attribute_not_equals(file_path, attribute, wrong_value, expected_flag):
    '''
    Make sure a certain DICOM file attribute is NOT equal to a particular value
    :param file_path: str. Test file path
    :param attribute: str.Attribute tag/description
    :param wrong_value: str. Value to test
    :param expected_flag: str. Expected flag in case of a wrong value

    '''

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    ds = pydicom.dcmread(file_path)

    temp_file = _replace_dicom_attribute(ds, attribute, wrong_value)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert expected_flag in reasons, "Flag {} expected for attribute '{}' and value '{}'. Got: {}".format(
        expected_flag, attribute, wrong_value, reasons)

    # Lowercase should not pass either
    temp_file = _replace_dicom_attribute(ds, attribute, wrong_value.lower())
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert expected_flag in reasons, "Flag {} expected for attribute '{}' and value '{}'. Got: {}".format(
        expected_flag, attribute, wrong_value, reasons)

    # Uppercase should not pass either
    temp_file = _replace_dicom_attribute(ds, attribute, wrong_value.upper())
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert expected_flag in reasons, "Flag {} expected for attribute '{}' and value '{}'. Got: {}".format(
        expected_flag, attribute, wrong_value, reasons)

    # Any other value should pass
    random_value = _get_random_dicom_value(ds[attribute])
    temp_file = _replace_dicom_attribute(ds, attribute, random_value)
    passed = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert passed, "Unexpected flags for attribute '{}' and value '{}': {}".format(attribute, random_value, reasons)

    # Blank should pass
    temp_file = _replace_dicom_attribute(ds, attribute, "")
    passed = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert passed, "Unexpected flags: for attribute '{}' and empty value. Flags: {}".format(attribute, reasons)

    # Missing attribute should pass
    temp_file = _replace_dicom_attribute(ds, attribute, None)
    passed = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert passed, "Unexpected flags for attribute '{}' missing: {}".format(attribute, reasons)


def _test_attribute_not_contains(file_path, attribute, forbidden_words, expected_flag):
    '''
    Make sure a certain DICOM file attribute is NOT equal to a particular value
    :param file_path: str. Test file path
    :param attribute: str.Attribute tag/description
    :param wrong_value: str. Value to test
    :param expected_flag: str. Expected flag in case of a wrong value
    '''

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    ds = pydicom.dcmread(file_path)

    setattr(ds, attribute, "")
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    passed = pp.preprocess([temp_file])
    failed_checks = pp.get_failed_checks()
    assert passed, "Unexpected flag with empty value for attribute '{}': {}".format(attribute, failed_checks)

    for word in forbidden_words:
        # Random capitalization
        if random.random() < 0.33:
            w = word.lower()
        elif random.random() < 0.66:
            w = word.upper()
        else:
            # Replace one random letter
            p = random.randint(0, len(word) - 1)
            word = list(word)
            word[p] = word[p].upper()
            w = "".join(word)

        setattr(ds, attribute,
                "This is a random description that contains the word {0}, which is forbidden. I repeat, this is forbidden: {0}".format(
                    w))
        temp_file = _get_temp_file_path()
        ds.save_as(temp_file)
        _ = pp.preprocess([temp_file])
        failed_checks = pp.get_failed_checks()
        os.remove(temp_file)
        assert expected_flag in failed_checks, "Flag {} should have been raised when attribute '{}' contains the word '{}'". \
            format(expected_flag, attribute, w)


def _test_attribute_not_diff(file_path, attribute, valid_values, expected_flag, wrong_values=None):
    '''
    The attribute may or may not exist; if it exists it must have a valid value
    :param file_path: str. Test file path
    :param attribute: str.Attribute tag/description
    :param valid_values: list. List of valid values
    :param expected_flag: str. Expected flag in case of a wrong value
    :param wrong_values: list (optional). Any value that is not valid for the field. If no value is specified, try to generate
                        a random one based on the DICOM field type and also the blank value when allow_missing==false
    '''
    _test_attribute_equals(file_path, attribute, valid_values, expected_flag, invalid_values=wrong_values,
                           allow_missing=True)


def _test_array_dim(original_file, expected_flag, dim_to_test, expected_range):
    '''
    Test files with a size mismatch in the image and the header
    :param ds: dicom dataset
    :param expected_flag: str. Expected flag error
    :param dim_to_test: dimension of pixel array to perturb (eg. Rows/Columns)
    '''
    # Range for rows and columns
    fields_to_test = ['Rows', 'Columns']

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Create an artificial image with a smaller/larger than allowed number of rows or columns
    for range_end in [0, 1]:  # Bottom end of range or top end
        ds = pydicom.dcmread(original_file)
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            ds.decompress()
        data = ds.pixel_array
        dtype = data.dtype
        s = expected_range[range_end] - 1 * (-1) ** range_end  # Math to subtract 1 if at bottom end, else add 1
        new_shape = list(data.shape)
        new_shape[dim_to_test] = s
        new_data = np.zeros(new_shape, dtype=dtype)
        ds.PixelData = new_data.tobytes()
        ds[fields_to_test[dim_to_test]].value = s
        temp_file = _get_temp_file_path()
        ds.save_as(temp_file)
        _ = pp.preprocess([temp_file])
        failed_checks = pp.get_failed_checks()
        assert expected_flag in failed_checks, "{} flag was expected for shape {}. Got: {}".format(
            expected_flag, new_shape, failed_checks)

###################################################
# General (aux) tests
###################################################

def test_baseline_studies(run_mode=None):
    '''
    Test that a study that should pass all the checks, actually does
    :param dataset_name: str. Dataset version (fixture defined in conftest.py file)
    '''
    # Read all the DICOM files from the baseline study
    data_manager = DataManager()
    config = _get_config(skip_study_checks=False, process_pixel_data=True)

    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())
    valid_studies = data_manager.get_valid_studies(run_mode=run_mode)
    for study in valid_studies:
        files = data_manager.get_dicom_images(study)

        # Make sure the study passes the tests with no changes
        passed = pp.preprocess(files)
        failed_checks = pp.get_failed_checks()
        assert passed, "Study {} failed for unexpected reasons: {}".format(study, failed_checks)

    # Get studies that shouldn't pass the Checker
    invalid_studies = data_manager.get_invalid_studies(run_mode=run_mode)
    for study in invalid_studies:
        files = data_manager.get_dicom_images(study)

        # Make sure the study passes the tests with no changes
        passed = pp.preprocess(files)
        assert not passed, f"Study {study} was expected to fail the Checker"


###################################################
# File Acceptance Criteria
###################################################

def test_T_137():
    '''
    Ensure some basic DICOM attributes are present.
    '''
    data_manager = DataManager()
    file_path = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)[0]
    _test_attribute_exists(file_path, "SOPInstanceUID", "FAC-200")
    _test_attribute_exists(file_path, "SeriesInstanceUID", "FAC-200")
    _test_attribute_exists(file_path, "StudyInstanceUID", "FAC-200")
    _test_attribute_exists(file_path, "PatientID", "FAC-200")
    _test_attribute_exists(file_path, "StudyDate", "FAC-200")
    _test_attribute_exists(file_path, "PatientBirthDate", "FAC-200")
    _test_attribute_exists(file_path, "Manufacturer", "FAC-80")
    _test_attribute_exists(file_path, "ManufacturerModelName", "FAC-90")


def test_T_138():
    '''
    Ensure SOPClassUID is valid and SOPClassUID-ImageType combination is allowed.
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())
    ds = pydicom.dcmread(files[0])
    temp_file = _get_temp_file_path()

    # Wrong SOPClassUID
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2.1"
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert "FAC-20" in reasons, "FAC-20 flag expected. Got: {}".format(reasons)

    #### HOLOGIC 2D
    allowed_combinations = (
        # DXm
        ("1.2.840.10008.5.1.4.1.1.1.2", ['DERIVED', 'PRIMARY']),
        ("1.2.840.10008.5.1.4.1.1.1.2", ['DERIVED', 'SECONDARY']),
        # Hologic I2D
        ("1.2.840.10008.5.1.4.1.1.1.2", ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D'])
        # SC
        # ("1.2.840.10008.5.1.4.1.1.7", ['DERIVED', 'PRIMARY'])
    )

    # Check all the valid combinations
    for sop_class_uid, image_type in allowed_combinations:
        ds.SOPClassUID = sop_class_uid
        ds.ImageType = image_type
        ds.save_as(temp_file)
        passed = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        assert passed, "Unexpected flag for combination {}-{}: {}".format(sop_class_uid, image_type, reasons)

    # Check some not valid combinations
    wrong_combinations = (
        ("1.2.840.10008.5.1.4.1.1.1.2", ['PRIMARY', 'SECONDARY']),
        ("1.2.840.10008.5.1.4.1.1.1.2", []),
        ("1.2.840.10008.5.1.4.1.1.1.2", 'XXX'),
        ("1.2.840.10008.5.1.4.1.1.1.2", ['ORIGINAL', 'PRIMARY', 'STEREO_SCOUT']),
        ("1.2.840.10008.5.1.4.1.1.7", ['DERIVED', 'SECONDARY']),
    )
    for sop_class_uid, image_type in wrong_combinations:
        ds.SOPClassUID = sop_class_uid
        ds.ImageType = image_type
        ds.save_as(temp_file)
        _ = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        assert "HOL-10" in reasons, "HOL-10 flag expected. Combination used: {}-{}. Got: {}".format(sop_class_uid,
                                                                                                    image_type, reasons)
    ## END HOLOGIC 2D

    ###### Hologic study - BT
    # files = data_manager.download_baseline_study(dataset_name, mask_files="BT*.dcm", study_key=DataManager.STUDY_03_HOLOGIC_BT)
    #                                    # max_num_files=1)
    files = data_manager.get_dicom_images(DataManager.STUDY_03_DBT_HOLOGIC)
    ds = pydicom.dcmread(files[0])
    allowed_combinations = (
        ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY']),
        ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY', 'VOLUME', 'NONE']),
        # I2D
        ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D']),
        # Slabs
        ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'MEAN'])
    )

    # Check all the valid combinations
    for sop_class_uid, image_type in allowed_combinations:
        ds.SOPClassUID = sop_class_uid
        ds.ImageType = image_type
        ds.save_as(temp_file)
        passed = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        assert passed, "Unexpected flag for combination {}-{}: {}".format(sop_class_uid, image_type, reasons)

    # Check some not valid combinations
    wrong_combinations = (
        ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'SECONDARY']),
        ("1.2.840.10008.5.1.4.1.1.13.1.3", ['PRIMARY', 'DERIVED']),
        ("1.2.840.10008.5.1.4.1.1.13.1.3", []),
        ("1.2.840.10008.5.1.4.1.1.13.1.3", 'XXX'),
        ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY', 'TOMO_SCOUT']),
    )
    for sop_class_uid, image_type in wrong_combinations:
        ds.SOPClassUID = sop_class_uid
        ds.ImageType = image_type
        ds.save_as(temp_file)
        _ = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        assert "HOL-10" in reasons, "HOL-10 flag expected. Combination used: {}-{}. Got: {}".format(
            sop_class_uid, image_type, reasons)
    # END Hologic study - BT

    ###### Hologic study - BT - SC
    # files = data_manager.download_baseline_study(dataset_name, mask_files="SC*.dcm",
    #                                    study_key=DataManager.STUDY_04_HOLOGIC_BT_SC, max_num_files=1)
    # ds = pydicom.dcmread(files[0])
    # allowed_combinations = (
    #     ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY']),
    # )
    #
    # # Check all the valid combinations
    # for sop_class_uid, image_type in allowed_combinations:
    #     ds.SOPClassUID = sop_class_uid
    #     ds.ImageType = image_type
    #     ds.save_as(temp_file)
    #     passed = pp.preprocess([temp_file])
    #     reasons = pp.get_failed_checks()
    #     assert passed, "Unexpected flag for combination {}-{}: {}".format(sop_class_uid, image_type, reasons)
    #
    # # Check some not valid combinations
    # wrong_combinations = (
    #     ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'SECONDARY']),
    #     ("1.2.840.10008.5.1.4.1.1.13.1.3", ['PRIMARY', 'DERIVED']),
    #     ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY', 'TOMO_SCOUT']),
    # )
    # for sop_class_uid, image_type in wrong_combinations:
    #     ds.SOPClassUID = sop_class_uid
    #     ds.ImageType = image_type
    #     ds.save_as(temp_file)
    #     passed = pp.preprocess([temp_file])
    #     reasons = pp.get_failed_checks()
    #     assert "HOL-10" in reasons, "HOL-10 flag expected. Combination used: {}-{}. Got: {}".format(
    #         sop_class_uid, image_type, reasons)
    # # END Hologic study - BT - SC

    ###### GE study - DXm
    # files = data_manager.get_dicom_images(DataManager.STUDY_02_GE)
    # # files = data_manager.download_baseline_study(dataset_name, mask_files="*.dcm", study_key=DataManager.STUDY_02_GE,
    # #                                              max_num_files=1)
    # ds = pydicom.dcmread(files[0])
    # allowed_combinations = (
    #     # DXm
    #     ("1.2.840.10008.5.1.4.1.1.1.2", ['ORIGINAL', 'PRIMARY', '']),
    #     # BT
    #     # ("1.2.840.10008.5.1.4.1.1.13.1.3", ['ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE']),
    #     # ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY', 'TOMOSYNTHESIS', 'GENERATED_2D']),
    #     # ("1.2.840.10008.5.1.4.1.1.13.1.3", ['DERIVED', 'PRIMARY', 'VOLUME', 'NONE'])
    # )
    #
    # # Check all the valid combinations
    # for sop_class_uid, image_type in allowed_combinations:
    #     ds.SOPClassUID = sop_class_uid
    #     ds.ImageType = image_type
    #     ds.save_as(temp_file)
    #     passed = pp.preprocess([temp_file])
    #     reasons = pp.get_failed_checks()
    #     assert passed, "Unexpected flag for combination {}-{}: {}".format(sop_class_uid, image_type, reasons)
    #
    # # Check some not valid combinations
    # wrong_combinations = (
    #     ("1.2.840.10008.5.1.4.1.1.1.2", ['PRIMARY', 'SECONDARY']),
    #     ("1.2.840.10008.5.1.4.1.1.1.2", []),
    #     ("1.2.840.10008.5.1.4.1.1.1.2", 'XXX'),
    #     ("1.2.840.10008.5.1.4.1.1.1.2", ['XXX']),
    # )
    # for sop_class_uid, image_type in wrong_combinations:
    #     ds.SOPClassUID = sop_class_uid
    #     ds.ImageType = image_type
    #     ds.save_as(temp_file)
    #     passed = pp.preprocess([temp_file])
    #     reasons = pp.get_failed_checks()
    #     assert "GE-10" in reasons, "GE-10 flag expected. Combination used: {}-{}. Got: {}".format(sop_class_uid,
    #                                                                                               image_type, reasons)
    ###### END GE study - DXm


def test_T_139():
    '''
    Ensure there is a valid Modality in DICOM header files.
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_equals(files[0], 'Modality', ["MG"], "FAC-21", invalid_values=["MM", ""])


def test_T_140():
    '''
    Ensure there is a valid Body Part Examined in DICOM header files
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_equals(files[0], 'BodyPartExamined', ["BREAST"], "FAC-23", invalid_values=["Breas", ""])


def test_T_141():
    '''
    Ensure there is not Loss Image Compression
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_equals(files[0], 'LossyImageCompression', ["00"], "FAC-24")


def test_T_142():
    '''
    Ensure the images are not Quality Control images
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_not_equals(files[0], 'QualityControlImage', 'YES', 'FAC-26')


def test_T_143():
    '''
    Ensure the image does not contain sufficient burned in annotation to identify the patient and date the image was acquired
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_equals(files[0], 'BurnedInAnnotation', ['NO'], 'FAC-27')

def test_T_144():
    '''
    Ensure the images have been acquired using a supported manufacturer scan.
    '''
    # Download the first DICOM image
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Use baseline studies and make sure they pass
    #commented out I believe GE studies are not suppose to pass checker anymore
    #files.append(data_manager.get_dicom_images(DataManager.STUDY_02_GE)[0])

    for f in files:
        passed = pp.preprocess([f])
        reasons = pp.get_failed_checks()
        assert passed, "Unexpected flag/s ({}) in file {}".format(reasons, f)

    # Missing manufacturer
    ds = pydicom.dcmread(files[0])
    temp_file = _replace_dicom_attribute(ds, "Manufacturer", None)
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert 'FAC-80' in reasons, "{} expected. Got: {}".format('FAC-80', reasons)

    # # # Missing Manufacturer model
    # ds = pydicom.dcmread(files[0])
    # temp_file = _replace_dicom_attribute(ds, "ManufacturerModelName", None)
    # ds.save_as(temp_file)
    # passed = pp.preprocess([temp_file])
    # reasons = pp.get_failed_checks()
    # assert 'FAC-90' in reasons, "FAC-90 expected. Got: {}".format(reasons)

    # Wrong manufacturer
    ds = pydicom.dcmread(files[0])
    temp_file = _replace_dicom_attribute(ds, "Manufacturer", "Siemens")
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert 'FAC-140' in reasons, "FAC-140 expected. Got: {}".format(reasons)
    #
    # # # Wrong Manufacturer model.
    # ds = pydicom.dcmread(files[0])
    # temp_file = _replace_dicom_attribute(ds, "ManufacturerModelName", "MAMMOMAT")
    # ds.save_as(temp_file)
    # passed = pp.preprocess([temp_file])
    # reasons = pp.get_failed_checks()
    # assert 'FAC-141' in reasons, "FAC-141 expected. Got: {}".format(reasons)


def test_T_145():
    '''
    Ensure the DICOM file contains pixel data.
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    config = Config()
    config[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
    config[Config.MODULE_ENGINE, 'skip_study_checks'] = True
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    ds = pydicom.dcmread(files[0])
    temp_file = _replace_dicom_attribute(ds, "PixelData", None)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert 'FAC-100' in reasons, "FAC-100 flag expected. Got: {}".format(reasons)


def test_T_146():
    '''
    Ensure the image type is 'For Presentation'
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_equals(files[0], 'PresentationIntentType', ['FOR PRESENTATION'], 'FAC-110', ['For Processing', 'random']
                           , allow_missing=True)


def test_T_147():
    '''
    Ensure the code image sequence is valid (only one element in the sequence and no modifiers applied).
    '''

    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    ds = pydicom.dcmread(files[0])

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Flag when ViewCodeSequence does not exist
    del ds['ViewCodeSequence']
    # Save file
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert 'FAC-160' in reasons, 'ViewCodeSequence not existing'

    # Flag when ViewCodeSequence has a length > 1
    ds = pydicom.dcmread(files[0])
    first = ds['ViewCodeSequence'][0]
    ds['ViewCodeSequence'].value.append(first)
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert 'FAC-160' in reasons, 'ViewCodeSequence length > 1 is not allowed'

    # Valid code values
    ds = pydicom.dcmread(files[0])
    for code in ('R-10226', 'R-10242'):
        ds['ViewCodeSequence'][0].CodeValue = code
        temp_file = _get_temp_file_path()
        ds.save_as(temp_file)
        passed = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        assert passed, "Unexpected failure for code '{}': {}".format(code, reasons)

    # Invalid codes
    for code in ('R-1540224', ''):
        ds['ViewCodeSequence'][0].CodeValue = code
        temp_file = _get_temp_file_path()
        ds.save_as(temp_file)
        _ = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        assert 'FAC-160' in reasons, "Flag 160 expected. Code: '{}'. Flags: {}".format(code, reasons)

    # Remove CodeValue
    del ds['ViewCodeSequence'][0]['CodeValue']
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert 'FAC-160' in reasons, "Flag 160 expected. Code: '{}'. Flags: {}".format(code, reasons)

    ### ViewModifierCodeSequence

    # Remove element
    ds = pydicom.dcmread(files[0])
    del ds['ViewCodeSequence'][0]['ViewModifierCodeSequence']
    ds.save_as(temp_file)
    passed = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert passed, "Unexpected failure when ViewModifierCodeSequence : {}".format(code, reasons)

    # Empty list
    ds = pydicom.dcmread(files[0])
    ds['ViewCodeSequence'][0]['ViewModifierCodeSequence'].value = []
    ds.save_as(temp_file)
    passed = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert passed, "Unexpected failure when ViewModifierCodeSequence : {}".format(code, reasons)

    # Adding some fake element (flag)
    ds = pydicom.dcmread(files[0])
    dummy_ds = Dataset()
    dummy_ds.FakeValue = "FAKE"
    seq = Sequence([dummy_ds])
    ds['ViewCodeSequence'][0].ViewModifierCodeSequence = seq
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    reasons = pp.get_failed_checks()
    assert 'FAC-170' in reasons, "Flag 170 expected. Code: '{}'. Flags: {}".format(code, reasons)


def test_T_148():
    '''
    Ensure the image does not represent a partial view.
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_not_equals(files[0], 'PartialView', "YES", "FAC-181")


def test_T_149():
    '''
    Ensure the intensity values of the pixels are correct.
    The min value should be 0.
    The max value should be at least XX, corresponding to 75% of the max value found empirically on data.
    '''

    # Download only first image
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    config = Config()
    config[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
    config[Config.MODULE_ENGINE, 'skip_study_checks'] = True
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    f = files[0]
    ds = pydicom.dcmread(f)

    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()

    # Test min value
    # MIN_VALUE = 2 ** (ds.HighBit - 1)  # Threshold value used as a reference
    # new_data = np.clip(ds.pixel_array, -1, None)  # Make sure the min value is bigger than MIN
    # new_data = ds.pixel_array
    # new_data = np.ones(new_data.shape) * -1
    # ds.PixelData = new_data.tobytes()
    # ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    # temp_file = _get_temp_file_path()
    # ds.save_as(temp_file)
    # pdb.set_trace()
    # passed = pp.preprocess([temp_file])
    # failed_checks = pp.get_failed_checks()
    # assert "IAC-20" in failed_checks, "IAC-20 flag was expected. Got: {}".format(failed_checks)

    # Test max value
    MIN_MAX_VALUE = (2 ** (ds.HighBit + 1) - 1) * 2 / 3  # Threshold value used as a reference
    new_data = np.clip(ds.pixel_array, 0, int(MIN_MAX_VALUE) - 1)  # Make sure max < X
    ds.PixelData = new_data.tobytes()
    temp_file = _get_temp_file_path()
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    failed_checks = pp.get_failed_checks()
    assert "IAC-20" in failed_checks, "IAC-20 flag was expected. Got: {}".format(failed_checks)

# def test_T_1_U15():
#     '''
#     Ensure the image orientation looks as expected in MLO images
#     '''
#     data_manager = DataManager()
#
#     config = Config.ProductionConfig()
#     config[Config.MODULE_ENGINE, 'skip_study_checks'] = True
#     pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())
#
#     # Rotate the images
#     temp_dir = tempfile.mkdtemp()
#     try:
#         # DBT study
#         new_files = []  # Preprocess rotated images only
#         files = data_manager.get_dicom_images(DataManager.STUDY_03_DBT_HOLOGIC)
#         for i in range(len(files)):
#             ds = dhutils.dh_dcmread(files[i])
#             view = ds.dh_getattribute('ViewPosition')
#             if view != "MLO":
#                 # Only MLO images are problematic
#                 continue
#             if ds.file_meta.TransferSyntaxUID.is_compressed:
#                 ds.decompress()
#             pixel_array = ds.pixel_array
#             if pixel_array.ndim == 2:
#                 pixel_array = np.rot90(ds.pixel_array, 2)
#             elif pixel_array.ndim == 3:
#                 pixel_array = np.rot90(ds.pixel_array, 2, (1, 2))
#             else:
#                 raise Exception("Unknown image format. Number of dimensions: {}".format(pixel_array.ndim))
#             ds.PixelData = pixel_array.tobytes()
#             p = os.path.join(temp_dir, "{}.dcm".format(i))
#             ds.save_as(p)
#             new_files.append(p)
#         _ = pp.preprocess(new_files)
#         failed_checks = pp.get_failed_checks()
#         assert 'IAC-30' in failed_checks, "Image was rotated and not flagged properly. Flags: {}".format(failed_checks)
#     finally:
#         shutil.rmtree(temp_dir)

#     f = files[0]
#     ds = pydicom.dcmread(f)
#     data = ds.pixel_array
#
#     # Make sure that initially all the checks pass
#     assert data.shape[0] == ds.Rows and data.shape[1] == ds.Columns, \
#         "Unexpected image size. Image array shape: {}; Rows: {}; Columns: {}".format(data.shape, ds.Rows, ds.Columns)
#
#     # Remove the Rows attribute
#     del ds['Rows']
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     failed_checks = pp.get_failed_checks()
#     assert 'IAC-10' in failed_checks, 'IAC-10 should have been raised. Failed checks: {}'.format(failed_checks)
#
#     # Remove the Columns attribute
#     ds = pydicom.dcmread(f)
#     del ds['Columns']
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     failed_checks = pp.get_failed_checks()
#     assert 'IAC-10' in failed_checks, 'IAC-10 should have been raised. Failed checks: {}'.format(failed_checks)
#
#     # Set the 'Rows' attribute to 0
#     ds = pydicom.dcmread(f)
#     ds['Rows'].value = 0
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     failed_checks = pp.get_failed_checks()
#     assert 'IAC-10' in failed_checks, 'IAC-10 should have been raised. Failed checks: {}'.format(failed_checks)
#
#     # Set the 'Columns' attribute to 0
#     ds = pydicom.dcmread(f)
#     ds['Columns'].value = 0
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     failed_checks = pp.get_failed_checks()
#     assert 'IAC-10' in failed_checks, 'IAC-10 should have been raised. Failed checks: {}'.format(failed_checks)
#
#     # Set the 'Rows' attribute to a wrong value
#     ds = pydicom.dcmread(f)
#     ds['Rows'].value = data.shape[0] + 1
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     failed_checks = pp.get_failed_checks()
#     assert 'IAC-10' in failed_checks, 'IAC-10 should have been raised. Failed checks: {}'.format(failed_checks)
#
#     # The code below would not fail because of the way pydicom read the pixel data (basically ignoring this attribute)
#     # Set the 'Columns' attribute to a wrong value
#     # ds = pydicom.dcmread(f)
#     # ds['Columns'].value = data.shape[1] + 1
#     # ds.save_as(temp_file)
#     # passed = pp.preprocess([temp_file])
#     # failed_checks = pp.get_failed_checks()
#     # assert 'IAC-10' in failed_checks, 'IAC-10 should have been raised. Failed checks: {}'.format(failed_checks)

# def test_T_1_U16():
#     '''
#     Ensure the image corresponds to a screening mammography looking for "forbidden" words in the StudyDescription field.
#     '''
#     # Download/read all the DICOM files from baseline study
#     data_manager = DataManager()
#     files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
#
#     config = Config.ProductionConfig()
#     config[Config.MODULE_ENGINE, 'process_pixel_data'] = False
#     pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())
#
#     ### Update the description of one of the files to include the list of forbidden words (FAC-180)
#     num_files = len(files)
#     for word in (
#             'dx',
#             'diag',
#             'non screen',
#             'non-screen',
#             'bx',
#             'biopsy',
#             'inject',
#             'localization',
#             'wire',
#             'marker',
#             'mag',
#             'magnification',
#             'loc',
#             'compression',
#             'galactogram',
#             'biop',
#             'needle',
#             'specimen',
#     ):
#         # Random capitalization
#         new_files = files.copy()
#         i = random.randint(0, num_files - 1)
#         ds = pydicom.dcmread(files[i])
#         if random.random() < 0.33:
#             w = word.lower()
#         elif random.random() < 0.66:
#             w = word.upper()
#         else:
#             # Replace one letter
#             p = random.randint(0, len(word) - 1)
#             word = list(word)
#             word[p] = word[p].upper()
#             w = "".join(word)
#
#         ds['StudyDescription'].value = "This is a random description that contains a forbidden attribute: {}".format(w)
#         temp_file = _get_temp_file_path()
#         ds.save_as(temp_file)
#         new_files[i] = temp_file
#         passed = pp.preprocess(new_files)
#         failed_checks = pp.get_failed_checks()
#         os.remove(temp_file)
#         assert not passed, "Checks passed when the study description contains the following word: {}".format(w)
#         assert 'FAC-180' in failed_checks, 'Expected FAC-180 flag. I got {}'.format(failed_checks)
#
#         # Uppercase
#         new_files = files.copy()
#         i = random.randint(0, num_files - 1)
#         ds = pydicom.dcmread(files[i])
#         w = w.upper()
#         ds['StudyDescription'].value = "This is a random description that contains a forbidden attribute: {}".format(w)
#         ds.save_as(temp_file)
#         new_files[i] = temp_file
#         passed = pp.preprocess(new_files)
#         failed_checks = pp.get_failed_checks()
#         os.remove(temp_file)
#         assert not passed, "Checks passed when the study description contains the following word: {}".format(w)
#         assert 'FAC-180' in failed_checks, 'Expected FAC-180 flag. I got {}'.format(failed_checks)
#
#         # Random capitalization
#         new_files = files.copy()
#         i = random.randint(0, num_files - 1)
#         ds = pydicom.dcmread(files[i])
#         c = random.randint(0, len(word) - 1)
#         w = w.lower().replace(w[c], w[c].upper())
#         ds['StudyDescription'].value = "This is a random description that contains a forbidden attribute: {}".format(w)
#         ds.save_as(temp_file)
#         new_files[i] = temp_file
#         passed = pp.preprocess(new_files)
#         failed_checks = pp.get_failed_checks()
#         os.remove(temp_file)
#         assert 'FAC-180' in failed_checks, 'Expected FAC-180 flag when the study description contains the following"' \
#                                            ' word: {}. I got {}'.format(word, failed_checks)
#

def test_T_150():
    '''
    Ensure the images have valid dimensions compliant with the manufacturer conformance docs.
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_array_dim(files[0], 'HOL-140', 0, [1750, 5000])
    _test_array_dim(files[0], 'HOL-150', 1, [1750, 5000])

    files = data_manager.get_dicom_images(DataManager.STUDY_02_GE)
    _test_array_dim(files[0], 'GE-140', 0, [1750, 5000])
    _test_array_dim(files[0], 'GE-150', 1, [1750, 5000])


###################################################
# Patient Acceptance Criteria
###################################################

def test_T_151():
    '''
    Ensure the patient has not breast implants.
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    _test_attribute_not_equals(files[0], "BreastImplantPresent", 'YES', 'PAC-30')


def test_T_152():
    '''
    Ensure the patient is at least 35 years old.
    '''
    # Download the whole study so that there are not study errors
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    config = _get_config(skip_study_checks=False, process_pixel_data=True)
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Modify the first file in different ways
    original_file = files[0]
    temp_file = _get_temp_file_path()

    # Modify the PatientAge and remove PatientBirthDate and StudyDate
    ds = pydicom.dcmread(original_file)
    ds['PatientAge'].value = '030Y'
    # del ds['PatientBirthDate']
    # del ds['StudyDate']
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])

    failed_checks = pp.get_failed_checks()
    assert 'PAC-10' in failed_checks, 'PAC-10 should have been raised. Failed checks: {}'.format(failed_checks)

    # Remove Patient Age and modify the birth date to an edge case (35 years minus 1 day) from StudyDate
    ds = pydicom.dcmread(original_file)
    wrong_date = datetime.datetime.strptime(ds.StudyDate, '%Y%m%d') - relativedelta(years=35, days=-1)
    ds['PatientBirthDate'].value = wrong_date.strftime("%Y%m%d")
    del ds['PatientAge']
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    failed_checks = pp.get_failed_checks()
    assert 'PAC-10' in failed_checks, 'PAC-10 should have been raised. Failed checks: {}'.format(failed_checks)

    # Keep a correct patient birth date but modify the patient age (inconsistent value)
    # This case should be flagged because PatientAge has priority
    ds = pydicom.dcmread(original_file)
    ds['PatientAge'].value = '039Y'
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    failed_checks = pp.get_failed_checks()
    assert 'PAC-10' in failed_checks, 'PAC-10 should have been raised. Failed checks: {}'.format(failed_checks)

    # Modify the birth date to an incorrect edge case but keep a correct patient age
    # The expected behavior is to pass the validation
    ds = pydicom.dcmread(original_file)
    wrong_date = datetime.datetime.today() - relativedelta(years=35, days=1)
    ds['PatientBirthDate'].value = wrong_date.strftime("%Y%m%d")
    ds.save_as(temp_file)
    _ = pp.preprocess([temp_file])
    failed_checks = pp.get_failed_checks()
    assert 'PAC-10' in failed_checks, 'PAC-10 should have been raised. Failed checks: {}'.format(failed_checks)


###################################################
# Study Acceptance Criteria
###################################################

def test_T_153():
    '''
    Ensure individual files and a study total size do not go over the max size allowed.
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    temp_file = _get_temp_file_path()
    ####### Individual file ########
    MAX_FILE_SIZE = 1.5 * 2 ** 30  # Max file size in bytes (1.5GB)
    ds = pydicom.dcmread(files[0])
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    data = ds.pixel_array
    dtype = data.dtype
    s = int(np.ceil(np.sqrt(MAX_FILE_SIZE)))
    new_data = np.zeros((s, s), dtype=dtype)
    ds.PixelData = new_data.tobytes()
    ds['Rows'].value = s
    ds['Columns'].value = s
    ds.save_as(temp_file)
    passed = pp.preprocess([temp_file])
    failed_checks = pp.get_failed_checks()
    os.remove(temp_file)
    assert not passed, "File size too big was expected"

    ####### Study ########
    pp._config['check_study'] = True
    MAX_STUDY_SIZE = 10 * (2 ** 30)  # Max study size: 10 GB
    MAX_ROWS_COLUMNS = 5000  # Max number of rows/files to be created for each image to generate a study > 10GB

    ds = pydicom.dcmread(files[0])
    data = ds.pixel_array
    dtype = data.dtype
    new_data = np.ones((MAX_ROWS_COLUMNS, MAX_ROWS_COLUMNS), dtype=dtype)
    ds.PixelData = new_data.tobytes()
    ds['Rows'].value = MAX_ROWS_COLUMNS
    ds['Columns'].value = MAX_ROWS_COLUMNS
    ds.save_as(temp_file)
    num_bytes = ds.BitsAllocated / 8
    total_file_size = (MAX_ROWS_COLUMNS ** 2) * num_bytes
    num_files_needed = np.math.ceil(MAX_STUDY_SIZE / total_file_size)
    _ = pp.preprocess([temp_file] * num_files_needed)
    failed_checks = pp.get_failed_checks()
    assert 'SAC-120' in failed_checks, 'SAC-120 expected (study too big). Got: {}'.format(failed_checks)


def test_T_154():
    '''
    Ensure main DICOM fields are consistent along all the images in a study ('StudyInstanceUID', 'PatientName', etc).
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    config = _get_config(skip_study_checks=False, process_pixel_data=True)
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Break the study. Change the value of one of the tested attributes in a random file
    num_files = len(files)
    val = "unknown"
    # Test the fields where we can just use random values
    for attr, flag in (
            ('PatientName', 'SAC-70'),
            # ('PatientBirthDate', 'SAC-80'),
            # ('PatientAge', 'SAC-81'),
            ('PatientID', 'SAC-90'),
            ('StudyInstanceUID', 'SAC-100'),
            ('StudyDate', 'SAC-110')):
            # ('Manufacturer', 'SAC-140')):
        try:
            print("Processing {}".format(attr))
            new_files = files.copy()
            i = random.randint(0, num_files - 1)
            ds = pydicom.dcmread(files[i])
            val = _get_random_dicom_value(ds[attr])
            ds[attr].value = val
            temp_file = _get_temp_file_path()
            ds.save_as(temp_file)
            new_files[i] = temp_file
            _ = pp.preprocess(new_files)
            reasons = pp.get_failed_checks()
            assert flag in reasons, "Flag '{}' expected. The attribute '{}' was modified in one of the " \
                                    "study files (value='{}') but the study was not flagged".format(flag, attr, val)
        except Exception as ex:
            print("ERROR in file '{}', attr='{}', value='{}'. {}".format(files[i], attr, val, ex))
            raise

    # Test the fields where we need to use valid values
    for attr, value, flag in (
            ('PatientBirthDate', datetime.date(1960, 1, 1), 'SAC-80'),
            ('PatientAge', '060Y', 'SAC-81'),
            ('Manufacturer', 'GE', 'SAC-140')
            ):
        try:
            if attr == "Manufacturer":
                warnings.warn("Manufacturer cannot be tested because at this point only Hologic is accepted")
                continue
            print("Processing {}".format(attr))
            new_files = files.copy()
            i = random.randint(0, num_files - 1)
            ds = pydicom.dcmread(files[i])
            setattr(ds, attr, value)
            temp_file = _get_temp_file_path()
            ds.save_as(temp_file)
            new_files[i] = temp_file
            _ = pp.preprocess(new_files)
            reasons = pp.get_failed_checks()
            assert flag in reasons, "Flag '{}' expected. The attribute '{}' was modified in one of the " \
                                    "study files (value='{}') but the study was not flagged".format(flag, attr,
                                                                                                    val)
        except Exception as ex:
            print("ERROR in file '{}', attr='{}', value='{}'. {}".format(files[i], attr, val, ex))
            raise


def test_T_155():
    '''
    Ensure a study contains the 4 images-lateralities (RMLO, RCC, LMLO, LCC).
    '''

    # Download all the DICOM images for the baseline study (2D)
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
    assert len(files) == 4, "The study should contain exactly 4 images. Got: {}".format(len(files))

    config = _get_config(skip_study_checks=False, process_pixel_data=True)
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Remove one image each time (in a random order)
    random.shuffle(files)
    for i in range(4):
        l = files.copy()
        # Read the laterality and position of the image is going to be removed from the study
        ds = pydicom.dcmread(files[i])
        laterality = ds.ImageLaterality
        position = ds.ViewPosition
        code = laterality + position
        l.remove(files[i])
        _ = pp.preprocess(l)
        failed_checks = pp.get_failed_checks()
        if code == 'LCC':
            expected_fail = 'SAC-50'
        elif code == 'RCC':
            expected_fail = 'SAC-60'
        elif code == 'LMLO':
            expected_fail = 'SAC-30'
        elif code == 'RMLO':
            expected_fail = 'SAC-40'
        else:
            raise Exception("Incorrect laterality/position")
        assert expected_fail in failed_checks, 'Expected flag ' + expected_fail

    # Remove two images and check that we get two 'SAC' errors
    # l = files[:2]
    # passed = pp.preprocess(l)
    # failed_checks = pp.get_failed_checks()
    #
    # assert len(failed_checks) >= 2, \
    #     'Expected two fails in the range SAC-30 - SAC-60. Got: "{}"'.format(failed_checks)
    #
    # for fail in failed_checks:
    #     assert fail in ('SAC-30', 'SAC-40', 'SAC-50', 'SAC-60'), \
    #         "Unexpected fail: '{}'. Total failures: '{}'".format(fail, failed_checks)
    #
    # # Introduce an "extra" image with a non valid view
    # ds = pydicom.dcmread(files[0])
    # ds.ViewPosition = 'XXX'
    # temp_file = _get_temp_file_path()
    # ds.save_as(temp_file)
    # passed = pp.preprocess(files + [temp_file])
    # failed_checks = pp.get_failed_checks()
    # assert 'SAC-20' in failed_checks, 'SAC-20 flag expected when a non-valid view is introduced (besides the valid ones)'

    # # Introduce an "extra" image with a valid view
    # ds = pydicom.dcmread(files[0])
    # temp_file = _get_temp_file_path()
    # ds.save_as(temp_file)
    # passed = pp.preprocess(files + [temp_file])
    # failed_checks = pp.get_failed_checks()
    # assert 'SAC-20' in failed_checks, 'SAC-20 flag expected when an extra "valid" view is introduced'


def test_T_156():
    '''
    Ensure the image transfer syntax method belongs to the list of formats that pydicom (with gdcm) can handle and
    are not lossy.
    For more infor about the supported formats check https://pydicom.github.io/pydicom/stable/image_data_handlers.html
    '''
    data_manager = DataManager()
    files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

    allowed_transfer_syntaxes = (
        pydicom.uid.ExplicitVRLittleEndian,
        pydicom.uid.ImplicitVRLittleEndian,
        pydicom.uid.ExplicitVRBigEndian,
        pydicom.uid.DeflatedExplicitVRLittleEndian,
        # pydicom.uid.RLELossless,
    #     pydicom.uid.JPEGLossless,
    #     pydicom.uid.JPEGLosslessP14,
    #     pydicom.uid.JPEGLSLossless,
    #     pydicom.uid.JPEG2000Lossless
    )

    # not_allowed_transfer_syntaxes = (
        # pydicom.uid.JPEGBaseline,
        # pydicom.uid.JPEGExtended,
        # pydicom.uid.JPEGLSLossy,
        # pydicom.uid.JPEG2000,
        # pydicom.uid.JPEG2000MultiComponent,
        # pydicom.uid.JPEG2000MultiComponentLossless,
        # ''
    # )

    config = _get_config()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    ds = pydicom.dcmread(files[0])
    for s in allowed_transfer_syntaxes:
        ds.file_meta.TransferSyntaxUID = s

        temp_file = _get_temp_file_path()
        ds.save_as(temp_file)
        _ = pp.preprocess([temp_file])
        reasons = pp.get_failed_checks()
        # assert passed, "Unexpected flag/s for image transfer syntax {}: {}".format(s, reasons)
        assert 'FAC-210' not in reasons, "Unexpected flag/s for image transfer syntax {}: {}".format(s, reasons)

    # for s in not_allowed_transfer_syntaxes:
    #     ds.file_meta.TransferSyntaxUID = s
    #     temp_file = _get_temp_file_path()
    #     ds.save_as(temp_file)
    #     passed = pp.preprocess([temp_file])
    #     reasons = pp.get_failed_checks()
    #     assert 'FAC-210' in reasons, "Expected flag FAC-210 for transfer syntax '{}'".format(s)


def test_T_157():
    '''
    Ensure Hologic manufacturer models DICOM attributes (that are not covered in other tests) contain valid values
    '''
    data_manager = DataManager()
    # Make sure we use Hologic studies
    file_dxm = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)[0]

    ## Check the attributes that are the same in the three modalities.
    # Tuple meaning: (attribute, valid_values, expected_flag, invalid_values_to_be_flagged, allow_nulls)
    combinations = (
        ('EstimatedRadiographicMagnificationFactor', [1, 1.073], 'HOL-20', [0.5, 2, '1.5'], True),
        ('PhotometricInterpretation', ['MONOCHROME2'], 'HOL-30', ['MONOCHROME1', 'PALETTE COLOR', 'RGB', '', 'monochrome2'], False),
        ('BitsAllocated', [16], 'HOL-40', [8], False),
        ('FocalSpots', [0.3], 'HOL-120', [0.5], True),
        ('ExposureStatus', ['NORMAL'], 'HOL-130', ['ABORTED', 'normal'], True),
        ('WindowCenter', [500, 2000, 2500], 'HOL-160', [499, 2501], None), # test single value here multivalues later
        ('WindowWidth', [500, 4000, 5000], 'HOL-170', [499, 5001], None),   # test single value here multivalues later
        ('WindowCenter', [], 'HOL-80', [''], False),  # test single value here multivalues later
        ('WindowWidth', [], 'HOL-80', [''], False),  # test single value here multivalues later
        ('PixelIntensityRelationship', ['LOG'], 'HOL-180', ['LIN', '', 'log'], False),
        ('PixelIntensityRelationshipSign', [-1], 'HOL-190', [1, ''], False),
        ('PresentationLUTShape', ['IDENTITY'], 'HOL-200', ['INVERSE', 'Identity'], False),
        ('WindowCenterWidthExplanation', [], 'HOL-80', ['Fake 1', ['Fake 1', 'Fake 2']], True),  # test single value here multivalues later
    )

    # for f in (file_dxm, file_bt, file_bt_sc):
    for f in (file_dxm,):
        for attr, valid, expected_flag, invalid_values, allow_missing in combinations:
            print("Processing {} in file {}...".format(attr, f))
            _test_attribute_equals(f, attr, valid, expected_flag, invalid_values=invalid_values,
                                   allow_missing=allow_missing)
    # test multivalue
    window_xx_combinations = (
        ('WindowCenter', [[500, 500, 500]], 'HOL-160', [[499, 499, 499], [2501, 2501, 2501]],None),
        ('WindowWidth', [[4000, 4000, 4000]], 'HOL-170',[[499, 499, 499], [5001, 5001, 5001]], None),
        ('WindowCenter', [], 'HOL-80', [[500,500], [500], 500], False),
        ('WindowWidth', [], 'HOL-80', [[500,500], [500], 500], False),
        ('WindowCenterWidthExplanation', [['Normal', 'High', 'Low']], 'HOL-80', ['Fake 1', ['Fake 1', 'Fake 2']], False),
    )
    ds = pydicom.dcmread(file_dxm)
    attributes = ['WindowCenter', 'WindowWidth', 'WindowCenterWidthExplanation']
    values = [[500, 500, 500], [500, 500, 500], ['Normal', 'High', 'Low']]
    temp_file = _replace_dicom_attribute(ds, attributes, values, tempfile.mktemp())



    for attr, valid, expected_flag, invalid_values, allow_missing in window_xx_combinations:
        print("Processing {} in file {}...".format(attr, temp_file))
        _test_attribute_equals(temp_file, attr, valid, expected_flag, invalid_values=invalid_values,
                               allow_missing=allow_missing)


    # Check for 'Series Description' forbidden words
    banned_words = ['(LE)', '(HE)', '(DES)']
    # for f in (file_dxm, file_bt, file_bt_sc):
    for f in (file_dxm,):
        print("Processing SeriesDescription in file {}...".format(f))
        _test_attribute_not_contains(f, 'SeriesDescription', banned_words, 'HOL-210')

    ## Check for the attributes that are different
    # Similar tuple as previous one, but this time we include the modality to be tested
    combinations = (
        (file_dxm, 'BitsStored', [10, 12], 'HOL-50', [8], False),
        # (file_bt, 'BitsStored', [10], 'HOL-50', [8, '10'], False),
        # (file_bt_sc, 'BitsStored', [10], 'HOL-50', [8, '10'], False),
        (file_dxm, 'HighBit', [9, 11], 'HOL-60', [8], False),
        # (file_bt, 'HighBit', [9], 'HOL-60', [8, '9'], False),
        # (file_bt_sc, 'HighBit', [9], 'HOL-60', [8, '9'], False),
        (file_dxm, 'Grid', ['IN', 'HTC_IN', 'NONE'], 'HOL-110', ['FIXED', 'in', ''], False),
    )
    for f, attr, valid, expected_flag, invalid_values, allow_missing in combinations:
        print("Processing {} in file {}...".format(attr, f))
        _test_attribute_equals(f, attr, valid, expected_flag, invalid_values=invalid_values,
                               allow_missing=allow_missing)


# def test_T_157_2():
#     '''
#     Ensure GE manufacturer models DICOM attributes (that are not covered in other tests) contain valid values
#     '''
#     data_manager = DataManager()
#     # Make sure we use GE studies. At the moment, only GE_DXM is supported
#     file_dxm = data_manager.get_dicom_images(DataManager.STUDY_02_GE)[0]
#
#     config = Config()
#     config[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
#     config[Config.MODULE_ENGINE, 'process_pixel_data'] = False
#     config[Config.MODULE_ENGINE, 'skip_study_checks'] = True
#     pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())
#
#     # Tuple meaning: (attribute, valid_values, expected_flag, invalid_values_to_be_flagged, allow_missing)
#     combinations = (
#         ('EstimatedRadiographicMagnificationFactor', [1], 'GE-20', [0.5, 2, '1.5'], True),
#         ('WindowCenterWidthExplanation', [['NORMAL', 'HARDER', 'SOFTER']], 'GE-80',
#             [['ABNORMAL', 'HARDER', 'SOFTER'], ['TWO', 'WRONG', ''], 'ABNORMAL'], False),
#         ('VOILUTFunction', ['SIGMOID'], 'GE-90', ['SG', 'sigmoid', ''], False),
#         ('Grid', [['RECIPROCATING', 'FOCUSED']], 'GE-110', [['FIXED', 'FOCUSED'], ['', '']], False),
#         ('FocalSpots', [0.3], 'GE-120', [0.5], True),
#         ('ExposureStatus', ['NORMAL'], 'GE-130', ['ABORTED', 'normal'], True),
#         ('PixelIntensityRelationship', ['LOG'], 'GE-180', ['LIN', '', 'log'], False),
#         ('PixelIntensityRelationshipSign', [-1], 'GE-190', [1, ''], False),
#         ('PresentationLUTShape', ['IDENTITY'], 'GE-200', ['INVERSE', 'Identity'], False),
#     )
#     for attr, valid, expected_flag, invalid_values, allow_missing in combinations:
#         print("Processing {}...".format(attr))
#         _test_attribute_equals(file_dxm, attr, valid, expected_flag, invalid_values=invalid_values,
#                                allow_missing=allow_missing)
#
#     # Attributes that trigger a pydicom/pillow exception
#     combinations = (
#         ('PhotometricInterpretation', ['MONOCHROME2'], 'GE-30', ['MONOCHROME1', 'PALETTE COLOR', 'RGB', '', 'monochrome2'], False),
#         ('BitsAllocated', [16], 'GE-40', [8], False),
#         ('BitsStored', [12], 'GE-50', [8], False),
#         ('HighBit', [11], 'GE-60', [8], False),
#     )
#     for attr, valid, expected_flag, invalid_values, allow_missing in combinations:
#         print("Processing 'failing' field {}...".format(attr))
#         try:
#             _test_attribute_equals(file_dxm, attr, valid, expected_flag, invalid_values=invalid_values,
#                                    allow_missing=allow_missing)
#         except AssertionError:
#             raise
#         except:
#             # Expected error
#             # extype, value, tb = sys.exc_info()
#             traceback.print_exc()
#             logging.warning("Error (probably expected) when testing the attribute '{}' for file {}".format(attr, file_dxm))
#
#
#     ## Window Center
#     MIN_ = 2241
#     MAX_ = 3300
#     # Valid case 1
#     ds = pydicom.dcmread(file_dxm)
#     ds['WindowCenterWidthExplanation'].value = ['NORMAL', 'HARDER', 'SOFTER']
#     ds['WindowCenter'].value = [str(random.randint(MIN_, MAX_)), str(random.randint(MIN_, MAX_)),
#                                 str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert passed, "Unexpected flag {}".format(reasons)
#
#     # Valid case 2. Order changed and only 'NORMAL' position has a valid value
#     ds['WindowCenterWidthExplanation'].value = ['HARDER', 'SOFTER', 'NORMAL']
#     ds['WindowCenter'].value = [str(MAX_ + 1), MAX_, str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert passed, "Unexpected flag {}".format(reasons)
#     # Invalid case 1. Lower min value
#     ds['WindowCenterWidthExplanation'].value = ['NORMAL', 'HARDER', 'SOFTER']
#     ds['WindowCenter'].value = [str(MIN_ - 1), str(random.randint(MIN_, MAX_)), str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     # pdb.set_trace()
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert 'GE-160' in reasons, "Expected flag GE-160 with WindowCenterWidthExplanation={} and WindowCenter={}".format(
#         ds['WindowCenterWidthExplanation'].value, ds['WindowCenter'].value
#     )
#     # Invalid case 2. Higher max value
#     ds['WindowCenterWidthExplanation'].value = ['NORMAL', 'HARDER', 'SOFTER']
#     ds['WindowCenter'].value = [str(MAX_ + 1), str(random.randint(MIN_, MAX_)), str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert 'GE-160' in reasons, "Expected flag GE-160 with WindowCenterWidthExplanation={} and WindowCenter={}".format(
#         ds['WindowCenterWidthExplanation'].value, ds['WindowCenter'].value
#     )
#
#     ## Window Width (analog to Window Center)
#     MIN_ = 500
#     MAX_ = 1250
#     # Valid case 1
#     ds = pydicom.dcmread(file_dxm)
#     ds['WindowCenterWidthExplanation'].value = ['NORMAL', 'HARDER', 'SOFTER']
#     ds['WindowWidth'].value = [str(random.randint(MIN_, MAX_)), str(random.randint(MIN_, MAX_)),
#                                str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert passed, "Unexpected flag {}".format(reasons)
#
#     # Valid case 2. Order changed and only 'NORMAL' position has a valid value
#     ds['WindowCenterWidthExplanation'].value = ['HARDER', 'SOFTER', 'NORMAL']
#     ds['WindowWidth'].value = [str(MAX_ + 1), MAX_, str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert passed, "Unexpected flag {}".format(reasons)
#     # Invalid case 1. Lower min value
#     ds['WindowCenterWidthExplanation'].value = ['NORMAL', 'HARDER', 'SOFTER']
#     ds['WindowWidth'].value = [str(MIN_ - 1), str(random.randint(MIN_, MAX_)), str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert 'GE-170' in reasons, "Expected flag GE-160 with WindowCenterWidthExplanation={} and WindowWidth={}".format(
#         ds['WindowCenterWidthExplanation'].value, ds['WindowWidth'].value
#     )
#     # Invalid case 2. Higher max value
#     ds['WindowCenterWidthExplanation'].value = ['NORMAL', 'HARDER', 'SOFTER']
#     ds['WindowWidth'].value = [str(MAX_ + 1), str(random.randint(MIN_, MAX_)), str(random.randint(MIN_, MAX_))]
#     temp_file = _get_temp_file_path()
#     ds.save_as(temp_file)
#     passed = pp.preprocess([temp_file])
#     reasons = pp.get_failed_checks()
#     assert 'GE-170' in reasons, "Expected flag GE-160 with WindowCenterWidthExplanation={} and WindowWidth={}".format(
#         ds['WindowCenterWidthExplanation'].value, ds['WindowWidth'].value
#     )

# def test_T_158():
#     """
#     2D/3D modalities consistent
#     """
#     config = _get_config(skip_study_checks=False, process_pixel_data=True)
#     pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())
#
#     # Get a study with Dxm and DBT images
#     data_manager = DataManager()
#     study_files_full_path = data_manager.get_dicom_images(DataManager.STUDY_03_DBT_HOLOGIC)
#
#     # First, preprocess all the files
#     passed = pp.preprocess(study_files_full_path)
#     assert passed, "Unexpected flags: {}".format(pp.get_failed_checks())
#
#     # Get the metadata to remove selected files
#     metadata_df = pp.get_metadata()
#     file_names = [f.replace(".dcm", "").replace("DXm.", "").replace("BT.", "") for f in
#                   map(os.path.basename, study_files_full_path)]
#
#     dxm_files = metadata_df.loc[metadata_df['SOPClassUID'] == DicomTypeMap.get_dxm_class_id(), 'SOPInstanceUID']
#     # dbt_files = metadata_df.loc[metadata_df['SOPClassUID'] == DicomTypeMap.get_dbt_class_id(), 'SOPInstanceUID']
#
#     # Remove a Dxm file
#     ix = file_names.index(dxm_files.iloc[0])
#     files_tmp = study_files_full_path.copy()
#     del files_tmp[ix]
#     _ = pp.preprocess(files_tmp)
#     assert 'SAC-130' in pp.get_failed_checks(), "Flag SAC-130 expected because there is a DBT file with no matching Dxm"
#
#     # Remove one of the DBT images (it should fail)
#     # files_tmp = study_files_full_path.copy()
#     # ix = file_names.index(dbt_files.iloc[0])
#     # del files_tmp[ix]
#     # passed = pp.preprocess(files_tmp)
#     # assert 'SAC-132' in pp.get_failed_checks(), "Flag SAC-132 expected when removing a DBT image in one laterality"
#
#     # Remove all the images for one laterality (only DBT). It should still fail because there are Dxms without DBT
#     files_tmp = []
#     for i in range(len(metadata_df)):
#         if metadata_df.iloc[i]['ImageLaterality'] != 'L' or \
#                 metadata_df.iloc[i]['SOPClassUID'] == DicomTypeMap.get_dxm_class_id():
#             files_tmp.append(study_files_full_path[i])
#     pp.preprocess(files_tmp)
#     assert 'SAC-131' in pp.get_failed_checks(), "Flag SAC-132 expected because there are Dxm files with no matching DBT"
#
#     # Remove all the DBT images
#     files_tmp = []
#     for i in range(len(metadata_df)):
#         if metadata_df.iloc[i]['SOPClassUID'] == DicomTypeMap.get_dxm_class_id():
#             files_tmp.append(study_files_full_path[i])
#     passed = pp.preprocess(files_tmp)
#     assert passed, "Unexpected flags: {}".format(pp.get_failed_checks())
#
#     # Create an "extra" Dxm image. It should pass
#     files_tmp = study_files_full_path.copy()
#     ds = pydicom.dcmread(files_tmp[0])
#     ds.SOPInstanceUID += ".2"
#     with tempfile.TemporaryDirectory() as temp_dir:
#         p = os.path.join(temp_dir, "temp_img.dcm")
#         ds.save_as(p)
#         files_tmp.append(p)
#         passed = pp.preprocess(files_tmp)
#         assert passed, "Unexpected flags: {}".format(pp.get_failed_checks())

def test_T_159():
    """
    Test for unsupported lateralities/views
    """
    config = _get_config(skip_study_checks=False, process_pixel_data=True)

    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Get a study with DBT and DXM files
    data_manager = DataManager()
    study_files_full_path = data_manager.get_dicom_images(DataManager.STUDY_03_DBT_HOLOGIC)
    processed_ok = pp.preprocess(study_files_full_path)
    assert processed_ok, "Baseline was not processed ok. Flags: {}".format(pp.get_failed_checks())

    # _test_attribute_equals(study_files_full_path[0], "ViewPosition", ["CC", "MLO"], "FAC-160")

    metadata_df = pp.get_metadata()
    file_names = [f.replace(".dcm", "").replace("DXm.", "").replace("BT.", "")
                  for f in map(os.path.basename, study_files_full_path)]

    # Select left images only (both DBT and Dxm)
    left_files = metadata_df[(metadata_df['ImageLaterality'] == 'L')]['dcm_path'].to_list()
    _ = pp.preprocess(left_files)
    failed_checks = pp.get_failed_checks()
    assert 'SAC-40' in failed_checks and 'SAC-60' in failed_checks, \
        "Expected  SAC-40 and SAC-60 flags. Got: {}".format(failed_checks)

    # Select right images only (both DBT and Dxm)
    right_files = metadata_df[metadata_df['ImageLaterality'] == 'R']['dcm_path'].to_list()
    _ = pp.preprocess(right_files)
    failed_checks = pp.get_failed_checks()
    assert 'SAC-30' in failed_checks and 'SAC-50' in failed_checks, \
        "Expected  SAC-30 and SAC-50 flags. Got: {}".format(failed_checks)

    # Add an extra file with a wrong laterality. The study should still pass
    files_tmp = study_files_full_path.copy()
    ds = pydicom.dcmread(files_tmp[0])
    ds.SOPInstanceUID += ".2"
    ds.ImageLaterality = "RR"
    with tempfile.TemporaryDirectory() as temp_dir:
        p = os.path.join(temp_dir, "temp_img.dcm")
        ds.save_as(p)
        files_tmp.append(p)
        passed = pp.preprocess(files_tmp)
        failed_checks = pp.get_failed_checks()
        assert passed, "Study should pass! Unexpected flags: {}".format(failed_checks)
        assert 'FAC-29' in failed_checks, "'FAC-29' was expected for the fake file {}".format(p)

def test_T_218():
    """
    Tests that the checks specified in 'checks_modes.csv' under deephealth_utils.data.dicom_specs have valid prefixes
    and are 'known' checks specified either in one of the other CSVs under deephealth_utils.data.dicom_specs or in some
    hardcoded lists of checks.
    """
    dh_utils_path = os.path.dirname(deephealth_utils.__file__)
    dicom_specs_path = os.path.join(dh_utils_path, 'data', 'dicom_specs')

    # Reading in CSVs
    checks_modes_df = pd.read_csv(os.path.join(dicom_specs_path, 'checks_modes.csv'))
    common_df = pd.read_csv(os.path.join(dicom_specs_path, 'common.csv'))
    checks_df = pd.read_csv(os.path.join(dicom_specs_path, 'checks.csv'))
    hologic_df = pd.read_csv(os.path.join(dicom_specs_path, 'hologic.csv'))
    ge_df = pd.read_csv(os.path.join(dicom_specs_path, 'ge.csv'))

    # Configurations
    #checks_in_validation_py = ['FAC-141', 'FAC-142']
    checks_in_validation_py = ['FAC-140', 'FAC-20']
    checks_in_checker_py = ['FAC-10', 'FAC-100', 'FAC-101']
    checks_in_checker_functions_py = ['GE-160', 'GE-170', 'GE-80','HOL-160', 'HOL-170', 'HOL-80']
    checks_in_multiple_locations = ['FAC-160','FAC-140', 'FAC-20']

    # Putting these CSVs together
    # For common_df, we need to parse FAC-200-X to FAC-200
    known_locations = [['-'.join(el.split('-')[0:2]) for el in np.array(common_df['AcceptanceCriteria'])],
                     np.array(checks_df['AcceptanceCriteria']),
                     np.array(hologic_df['AcceptanceCriteria']),
                     np.array(ge_df['AcceptanceCriteria']),
                     checks_in_validation_py,
                     checks_in_checker_py,
                     checks_in_checker_functions_py]

    accepted_check_prefixes = ['FAC', 'SAC', 'PAC', 'IAC', 'HOL', 'GE']

    # Testing the checks contained in checks_modes_df
    for idx, row in checks_modes_df.iterrows():

        ac = row['Acceptance Criteria Number']

        assert any(ac.startswith(prefix) for prefix in accepted_check_prefixes), \
            'AC {} does not have an accepted check prefix'.format(ac)

        # Assert that it is in a known location
        ac_in_csv = [ac in location for location in known_locations]

        if ac not in checks_in_multiple_locations:
            assert sum(ac_in_csv) == 1, '{} not found in expected location(s)'.format(ac)

        if ac.startswith('IAC'):
            # Assert that it is in checks_that_need_pixels()
            assert ac in Checker.checks_that_need_pixels(), '{} is not in Checker.checks_that_need_pixels()'.format(ac)

