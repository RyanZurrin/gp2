from glob import glob
import pytest
import pandas as pd
import os
import os.path as osp
import tempfile
import shutil
import pydicom

import centaur_deploy.constants as const_deploy
from centaur_deploy.deploys.config import Config
from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_engine.preprocessor import Preprocessor
import centaur_engine.helpers.helper_misc as helper_misc
from centaur_test.data_manager import DataManager
import centaur_test.utils as utils
from deephealth_utils.data.checker import Checker
import deephealth_utils.data.parse_logs as parse_logs
from deephealth_utils.data.utils import dh_dcmread


@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    """
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    """
    record_xml_attribute("classname", "4_Integration")

def test_T_4_I01():
    """
    Create a new logger and make sure that the right messages are written.
    Preprocess the baseline study modified so that one file fails, and make sure the error is logged
    log files contain both flags.
    """
    data_manager = DataManager()
    # Preprocess the files
    config = Config()
    config[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
    config[Config.MODULE_ENGINE, 'process_pixel_data'] = False
    config[Config.MODULE_ENGINE, 'checks_to_ignore'] = Checker.checks_that_need_pixels()
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Initialize logger
    output_folder = tempfile.mkdtemp(suffix="_logger")
    try:
        logger, logger_file = helper_misc.create_logger(output_folder, return_path=True)
        # Check there are no errors when writing and debug messages are not logged
        logger.info("This is an info message that should be logged")
        logger.info("This is a warning message that should be logged")
        logger.debug("DO NOT LOG THIS")
        with open(logger_file, 'r') as f:
            s = f.read()
        assert "This is an info message that should be logged" in s, "Log message not found"
        assert "This is a warning message that should be logged" in s, "Log message not found"
        assert "DO NOT LOG THIS" not in s, "Logger is saving debug messages when only info and above were expected"

        pp.set_logger(logger)

        # Read some files to be processed
        hologic_files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)

        ds = pydicom.dcmread(hologic_files[0])
        # Remove a mandatory attribute
        del ds['StudyInstanceUID']
        temp = osp.join(tempfile.gettempdir(), "0.dcm")
        ds.save_as(temp)
        hologic_files[0] = temp
        passed = pp.preprocess(hologic_files)
        reasons = pp.get_failed_checks()
        with open(logger_file, 'r') as f:
            s = f.read()
        # Make sure that the study did not pass the validations and the flag was logged
        for flag in ("FAC-200", "SAC-50"):
            assert flag in reasons, "StudyInstanceUID not present, flag {} expected".format(flag)
            assert flag in s, "Flag {} could not be found in logger. Full content: {}".format(flag, s)
    finally:
        if osp.isdir(output_folder):
            shutil.rmtree(output_folder)

def test_T_4_I02():
    """
    Validate dicom_df parsed after logging.
    Create a new logger and preprocess two studies (one correct and one wrong)
    Ensure the log messages can be created and the dataframe is parsed correctly and it contains the expected info.
    """
    # Preprocess the files
    config = Config()
    config[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Initialize logger
    output_folder = tempfile.mkdtemp(suffix="_logger")
    try:
        logger, logger_file = helper_misc.create_logger(output_folder, return_path=True)
        pp.set_logger(logger)

        # Read some files to be processed
        data_manager = DataManager()
        hologic_files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
        hologic_files_2 = data_manager.get_dicom_images(DataManager.STUDY_04_HOLOGIC)

        # Set a wrong value for SOPClassUID ("Mammography for Processing instead of Presentation) in one of the files
        # (Note: do not pick the first one because the first file folder is used as study name in the dataframe)
        ds = pydicom.dcmread(hologic_files[-1])
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2.1"
        temp_ds_file1 = osp.join(tempfile.gettempdir(), "0.dcm")
        ds.save_as(temp_ds_file1)
        hologic_files[-1] = temp_ds_file1

        # Set a wrong value in the Modality for one of the files
        ds = pydicom.dcmread(hologic_files[-2])
        ds.Modality = "MGG"
        temp_ds_file2 = osp.join(tempfile.gettempdir(), "1.dcm")
        ds.save_as(temp_ds_file2)
        hologic_files[-2] = temp_ds_file2

        passed = pp.preprocess(hologic_files)
        # Expected flags
        reasons = set(pp.get_failed_checks())
        expected_flags = {"FAC-20", "FAC-21", "SAC-30", "SAC-40"}
        assert reasons == expected_flags, "Expected flags: {}. Got: {}".format(expected_flags, reasons)

        passed = pp.preprocess(hologic_files_2)
        assert passed, "The study should have passed the validations. Got errors: {}".format(pp.get_failed_checks())

        # Remove temp files
        os.remove(temp_ds_file1)
        os.remove(temp_ds_file2)

        # Parse the log file
        parser = parse_logs.LogReader(logger_file)
        parser.iter_lines()
        dicom_df = parser.get_file_df()
        dicom_df = dicom_df.set_index('filename')

        # Basic assertions
        assert len(dicom_df) == len(hologic_files) + len(hologic_files_2), "{} files expected in the dicom dataframe. Got {}". \
            format(len(hologic_files) + len(hologic_files_2), len(dicom_df))
        ge_df = dicom_df[dicom_df['study'] == DataManager.STUDY_02_GE]
        assert len(ge_df[ge_df['failed_checks'] != '']) == 0, "There seems to be unexpected failed checks for correct study"

        # Advanced assertions
        for f, flags, expected in (('0.dcm', ('FAC-20'),
                                    ('1.2.840.10008.5.1.4.1.1.1.2.1', 'Lorad Selenia_1.2.840.10008.5.1.4.1.1.1.2.1')),
                                  ('1.dcm', 'FAC-21', ('MGG',)),):
            # failed_checks has the right flags

            assert dicom_df.loc[f]['failed_checks'] == '{}'.format(
                flags), "Expected failed-checks column value in file {}: '{}'. Got: {}". \
                format(f, flags, dicom_df.loc[f]['failed_checks'])
            for flag, expected_val in zip(flags.split('|'), expected):
                # the corresponding flag column has the right value
                assert dicom_df.loc[f][flag] == expected_val, \
                    "Expected column value in file {}: {}. Got: {}".format(f, expected_val, dicom_df.loc[f][flag])
                # Only that corresponding flag column has a value
                for column in [c for c in dicom_df.columns.tolist() if c.startswith('FAC-') and c not in flags]:
                    assert pd.isnull(dicom_df.loc[f][column]), "Expected null value for file {} and column {}. Got: {}".\
                        format(f, column, dicom_df.loc[f][column])
    finally:
        if osp.isdir(output_folder):
            shutil.rmtree(output_folder)

def test_T_4_I03():
    """
    Validate summary dataframes parsed after logging (use case 1)
    Create a new logger and preprocess two studies (one correct and one wrong).
    Ensure the log messages can be created and the dataframe is parsed correctly and contains the expected info.
    The first failed image is checked
    """
    # Preprocess the files
    config = Config()
    config[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
    config[Config.MODULE_ENGINE, 'process_pixel_data'] = True
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Initialize logger
    output_folder = tempfile.mkdtemp(suffix="_logger")
    temp_ds_file = ""
    try:
        logger, logger_file = helper_misc.create_logger(output_folder, return_path=True)
        pp.set_logger(logger)

        # Read some files to be processed
        data_manager = DataManager()
        hologic_files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
        hologic_files_2 = data_manager.get_dicom_images(DataManager.STUDY_04_HOLOGIC)

        hologic_path = '/'.join(hologic_files[0].split('/')[0:-1])
        hologic_path2 = '/'.join(hologic_files_2[0].split('/')[0:-1])

        # Modify Patient Name in one of the files in the GE study
        ds = pydicom.dcmread(hologic_files_2[0])
        patient_name = str(ds['PatientName'].value)
        ds['PatientName'].value = patient_name + '2'
        # Saved here because the preprocessor expects all the files in the same folder
        temp_ds_file = osp.join(hologic_path2, "temp_000.dcm")
        ds.save_as(temp_ds_file)
        hologic_files_2[0] = temp_ds_file

        pp.study_dir = hologic_path
        passed = pp.preprocess(hologic_files)
        # Expected pass
        assert passed, "The study should have passed the validations"

        pp.study_dir = hologic_path2
        _ = pp.preprocess(hologic_files_2)
        reasons = pp.get_failed_checks()
        assert reasons == ['SAC-70'], "Only SAC-70 flag expected. Got: {}".format(reasons)

        # Parse the log file
        parser = parse_logs.LogReader(logger_file)
        parser.iter_lines()
        dicom_df = parser.get_file_df()
        study_df = parser.get_study_df().set_index('filename')

        # Assertions
        # No FACs
        assert len(dicom_df) == len(hologic_files_2) + len(hologic_files), "Total number of dicom rows expected: {}. Got: {}".\
            format(len(hologic_files_2) + len(hologic_files), len(dicom_df))
        assert len(dicom_df[dicom_df['failed_checks'] != '']) == 0, "Some unexpected failures were found in failed_checks column"
        # 2 studies
        assert len(study_df) == 2, "2 rows expected in the study dataframe. Got {}".format(len(study_df))
        # Hologic OK
        assert study_df.loc[hologic_path]['failed_checks'] == '', "Expected: (). Got: {}".\
            format(study_df.loc[hologic_path]['failed_checks'])
        # GE failed
        assert study_df.loc[hologic_path2]['failed_checks'] == "SAC-70", "Expected: ('SAC-70',). Got: {}".\
            format(study_df.loc[hologic_path2]['failed_checks'])
        # Only corresponding flag columns are present
        assert study_df.columns.tolist() == ['failed_checks', 'SAC-70', 'passes_checker'], \
            "Expected columns: ['failed_checks', 'SAC-70', 'passes_checker']. Got: {}".format(study_df.columns.tolist())
        # Flag column value is correct
        expected_value = "PatientName: ('{0}2', '{0}', '{0}', '{0}')".format(patient_name)
        assert study_df.loc[hologic_path2]['SAC-70'] == expected_value, \
            "Expected value for column 'SAC-70': {}. Got: {}".format(expected_value, study_df.loc[hologic_path2]['SAC-70'])
    finally:
        if os.path.isfile(temp_ds_file):
            os.remove(temp_ds_file)
        if osp.isdir(output_folder):
            shutil.rmtree(output_folder)

def test_T_4_I04():
    """
    Validate summary dataframes parsed after logging (use case 2)
    Create a new logger and preprocess two studies (one correct and one wrong).
    Ensure the log messages can be created and the dataframe is parsed correctly and contains the expected info.
    The last failed image is checked
    """
    # Preprocess the files
    config = Config()
    config[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
    pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

    # Initialize logger
    output_folder = tempfile.mkdtemp(suffix="_logger")
    temp_ds_file = ""
    try:
        logger, logger_file = helper_misc.create_logger(output_folder, return_path=True)
        pp.set_logger(logger)

        # Read some files to be processed
        data_manager = DataManager()
        hologic_files = data_manager.get_dicom_images(DataManager.STUDY_01_HOLOGIC)
        #ge_files = data_manager.get_dicom_images(DataManager.STUDY_02_GE)
        hologic_files2 = data_manager.get_dicom_images(DataManager.STUDY_04_HOLOGIC)

        hologic_path = '/'.join(hologic_files[0].split('/')[0:-1])
        hologic_path2 = '/'.join(hologic_files2[0].split('/')[0:-1])

        # Modify Patient Name in one of the files in the GE study
        # (Note: do not pick the first one because the first file folder is used as study name in the dataframe)
        ds = pydicom.dcmread(hologic_files2[-1])
        patient_name = str(ds['PatientName'].value)
        ds['PatientName'].value = patient_name + '2'
        # Save here because the preprocessor expects all the files in the same folder
        temp_ds_file = osp.join(hologic_path2, "temp_000.dcm")
        ds.save_as(temp_ds_file)
        hologic_files2[-1] = temp_ds_file

        pp.study_dir = hologic_path
        passed = pp.preprocess(hologic_files)
        # Expected pass
        assert passed, "The study should have passed the validations"

        pp.study_dir = hologic_path2
        passed = pp.preprocess(hologic_files2)
        reasons = pp.get_failed_checks()
        assert reasons == ['SAC-70'], "Only SAC-70 flag expected. Got: {}".format(reasons)

        # Parse the log file
        parser = parse_logs.LogReader(logger_file)
        parser.iter_lines()
        dicom_df = parser.get_file_df()
        study_df = parser.get_study_df().set_index('filename')
        # Assertions
        # No FACs
        assert len(dicom_df) == len(hologic_files2) + len(hologic_files), "Total number of dicom rows expected: {}. Got: {}".\
            format(len(hologic_files2) + len(hologic_files), len(dicom_df))
        assert len(dicom_df[dicom_df['failed_checks'] != '']) == 0, "Some unexpected failures were found in failed_checks column"
        # 2 studies
        assert len(study_df) == 2, "2 rows expected in the study dataframe. Got {}".format(len(study_df))
        # Hologic OK
        assert study_df.loc[hologic_path]['failed_checks'] == '', "Expected: (). Got: {}".\
            format(study_df.loc[hologic_path]['failed_checks'])
        # GE failed
        assert study_df.loc[hologic_path2]['failed_checks'] == "SAC-70", "Expected: SAC-70. Got: {}".\
            format(study_df.loc[hologic_path2]['failed_checks'])
        # Only corresponding flag columns are present
        assert study_df.columns.tolist() == ['failed_checks', 'SAC-70', 'passes_checker'], \
            "Expected columns: ['failed_checks', 'SAC-70', 'passes_checker']. Got: {}".format(study_df.columns.tolist())
        # Flag column value is correct
        expected_value = "PatientName: ('{0}', '{0}', '{0}', '{0}2')".format(patient_name)
        assert study_df.loc[hologic_path2]['SAC-70'] == expected_value, \
            "Expected value for column 'SAC-70': {}. Got: {}".format(expected_value, study_df.loc[hologic_path2]['SAC-70'])
    finally:
        if os.path.isfile(temp_ds_file):
            os.remove(temp_ds_file)
        if osp.isdir(output_folder):
            shutil.rmtree(output_folder)


@pytest.mark.cadt
def test_T_222():
    """
    test Saige-Q SR making checking metadata and category assignment
    Returns:None

    """
    dm = DataManager()
    dm.set_baseline_params(const_deploy.RUN_MODE_CADT)

    studies_valid  = dm.get_valid_studies()
    all_studies = dm.get_all_studies()

    for study in all_studies:
        study_path = os.path.join(dm.baseline_dir_centaur_output, study)
        study_sr_path_list = glob(os.path.join(study_path, 'DH_SaigeQ_SR_*'))
        if study not in studies_valid:
            assert len(study_sr_path_list) == 0 , "Studies that did not pass checker shouldnt have SR output"
            continue
        assert len(study_sr_path_list) ==1, "Studies that passed checker should only have one DH_SaigeQ_SR_XXX file"
        study_sr_path = study_sr_path_list[0]

        study_results_path = os.path.join(study_path, const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
        sr = dh_dcmread(study_sr_path)
        study_results = StudyDeployResults.from_json(study_results_path)

        assert check_sr_baseline(sr, study_results), "mismatching metadata for study {}".format(study)
        assert check_category(sr, study_results), "mismatching category for study {}".format(study)


def check_sr_baseline(sr, study_results):
    """
    check to see if the metadata match between results and SR
    Args:
        sr: StructuredReport instance
        study_results: StudyDeployResults instance

    Returns: bool

    """
    study_attributes = ['PatientName',
                        'PatientID',
                        'StudyInstanceUID',
                        'StudyDate',
                        'ReferringPhysicianName',
                        'StudyID',
                        'StudyTime',
                        'AccessionNumber',
                        'PatientBirthDate',
                        'PatientSex']
    for attr in study_attributes:
        if sr.dh_getattribute(attr) != study_results.metadata[attr].values[0]:
            return False
        return True


def check_category(sr, study_results):
    """
    check to see if category matches
    Args:
        sr: StructuredReport instance
        study_results: StudyDeployResults instance

    Returns: bool

    """
    study_results_cat = study_results.results['study_results']['total']['category']
    sr_cat = sr.ContentSequence[2].ContentSequence[2].TextValue
    assert sr_cat in ['', 'Suspicious'], "invalid SR category {}".format(sr_cat)
    sr_cat = 0 if sr_cat == '' else 1
    if sr_cat == study_results_cat:
        return True
    return False


# if __name__ == "__main__":
    # test_Saige_Q_SR()
    # test_T_4_I02()
    # test_T_4_I03()
    # test_T_4_I04()
