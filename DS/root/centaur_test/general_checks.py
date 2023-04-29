import os
import pytest
import traceback
from centaur_test.data_manager import DataManager

"""
Mainly import tests and also data test to verify the integrity of the testing dataset
"""

def test_centaur_engine():
    try:
        import centaur_engine
    except:
        traceback.print_exc()
        pytest.fail("centaur_engine module could not be imported")

def test_deephealth_utils():
    try:
        import deephealth_utils
    except:
        traceback.print_exc()
        pytest.fail("deephealth_utils module could not be imported")


def test_external_packages():
    try:
        from google.cloud import storage
    except:
        traceback.print_exc()
        pytest.fail("google cloud storage could not be imported. Try running pip install google-cloud-storage. Check https://cloud.google.com/python/setup for additional info")

def test_pytest_html():
    try:
        from py.xml import html
    except:
        traceback.print_exc()
        pytest.fail("pytest-html not installed. You cannot generate html reports. Try pip install pytest-html")

def test_cat1_imports():
    try:
        import centaur_test.cat_1_study_check.unit_tests
    except:
        traceback.print_exc()
        pytest.fail("Some imports for category 1 went wrong")

def test_cat2_imports():
    try:
        import centaur_test.cat_2_study_receive.unit_tests
        import centaur_test.cat_2_study_receive.integration_tests
    except:
        traceback.print_exc()
        pytest.fail("Some imports for category 2 went wrong")


def test_cat3_imports():
    try:
        import centaur_test.cat_3_study_run.unit_tests
        import centaur_test.cat_3_study_run.integration_tests
    except:
        traceback.print_exc()
        pytest.fail("Some imports for category 3 went wrong")

def test_cat6_imports():
    try:
        import centaur_test.cat_6_installation.unit_tests
    except:
        traceback.print_exc()
        pytest.fail("Some imports for category 6 went wrong")

def test_cat8_imports():
    try:
        import centaur_test.cat_8_system.system_tests
    except:
        traceback.print_exc()
        pytest.fail("Some imports for category 8 went wrong")

def test_data_files():
    """
    Validate that all the local files are available and they have the right md5 hash
    :param data_manager: data.data_manager.DataManager object (fixture defined in conftest.py file)
    :param dataset: str. Dataset version (fixture defined in conftest.py file)
    """
    data_manager = DataManager()
    df = data_manager.get_filelist_df()
    assert "md5" in df.columns, "'md5' column could not be found in the filelist dataframe. File columns: {}".\
        format(df.columns.to_list())

    # Go over all the files and validate the md5 hash
    for key, row in df.iterrows():
        local_paths = data_manager.get_files("^{}$".format(row['file_path']))
        assert len(local_paths) == 1 and os.path.isfile(local_paths[0]), "Error in row:\n{}.\nLocal path obtained: {}".\
            format(row, local_paths)
        local_path = local_paths[0]
        local_md5 = data_manager.md5(local_path)

        assert local_md5 == row['md5'], "MD5 signature does not match for file {}. Expected: {}; Calculated: {}". \
                                        format(local_path, row['md5'], local_md5)
