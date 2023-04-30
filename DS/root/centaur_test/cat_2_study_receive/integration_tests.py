import pydicom
import pytest
import os
import re
import shutil
import subprocess
import time

import os.path as osp
import tempfile

from centaur_io.input.dir_monitor_input import DirMonitorInput
from centaur_io.input.pacs_input import PACSInput
from centaur_test.data_manager import DataManager
import centaur_deploy.constants as const_deploy
from centaur_deploy.deploys.studies_db import StudiesDB
from centaur_test.dicom_sender import DicomSender

@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    '''
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    '''
    record_xml_attribute("classname", "2_Integration")

@pytest.mark.timeout(60)
def test_T_161():
    """
    Check that the Pacs IO class is able to receive DICOM studies using DCMTK PACS emulator
    """
    data_manager = DataManager()
    temp_folder = tempfile.mkdtemp()
    try:
        input_dir = osp.join(temp_folder, "input")
        os.makedirs(input_dir)
        output_dir = osp.join(temp_folder, "output")
        os.makedirs(output_dir)
        print("Output folder: {}".format(output_dir))

        # Create StudiesDB
        studies_db = StudiesDB(os.path.join(output_dir, const_deploy.STUDIES_DB_FILE_NAME))
        # Init listener object
        print("Starting listener...")
        port = 29999
        studies_uids = {}
        dicom_sender = DicomSender(port)
        with PACSInput(input_dir, port, "Centaur-receive", studies_db, 180) as pacsIO:
            # Pick one Dxm and one Dbt study
            pacsIO.start_receiver()
            studies = data_manager.get_all_studies()
            root_folder = data_manager.data_studies_dir

            for study in studies:
                print("Sending {}".format(study))
                study_path = osp.join(root_folder, study)
                # Save the expected uid
                files = os.listdir(study_path)
                ds = pydicom.dcmread(osp.join(study_path, files[0]))
                studies_uids[study] = ds.StudyInstanceUID
                dicom_sender.send_study(study_path)


        # Check the output folder to check that the studies have been "sent"
        print("Waiting 5 seconds for studies to be processed...")
        time.sleep(5)
        for study in studies:
            dest_study_path = osp.join(input_dir, "+B_" + studies_uids[study])
            assert osp.isdir(dest_study_path), f"Study not found in folder {dest_study_path}"
            source_study_path = osp.join(root_folder, study)
            # TODO: ensure the files are identical (MD5 is different)...
            assert len(os.listdir(dest_study_path)) == len(os.listdir(source_study_path)), \
                f"Files do not match in folders {dest_study_path} and {source_study_path}"
            # import glob
            # files_src = glob.glob(source_study_path + "/*")
            # src_set = {DataManager.md5(f) for f in files_src}
            # files_dst =  glob.glob(dest_study_path + "/*")
            # dst_set = {DataManager.md5(f) for f in files_dst}
            # assert src_set == dst_set

    finally:
        shutil.rmtree(temp_folder)


def test_T_213():
    """ Test whether input dir monitoring service inserts a new input study into StudiesDB

    Setup:
        1. Create an input_dir and studies_db_dir
        2. Create StudiesDB
        3. Start monitor service as a thread

    Test:
        1. Test whether monitor service does not insert a row for absent study into StudiesDB
            a) query StudiesDB by input_path and check no row is returned
        2. Test whether monitor service inserts a row for received study into StudiesDB
            a) query StudiesDB by input_path and check a new row is returned.
    """
    NEW_STUDY_THRESHOLD_SECONDS = 5

    # Test input_dir monitor using test cases stored in Centaur docker container
    data_manager = DataManager()
    temp_folder = tempfile.mkdtemp()
    try:
        # Create working directories
        input_dir = os.path.join(temp_folder, "input")
        os.mkdir(input_dir)
        studies_db_dir = os.path.join(temp_folder, "output")
        os.mkdir(studies_db_dir)

        # Create StudiesDB (created in current directory for this test)
        studies_db_path = os.path.join(studies_db_dir, const_deploy.STUDIES_DB_FILE_NAME)
        studies_db = StudiesDB(studies_db_path)

        # Start dir monitor
        input_dir_monitor = DirMonitorInput(input_dir, studies_db,
                                            new_study_threshold_seconds=NEW_STUDY_THRESHOLD_SECONDS)

        # Test with three test cases
        study_names = [DataManager.STUDY_01_HOLOGIC, DataManager.STUDY_02_GE, DataManager.STUDY_03_DBT_HOLOGIC]
        for study_name in study_names:
            print("Testing with {}...".format(study_name))
            study_src_dir = data_manager.get_input_dir(study_name)
            input_path = "{}/{}".format(input_dir, study_name)

            print("Check whether StudiesDB is not updated...")
            df = studies_db.query_by_input_path(input_path)
            assert df.empty, f"A row for the study {study_name} should not be inserted by input_dir_monitor"

            print("Copying a test case into input_dir...")
            shutil.copytree(study_src_dir, input_path)
            for f in os.listdir(input_path):
                os.utime(os.path.join(input_path, f))

            print("Wait until input_dir_monitor inserts a row for the study into StudiesDB...")
            assert input_dir_monitor.check_new_studies() == 0, \
                   f"Study {study_name} inserted in StudiesDB before expected"
            time.sleep(NEW_STUDY_THRESHOLD_SECONDS+4)
            assert input_dir_monitor.check_new_studies() == 1, \
                f"Study {study_name} was expected to be inserted but it doesn't look like that"

            df = studies_db.query_by_input_path(input_path)
            assert len(df) == 1, "A row for the study should be inserted by input_dir_monitor"
    finally:
        shutil.rmtree(temp_folder)

# if __name__ == '__main__':
#     test_T_213()
#     print("DONE!")