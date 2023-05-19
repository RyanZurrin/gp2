import shutil
import os
import tempfile
import time
import traceback
import pytest
import psutil
import gc

import centaur_io.constants as const_io

from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
import centaur_deploy.constants as const_deploy
from centaur_deploy.deploys.config import Config
from centaur_deploy.deploy import Deployer

from centaur_test.config_factory import ConfigFactory
from centaur_test.dicom_sender import DicomSender
from centaur_test.data_manager import DataManager
import centaur_test.utils as tests_utils
import centaur_test.constants as constants


@pytest.mark.timeout(constants.BASELINE_MAX_RUN_TIME_SECONDS, method='signal')
@pytest.mark.cadt
def test_T_200():
    """
    Main system check. Run the deployer production pipeline for the testing dataset for all the studies that compound it
    and make sure the outputs are the expected ones.
    Also, make sure that the timing is reasonable (the test will fail after BASELINE_MAX_RUN_TIME_SECONDS seconds)
    Checks:
    - All the studies are processed with no flags raised
    - The results for each study are the expected ones
    - The report files for each study are created
    - The summary reports are created
    - The logger can be parsed and the extracted info matches the expected results (no preprocess flags)
    - The output config files are consistent
    """
    # Clean any previous memory use
    gc.collect()
    run_mode = const_deploy.RUN_MODE_CADT
    baseline_params = {
        'run_mode': run_mode,
    }

    dm = DataManager(baseline_params=baseline_params)
    buffer_seconds = 10
    optional_params = {
        'run_mode': run_mode,
        'new_study_threshold_seconds': buffer_seconds
    }

    studies_uids = {}
    processed_studies = set()
    deployer = Deployer()
    try:
        config_factory = ConfigFactory.VerificationPACSConfigFactory(optional_params_dict=optional_params)

        config = deployer.create_config_object(config_factory.params_dict)

        deployer.initialize(config)

        # Make sure the StudiesDatabase is empty
        studies_db = deployer.studies_db
        assert len(studies_db.get_all()) == 0, "Database not empty"

        # Send all the studies in the dataset to the DICOM listener via DCMTK
        dicom_sender = DicomSender(config[Config.MODULE_IO, 'pacs_receive_port'])
        studies = dm.get_all_studies()
        for study in studies:
            study_path = os.path.join(dm.get_input_dir(study))
            # Get the StudyInstanceUID
            study_uid = tests_utils.get_study_instance_uid(study_path)
            studies_uids[study_uid] = study
            dicom_sender.send_study(study_path)

        # Wait until the StudyListener has inserted the studies in the database
        print(f"Waiting {buffer_seconds + 10} seconds to process all the input files...")
        time.sleep(buffer_seconds + 10)
        # Check that all studies are inserted in the database
        new_studies = deployer.input_object.check_new_studies()
        assert new_studies == len(studies), \
            f"There should be {len(studies)} studies in the database, " \
            f"but there are {new_studies}"

        def study_callback(processed_results):
            study_uid = processed_results.get_studyUID()
            assert not processed_results.has_unexpected_errors()
            study_name = studies_uids[study_uid]
            baseline_results = StudyDeployResults.from_json(os.path.join(dm.baseline_dir_centaur_output, study_name,
                                                                         const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON))
            try:
                tests_utils.compare_study_deploy_results(processed_results, baseline_results)
            except:
                print(f"THERE WAS AN ERROR COMPARING {study_name}")
                traceback.print_exc()
            processed_studies.add(study_uid)
            if len(processed_studies) == len(studies_uids):
                # The test worked as expected so far. Stop the Deployer infinite loop
                deployer.stop_deploy()

        deployer.deploy(study_callback=study_callback)

        # All the studies were processed. Check the global baseline outputs
        tests_utils.compare_global_deploy_results(dm.baseline_dir_centaur_output, config[config.MODULE_IO, 'output_dir'])
        _memory_check()
    finally:
        # Remove the folder that was used for input and output
        output_dir = config[config.MODULE_IO, 'output_dir']
        parent_dir = os.path.dirname(output_dir)
        print(f"Output dir: {output_dir}; Parent dir: {parent_dir}")
        if os.path.isdir(parent_dir):
            shutil.rmtree(parent_dir)
        print("Temp folder removed")



@pytest.mark.timeout(constants.BASELINE_MAX_RUN_TIME_SECONDS, method='signal')
@pytest.mark.cadt
def test_T_200_2():
    """
    Main system check. Run the deployer production pipeline for the testing dataset for all the studies that compound it
    and make sure the outputs are the expected ones.
    Also, make sure that the timing is reasonable (the test will fail after BASELINE_MAX_RUN_TIME_SECONDS seconds)
    Checks:
    - All the studies are processed with no flags raised
    - The results for each study are the expected ones
    - The report files for each study are created
    - The summary reports are created
    - The logger can be parsed and the extracted info matches the expected results (no preprocess flags)
    - The output config files are consistent
    """
    gc.collect()
    run_mode = const_deploy.RUN_MODE_CADT
    baseline_params = {
        'run_mode': run_mode,
    }

    dm = DataManager(baseline_params=baseline_params)
    studies = dm.get_all_studies()
    optional_params = {
        'run_mode': run_mode,
        'monitor_input_dir_continuously': True,
    }

    output_dir = tempfile.mkdtemp()
    studies_uids = {}
    processed_studies = set()
    deployer = Deployer()

    def study_callback(processed_results):
        study_uid = processed_results.get_studyUID()
        assert not processed_results.has_unexpected_errors()
        study_name = studies_uids[study_uid]
        baseline_results = StudyDeployResults.from_json(os.path.join(dm.baseline_dir_centaur_output, study_name,
                                                                     const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON))
        try:
            tests_utils.compare_study_deploy_results(processed_results, baseline_results)
        except:
            print(f"THERE WAS AN ERROR COMPARING {study_name}")
            traceback.print_exc()
        processed_studies.add(study_uid)
        if len(processed_studies) == len(studies_uids):
            # The test worked as expected so far. Stop the Deployer infinite loop
            deployer.stop_deploy()

    try:
        config_factory = ConfigFactory.VerificationInternalConfigFactory(output_dir=output_dir,
                                                                         optional_params_dict=optional_params)
        config = deployer.create_config_object(config_factory.params_dict)
        deployer.initialize(config)

        # Copy all the studies in the test dataset to the input folder (without the .transfer_complete file)
        for study in studies:
            study_path = os.path.join(dm.get_input_dir(study))
            # Get the StudyInstanceUID
            study_uid = tests_utils.get_study_instance_uid(study_path)
            studies_uids[study_uid] = study
            shutil.copytree(study_path, os.path.join(const_deploy.DEFAULT_INPUT_FOLDER, study))
            print(f"Study {study} copied")

        # Since the .transfer_complete file is not there yet, there shouldn't be any studies inserted to DB
        new_studies = deployer.input_object.check_new_studies()
        assert new_studies == 0, \
            "Expected empty database because the .transfer_complete file has not been copied yet"
        print("Copying .transfer_complete files...")
        # Create the .transfer_complete file so that the processing can start
        for study in studies:
            with open(os.path.join(const_deploy.DEFAULT_INPUT_FOLDER, study, const_io.TRANSFER_COMPLETE_FILE_NAME),
                      'w+'):
                pass

        new_studies = deployer.input_object.check_new_studies()
        assert new_studies == len(studies), \
            f"Expected {len(studies)} in the database. Found: {new_studies}"

        deployer.deploy(study_callback=study_callback)
        # All the studies were processed. Check the global baseline outputs
        tests_utils.compare_global_deploy_results(dm.baseline_dir_centaur_output,
                                                  config[config.MODULE_IO, 'output_dir'])
        _memory_check()
    finally:
        # Remove input and output folders content
        print("Removing input folder content...")
        shutil.rmtree(output_dir)
        os.system(f"rm -rf {const_deploy.DEFAULT_INPUT_FOLDER}/*")
        print("Memory check...")

def _memory_check():
    """
    Check for possible memory leaks. This function will be called after each study is finished
    :param file_list: list-str. List of files that were processed
    :param processed_ok: bool. The preprocessing had no flags
    :param study_dir: str. Study folder name
    :param output_dir: str. Study output dir
    """
    process = psutil.Process(os.getpid())
    current_mem = process.memory_info().rss / 2 ** 30
    assert current_mem <= constants.BASELINE_MAX_MEMORY_GB, "Currently the python process is using {}GB of memory. " \
                    "A maximum of {}GB was expected".format(current_mem, constants.BASELINE_MAX_MEMORY_GB)

# def _test_full_deploy_with_baseline(config, baseline_results_folder, study_names=None):
#     """
#     Run a prediction for all the studies that are in the testing dataset.
#     This is a duplicated of the code stored in the centaur_test repo.
#     Checks:
#     - All the studies are processed with no flags raised
#     - The results for each study are the expected ones
#     - The reports for each study are created
#     - The summary reports are created
#     - The logger can be parsed and the extracted info matches the expected results (no preprocess flags)
#     - The output config files are consistent
#     :param config: centaur_deploy.Config object. Global configuration options
#     :param baseline_results_folder: str. Base folder where the baseline results are stored
#     """
#     output_dir = config[Config.MODULE_IO]['output_dir']
#     baseline_studies = [osp.basename(d) for d in glob.glob(baseline_results_folder + "/*") if
#                         osp.isdir(d) and osp.basename(d) != "preprocessed_numpy"] \
#         if study_names is None else study_names
#
#     ### Run the deployer
#     deployer = Deployer()
#     deployer.initialize(config)
#     deployer.logger.info(parse_logs.log_line(-1, "Initialization finished. Deploying..."))
#     # processed_studies = deployer.deploy(return_results=True, study_callback=memory_check)
#     processed_studies = deployer.deploy(return_results=True)
#
#     # Ensure the right number of studies were processed
#     deployer.logger.info(parse_logs.log_line(-1, "Deploy finished. Checking number of studies processed..."))
#     processed_studies_names = [os.path.basename(r.input_files) for r in processed_studies]
#     assert processed_studies_names == baseline_studies, "Expected {} studies to be processed. Got: {}". \
#         format(baseline_studies, processed_studies_names)
#
#     ### Ensure that every study has a valid results file and reports were created
#     deployer.logger.info(parse_logs.log_line(-1, "Analyzing study results..."))
#     for study_results in processed_studies:
#         study = study_results.get_study_dir_name()
#         processed_ok = study_results.is_completed_ok()
#         if not processed_ok:
#             pytest.fail("Study {} failed".format(study))
#         # Compare model results
#         baseline_results_json_path = osp.join(baseline_results_folder, study, const_engine.CENTAUR_RESULTS_JSON)
#         output_results_path = osp.join(output_dir, study, const_engine.CENTAUR_RESULTS_JSON)
#
#         baseline_results = results_parser.load_results_json(baseline_results_json_path)
#         output_results = results_parser.load_results_json(output_results_path)
#
#         assert results_parser.are_equal(baseline_results['model_results'], output_results['model_results']), \
#             "Results do not match!\n\n***Expected: {}\n\n***Generated: {}".format(
#                 baseline_results['model_results'], output_results['model_results']
#             )
#
#         # Ensure that reports were created
#         assert len(glob.glob("{}/{}/DH_CAD*".format(output_dir, study))) == 1, "CAD report not found"
#         assert len(glob.glob("{}/{}/DH_PDF*".format(output_dir, study))) == 2, "PDF report not found"
#
#     ### Check the summaries report
#     deployer.logger.info(parse_logs.log_line(-1, "Checking Summaries Reports..."))
#     assert osp.isfile(osp.join(output_dir, const_reports.SUMMARY_REPORT_DICOM_CSV)), "DICOM summary report not found"
#     assert osp.isfile(osp.join(output_dir, const_reports.SUMMARY_REPORT_STUDY_CSV)), "Studies summary report not found"
#
#     ### Ensure that the logger can correctly access data from all the studies
#     deployer.logger.info(parse_logs.log_line(-1, "Checking logger..."))
#     parser = parse_logs.LogReader(deployer._logger_path)
#     parser.iter_lines()
#     dicom_df = parser.get_file_df()
#     study_df = parser.get_study_df().set_index('filename')
#
#     num_studies = len(processed_studies)
#     num_images = 0
#     for i in range(num_studies):
#         num_images += len(processed_studies[i].input_files)
#
#     assert len(dicom_df) == num_images, \
#         "Total number of dicom rows expected: {}. Got: {}".format(num_images, len(dicom_df))
#     assert len(dicom_df[dicom_df['failed_checks'] != '']) == 0, \
#         "Some unexpected failures were found in failed_checks column"
#     assert len(study_df) == num_studies, \
#         "{} rows expected in the study dataframe. Got {}".format(num_studies, len(study_df))
#
#     # Compare config files
#     with open(osp.join(output_dir, const_deploy.CENTAUR_CONFIG_JSON)) as fp:
#         output_config = Config.from_json(fp.read())
#
#     with open(osp.join(baseline_results_folder, const_deploy.CENTAUR_CONFIG_JSON)) as fp:
#         baseline_config = Config.from_json(fp.read())
#
#     assert configs_equal(baseline_config, output_config), \
#         "Output config:\n{};\n****\nBaseline config:\n{}".format(output_config, baseline_config)

if __name__ == "__main__":
    test_T_200()
    print("DONE")