import os
import os.path as osp
import shutil
import tempfile
import pytest

from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from keras import backend as K

from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_test.config_factory import ConfigFactory
from deephealth_utils.misc import results_parser
import centaur_engine.helpers.helper_misc as helper_misc

from centaur_engine.engine import Engine
from centaur_engine.helpers import helper_model
from centaur_engine.models.model_selector import ModelSelector
from centaur_engine.preprocessor import Preprocessor

from centaur_deploy.deploys.config import Config
import centaur_deploy.constants as const
import centaur_test.utils as utils
from centaur_deploy.deploy import Deployer
from centaur_test.data_manager import DataManager

###########################
# AUX FUNCTIONS
###########################
@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    """
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    """
    record_xml_attribute("classname", "3_Integration")

def teardown_function(function):
    """
    Method that is called after each test function is called
    Args:
        function: function called
    """
    print(f"************ Called function {function} *******************")
    K.clear_session()

# def _run_pipeline_baseline_dataset_with_deployer(config):
#     """
#     Run a prediction for all the studies that are in a dataset.
#     It just checks that the results "look ok", it doesn't compare exact results.
#     Use the Deployer object
#     :param config: Config instance. Parameters that will be used when running the prediction trough the deployer
#     """
#     try:
#         data_manager = DataManager()
#         # Get the root folder for all the dicom files
#         dicom_folder = data_manager.data_studies_dir
#         output_dir = tempfile.mkdtemp("_cent_output")
#         studies = data_manager.get_all_studies()
#
#         config[Config.MODULE_IO, 'input_dir'] = dicom_folder
#         config[Config.MODULE_IO, 'output_dir'] = output_dir
#         deployer = Deployer()
#         deployer.initialize(config)
#
#         config_copy = config.copy()
#         deployer.deploy()
#
#         # Ensure that every study has a valid results file, including metadata
#         for study in studies:
#             utils.validate_study_output_folder(os.path.join(output_dir, study))
#
#         # Ensure that the config file was saved correctly and it was not modified
#         config_file_path = osp.join(output_dir, const.CENTAUR_CONFIG_JSON)
#         assert osp.isfile(config_file_path), "Config file path not found ({})".format(config_file_path)
#         with open(config_file_path, "r") as f:
#             s = f.read()
#             config = Config.from_json(s)
#         assert config == config_copy, "Config files do not match! Was the Config object modified?"
#     finally:
#         shutil.rmtree(output_dir)
#         print("{} removed".format(output_dir))


###########################
# TESTS
###########################
def test_T_169(run_mode):
    """
    Preprocess the valid studies in the dataset and predict the results (without using the Engine class)
    Check the results of each study "look valid", but not necessarily are the same that in the baseline
    """
    data_manager = DataManager()
    studies = data_manager.get_valid_studies(run_mode=run_mode)
    for study in studies:
        print("Running study {}...".format(study))
        dcm_files = data_manager.get_dicom_images(study)
        # Preprocess the files
        config = Config()
        pp = Preprocessor(config[Config.MODULE_ENGINE], logger=utils.get_testing_logger())

        passed = pp.preprocess(dcm_files)
        reasons = pp.get_failed_checks()
        assert passed, "Unexpected flags when preprocessing the files: {}".format(reasons)
        metadata_df = pp.metadata

        # Get model files
        centaur_model_version = helper_model.get_actual_version_number('model')
        thresholds_version = helper_model.get_actual_version_number(run_mode.lower())
        if run_mode == const.RUN_MODE_CADT:
            model = ModelSelector.select(centaur_model_version, thresholds_version, None)
        elif run_mode == const.RUN_MODE_CADX:
            model = ModelSelector.select(centaur_model_version, None, thresholds_version)
        else:
            raise ValueError(f"Unexpected run mode {run_mode}."
                             f" It should be one of {const.RUN_MODE_CADT}, {const.RUN_MODE_CADX}")
        # Predict
        model.set_pixel_data(pp.pixel_data)
        results = model(metadata_df)
        proc_info = results['proc_info']
        # Make sure that the results "look ok"
        utils.validate_dicom_results_study(results['dicom_results'], results['study_results'], proc_info, metadata_df)

# def test_T_3_I02():
#     """
#     Read a folder with several studies. Preprocess and predict them, without using the Engine class.
#     Ensure the results match exactly the expected in the baseline, both the preprocessing and the prediction
#     """
#     # Create logger
#     output_folder = tempfile.mkdtemp(suffix="_logger")
#     logger, logger_file = helper_misc.create_logger(output_folder, return_path=True)
#
#     try:
#         data_manager = DataManager()
#         studies = data_manager.get_all_studies()
#         num_files = 0
#
#         for study in studies:
#             print ("Processing study {}...".format(study))
#             # Read DICOM files
#             file_list = data_manager.get_dicom_images(study)
#             num_files += len(file_list)
#             # Preprocess study
#             config = Config.ProductionConfig()
#             pp = Preprocessor(config[Config.MODULE_ENGINE], logger=logger)
#             passed = pp.preprocess(file_list)
#             assert passed, "Unexpected flags when preprocessing the files: {}".format(pp.get_failed_checks())
#             # Make sure the numpy arrays processed are the same as the baseline
#             metadata_df = pp.get_metadata()
#             assert len(metadata_df) == len(file_list) , "Metadata not expected. Baseline: {}; Got: {}".format(file_list, metadata_df)
#             for i in range(len(metadata_df)):
#                 row = metadata_df.iloc[i]
#                 generated_arrays_for_image = pp.pixel_data[i]
#                 numpy_files_for_image = glob.glob("{}/preprocessed_numpy/{}/*.npy".format(data_manager.baseline_dir_centaur_output, row['SOPInstanceUID']))
#                 assert len(generated_arrays_for_image) == len(numpy_files_for_image), \
#                     "Expected {} arrays for dicom image {}. Got: {}". \
#                         format(len(numpy_files_for_image), row['SOPInstanceUID'], len(generated_arrays_for_image))
#
#                 for g in range(len(generated_arrays_for_image)):
#                     generated_array = generated_arrays_for_image[g]
#                     baseline_array = np.load(numpy_files_for_image[g])
#                     assert np.array_equal(generated_array,
#                                           baseline_array), "Numpy array inconsistent in file:\n{}".format(row)
#
#             # Run model prediction and compare the results with the expected output
#             centaur_model_version = helper_model.get_agg_real_model_version()
#             model = ModelSelector.select(centaur_model_version)
#             model.set_pixel_data(pp.get_pixel_data())
#             predicted_results = model(metadata_df)
#
#             Engine.results_post_processing(predicted_results, pp.get_metadata())
#             utils.compare_model_results_to_baseline(study, predicted_results)
#
#         # Ensure that we can parse accurately the logger info
#         parser = parse_logs.LogReader(logger_file)
#         parser.iter_lines()
#         dicom_df = parser.get_file_df()
#         assert len(dicom_df) == num_files, "Expected {} rows in the logger metadata. Got: {}".format(num_files, len(dicom_df))
#         study_df = parser.get_study_df()
#         assert len(study_df) == len(studies), "Expected {} studies in the logger. Got: {}".format(len(studies), len(study_df))
#     finally:
#         if osp.isdir(output_folder):
#             shutil.rmtree(output_folder)

def test_T_170(run_mode):
    """
    Read a folder with several studies (valid ones). Preprocess and predict them, using the Engine class.
    Ensure the results match exactly the expected in the baseline, both the preprocessing and the prediction
    """
    # Create logger
    output_folder = tempfile.mkdtemp(suffix="_logger")
    try:
        logger, logger_file = helper_misc.create_logger(output_folder, return_path=True)
        data_manager = DataManager()
        data_manager.set_baseline_params(run_mode)
        studies = data_manager.get_valid_studies(run_mode=run_mode)

        config = Config()
        config[Config.MODULE_ENGINE, 'run_mode'] = run_mode


        # Get model files
        centaur_model_version = helper_model.get_actual_version_number('model')
        thresholds = helper_model.get_actual_version_number(run_mode.lower())
        if run_mode == const.RUN_MODE_CADT:
            model = ModelSelector.select(centaur_model_version, thresholds, None)
        elif run_mode == const.RUN_MODE_CADX:
            model = ModelSelector.select(centaur_model_version, None, thresholds)
        else:
            raise ValueError(f"Unexpected run mode {run_mode}."
                             f" It should be one of {const.RUN_MODE_CADT}, {const.RUN_MODE_CADX}")

        # Set operating points (in CADt mode only)
        if run_mode == const.RUN_MODE_CADT:
            op_point_key = config[Config.MODULE_ENGINE, 'cadt_operating_point_key']
            op_point_dxm = model.config['cadt_thresholds'][DicomTypeMap.DXM][op_point_key]
            op_point_dbt = model.config['cadt_thresholds'][DicomTypeMap.DBT][op_point_key]
            config.set_cadt_operating_points(op_point_dxm, op_point_dbt, explicitly_set=False)


        for study in studies:
            file_list = data_manager.get_dicom_images(study)
            # Build the Engine object
            pp = Preprocessor(config[Config.MODULE_ENGINE], logger=logger)
            engine = Engine(preprocessor=pp, model=model, reuse=False, logger=logger,
                            config=config[Config.MODULE_ENGINE])
            engine.set_file_list(file_list)
            engine.set_output_dir(output_folder)

            # Validate file preprocessing
            valid = engine.preprocess()
            assert valid, "Study did NOT pass the Preprocessor validations. Errors: {}".format(engine.preprocessor.get_failed_checks())
            df = engine.preprocessor.get_metadata()
            assert len(df) == len(file_list), "Expected {} rows in the metadata. Got: {}".format(len(file_list), len(df))

            # Read the baseline results
            baseline_results = StudyDeployResults.from_json(os.path.join(data_manager.baseline_dir_centaur_output,
                                                                         study,
                                                                         const.CENTAUR_STUDY_DEPLOY_RESULTS_JSON))
            # Predict the results
            engine.evaluate()

            # Compare to baseline results
            assert results_parser.are_equal(engine.results_raw, baseline_results.results_raw)
            assert results_parser.are_equal(engine.results, baseline_results.results)

            # Clean temp files
            engine.clean()
            if not config[Config.MODULE_ENGINE, 'save_to_ram']:
                # Make sure temp files were removed
                for f in df['np_paths']:
                    assert not osp.isfile(f), "Temp file {} was not removed".format(f)
    finally:
        if osp.isdir(output_folder):
            shutil.rmtree(output_folder)

# def test_T_3_I03():
#     """
#     Full deploy for all the studies in a dataset. No PACS - No reports.
#     Only check the results "look ok" (do not compare to baseline)
#     """
#     # Set only parameters of interest. The rest will be None
#     config = Config.ProductionConfig()
#     config[Config.MODULE_DEPLOY, 'debug'] = True
#     config[Config.MODULE_IO, 'pacs'] = False
#     config[Config.MODULE_REPORTS, 'reports'] = []
#     config[Config.MODULE_DEPLOY, 'exit_on_error'] = True
#
#     _run_pipeline_baseline_dataset_with_deployer(config)

# def test_T_3_I04():
#     """
#     Full deploy for all the studies in a dataset. No PACS - All reports.
#     This is the equivalent to system test T_S01 that checks the main deploy elements.
#     The results are compared to the baseline
#     """
#     dm = DataManager()
#     input_dir = dm.data_studies_dir
#     output_dir = tempfile.mkdtemp(prefix="temp_output_folder_")
#
#     # Set only parameters of interest. The rest will be None
#     config = Config.ProductionConfig()
#     config[Config.MODULE_IO, 'input_dir'] = input_dir
#     config[Config.MODULE_IO, 'output_dir'] = output_dir
#
#     try:
#         utils.test_full_deploy(config, dm.baseline_dir_centaur_output)
#     finally:
#         shutil.rmtree(output_dir)
#         print("Temp folder removed")



if __name__ == "__main__":
    test_T_170("CADt")
    print("OK!")