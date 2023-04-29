import copy
import glob
import os
import pytest
import json
import numpy as np
import shutil
import tempfile

from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_test.config_factory import ConfigFactory
from deephealth_utils.misc import results_parser
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
import deephealth_utils.data.parse_logs as parse_logs
from centaur_deploy.deploys.config import Config
from centaur_engine.helpers import helper_model, helper_results_processing, helper_preprocessor
from centaur_engine.helpers.helper_category import CategoryHelper
import centaur_engine.helpers.helper_results_processing as results_processing
from centaur_engine.models.model_selector import ModelSelector
import centaur_engine.constants as const_eng
import centaur_deploy.constants as const_deploy
from centaur_test.data_manager import DataManager
from centaur_deploy.deploy import Deployer

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
    record_xml_attribute("classname", "3_Unit")


###################################################
# TESTS
###################################################
# def test_T_3_U01():
#     '''
#     Ensure a prediction is correct for a Dxm study that contains all the required DICOM files/attributes.
#     Metadata will be also pre-loaded from the baseline dataset
#     '''
#     data_manager = DataManager()
#     centaur_model_version = helper_model.get_agg_real_model_version()
#     centaur_model = ModelSelector.select(centaur_model_version)
#     p = os.path.join(data_manager.baseline_dir_centaur_output, DataManager.STUDY_01_HOLOGIC, const_eng.CENTAUR_RESULTS_JSON)
#     assert os.path.isfile(p), "Results file not found: {}".format(p)
#     with open(p, 'rb') as f:
#         baseline = json.load(f)
#     metadata_df = pd.read_json(baseline['metadata'])
#
#     # Convert relative paths to absolute paths
#     metadata_df['dcm_path'] = (data_manager.data_dir + "/") + metadata_df['dcm_path'].astype(str)
#     metadata_df['np_paths'] = (data_manager.baseline_dir + "/") + metadata_df['np_paths'].astype(str)
#     # Predict the results
#     predicted_results = centaur_model(metadata_df)
#     # Compare to baseline
#     compare_model_results_to_baseline(DataManager.STUDY_01_HOLOGIC, predicted_results)




def test_T_162(run_mode):
    """
    Test slice number in a 3D image
    """
    study_deploy_results = _get_results_example(run_mode)
    results_orig = study_deploy_results.results
    model_results_orig = results_parser.get_model_results(results_orig)
    # proc_info_orig = results_parser.get_proc_info(results_orig)
    metadata_df = study_deploy_results.metadata

    config = Config()
    max_bbxs_displayed_total = config[Config.MODULE_ENGINE, 'results_pproc_max_bbxs_displayed_total']
    max_bbxs_displayed_intermediate = config[Config.MODULE_ENGINE, 'results_pproc_max_bbxs_displayed_intermediate']
    min_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_min_relative_bbx_size_allowed']
    max_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_max_relative_bbx_size_allowed']

    # OK case (after capping the number of bounding boxes per image)
    if run_mode == const_deploy.RUN_MODE_CADX:
        model_results = helper_results_processing.cap_bbx_number_per_image(model_results_orig, max_bbxs_displayed_total,
                                                                       max_bbxs_displayed_intermediate)
    else:
        model_results = copy.copy(model_results_orig)
    helper_results_processing.model_results_sanity_checks(model_results, study_deploy_results.results_raw,
                                                          metadata_df, max_bbxs_displayed_total,
                                                          min_relative_bbx_size_allowed, max_relative_bbx_size_allowed,
                                                          run_mode=run_mode)

    instance_uid_3d = metadata_df[metadata_df.SOPClassUID == DicomTypeMap.get_dbt_class_id()].iloc[0]['SOPInstanceUID']

    # Remove slice number from proc_info
    # proc_info_orig = copy.deepcopy(proc_info)
    del model_results['proc_info'][instance_uid_3d]['num_slices']
    try:
        helper_results_processing.model_results_sanity_checks(model_results, study_deploy_results.results_raw,
                                                              metadata_df, max_bbxs_displayed_total,
                                                              min_relative_bbx_size_allowed,
                                                              max_relative_bbx_size_allowed,
                                                              run_mode=run_mode)
        pytest.fail("Error expected (no num_slice in proc_info)")
    except AssertionError:
        pass

    # Set a wrong slice number
    model_results['dicom_results'][instance_uid_3d]['none'][0]['slice'] = -1
    try:
        helper_results_processing.model_results_sanity_checks(model_results, study_deploy_results.results_raw,
                                                              metadata_df, max_bbxs_displayed_total,
                                                              min_relative_bbx_size_allowed,
                                                              max_relative_bbx_size_allowed,
                                                              run_mode=run_mode)
        pytest.fail("Error expected (slice of out bounds)")
    except AssertionError:
        pass

    model_results['dicom_results'][instance_uid_3d]['none'][0]['slice'] = 10
    try:
        helper_results_processing.model_results_sanity_checks(model_results, study_deploy_results.results_raw,
                                                              metadata_df, max_bbxs_displayed_total,
                                                              min_relative_bbx_size_allowed,
                                                              max_relative_bbx_size_allowed,
                                                              run_mode=run_mode)
        pytest.fail("Error expected (slice of out bounds)")
    except AssertionError:
        pass

def test_T_226(run_mode):
    """
    Ensure an empty study can be processed
    Args:
        run_mode (str): CADt, CADx...
    """
    baseline_params = {'run_mode': run_mode}
    output_dir = tempfile.mkdtemp(prefix="temp_output_folder_")
    input_dir = tempfile.mkdtemp(prefix="temp_input_folder_")
    # Create an empty folder as input
    study_name = "EMPTY"
    os.mkdir(os.path.join(input_dir, study_name))
    config_factory = ConfigFactory.VerificationInternalConfigFactory(
        input_dir=input_dir, output_dir=output_dir,
        optional_params_dict=baseline_params)

    deployer = Deployer()

    config = deployer.create_config_object(config_factory.params_dict)
    deployer.initialize(config)

    # Run the empty study
    deployer.deploy()

    # Make sure there is results_full.json file generated
    results_file = os.path.join(output_dir, study_name, const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(results_file), "Results file not generated"

    # Try to create a StudyDeployResults object and check it's valid
    results = StudyDeployResults.from_json(results_file)
    assert results.input_files == [], f"Expected empty list for input files. Got: {results.input_files}"
    assert results.input_dir == os.path.join(input_dir, study_name), \
           f"Input dir doesn't match (Expected: os.path.join(input_dir, study_name); Got: {results.input_dir}"
    assert results.get_study_dir_name() == study_name, \
        f"Study name doesn't match. Expected {study_name}; Got: {results.get_study_dir_name()}"
    assert results.results_raw is None, f"results_raw expected to be None. Got: {results.results}"
    assert results.results is None, f"Results expected to be None. Got: {results.results}"
    assert results.get_error_message(), "Error message was expected"



# def test_T_3_U02():
#     """
#     Test the categories in bounding boxes are not higher than their corresponding laterality/study
#     """
#     # Load a results.json example file
#     original_study_results =  _get_results_example()
#     original_model_results = original_study_results.results
#     model_results = results_parser.get_model_results(original_model_results)
#     dicom_results = results_parser.get_dicom_results(original_model_results)
#     study_results = results_parser.get_study_results(original_model_results)
#
#     # Force the category for the lateralities/studies
#     study_results['L']['category'] = CategoryHelper.INTERMEDIATE
#     study_results['R']['category'] = CategoryHelper.HIGH
#     study_results['total']['category'] = CategoryHelper.HIGH
#
#     # Get the first image with a 'left' laterality
#     metadata = original_study_results.metadata
#     row = metadata[metadata['ImageLaterality'] == 'L'].iloc[0]
#     # Force a higher category for the first bbx in that image
#     instance_uid = row['SOPInstanceUID']
#     dicom_results[instance_uid]['none'][0]['category'] = CategoryHelper.HIGH
#
#     # Force a higher category for the first bbx in an image that belongs to other laterality (to ensure it doesn't change)
#     other_instance_uid = metadata[metadata['ImageLaterality'] == 'R'].iloc[0]['SOPInstanceUID']
#     dicom_results[other_instance_uid]['none'][0]['category'] = CategoryHelper.HIGH
#
#     model_results = results_processing.cap_bounding_box_category(model_results, metadata)
#
#     dicom_results = model_results['dicom_results']
#     cat = dicom_results[instance_uid]['none'][0]['category']
#     assert cat <= CategoryHelper.INTERMEDIATE, "Expected category {} in bounding box because the laterality has that value. Got: {}". \
#         format(CategoryHelper.INTERMEDIATE, cat)
#
#     cat = dicom_results[other_instance_uid]['none'][0]['category']
#     assert cat == CategoryHelper.HIGH, "The category for a bounding box changed, when it was in another laterality"
#
#     # SCENARIO 2: the score for one laterality is higher than the score for the study
#     original_model_results = _get_results_example()
#     model_results = results_parser.get_model_results(original_model_results)
#     dicom_results = results_parser.get_dicom_results(original_model_results)
#     study_results = results_parser.get_study_results(original_model_results)
#
#     study_category = CategoryHelper.INTERMEDIATE
#     study_results['total']['category'] = study_category
#     study_results['R']['category'] = CategoryHelper.HIGH
#
#     model_results = results_processing.cap_bounding_box_category(model_results, metadata)
#     study_results = model_results['study_results']
#     dicom_results = model_results['dicom_results']
#     assert study_results['R']['category'] <= study_category, \
#         "The category for laterality 'R' was {} when the study was {}".format(study_results['R']['category'],
#                                                                               study_category)
#
#     # Make sure that no bbx has a higher category than the study
#     for instance_uid, transforms in dicom_results.items():
#         for transform, bbxs in transforms.items():
#             for i in range(len(bbxs)):
#                 bbx = bbxs[i]
#                 cat = bbx['category']
#                 assert cat <= study_category, "Found bbx {}/{}/{} with category {}. Expected: {}".format(
#                     instance_uid, transform, i, cat, study_category)


# def test_T_3_U03():
#     """
#     Test the maximum number of displayed bounding boxes per image
#     """
#     results = _get_results_example()
#     model_results = results_parser.get_model_results(results)
#     dicom_results = results_parser.get_dicom_results(results)
#
#     config = Config.ProductionConfig()
#     max_bbxs_displayed_total = config[Config.MODULE_ENGINE, 'results_pproc_max_bbxs_displayed_total']
#     max_bbxs_displayed_intermediate = config[Config.MODULE_ENGINE, 'results_pproc_max_bbxs_displayed_intermediate']
#
#     instance_uid = list(dicom_results.keys())[0]
#     bbxs = [
#         {'category': CategoryHelper.HIGH, 'score': 0},
#         {'category': CategoryHelper.HIGH, 'score': 0},
#         {'category': CategoryHelper.HIGH, 'score': 0},
#         {'category': CategoryHelper.HIGH, 'score': 0},
#         {'category': CategoryHelper.HIGH, 'score': 0},
#         {'category': CategoryHelper.INTERMEDIATE, 'score': 0},
#         {'category': CategoryHelper.LOW, 'score': 0},
#     ]
#     dicom_results[instance_uid]['none'] = bbxs
#     model_results = results_processing.cap_bbx_number_per_image(model_results, max_bbxs_displayed_total,
#                                                                 max_bbxs_displayed_intermediate)
#     dicom_results = model_results['dicom_results']
#
#     new_bbxs = model_results['dicom_results'][instance_uid]['none']
#     expected = [{'category': CategoryHelper.HIGH, 'score': 0}] * max_bbxs_displayed_total
#     assert new_bbxs == expected
#
#     # Scenario 2
#     bbxs = [
#         {'category': CategoryHelper.INTERMEDIATE, 'score': 0},
#         {'category': CategoryHelper.INTERMEDIATE, 'score': 0},
#         {'category': CategoryHelper.INTERMEDIATE, 'score': 0},
#         {'category': CategoryHelper.INTERMEDIATE, 'score': 0},
#         {'category': CategoryHelper.LOW, 'score': 0},
#     ]
#     model_results['dicom_results'][instance_uid]['none'] = bbxs
#     model_results = results_processing.cap_bbx_number_per_image(model_results, max_bbxs_displayed_total,
#                                                                 max_bbxs_displayed_intermediate)
#     new_bbxs = model_results['dicom_results'][instance_uid]['none']
#     expected = [{'category': CategoryHelper.INTERMEDIATE, 'score': 0}] * max_bbxs_displayed_intermediate
#     assert expected == new_bbxs
#
#     # Scenario 3
#     bbxs = [
#         {'category': CategoryHelper.HIGH, 'score': 0},
#         {'category': CategoryHelper.INTERMEDIATE, 'score': 0},
#         {'category': CategoryHelper.LOW, 'score': 0},
#     ]
#     model_results['dicom_results'][instance_uid]['none'] = bbxs
#     model_results = results_processing.cap_bbx_number_per_image(model_results, max_bbxs_displayed_total,
#                                                                 max_bbxs_displayed_intermediate)
#     new_bbxs = model_results['dicom_results'][instance_uid]['none']
#     expected = [
#         {'category': CategoryHelper.HIGH, 'score': 0},
#         {'category': CategoryHelper.INTERMEDIATE, 'score': 0},
#     ]
#     assert expected == new_bbxs
#
#     # Scenario 4
#     bbxs = [
#         {'category': CategoryHelper.LOW, 'score': 0.1},
#         {'category': CategoryHelper.LOW, 'score': 0.05},
#         {'category': CategoryHelper.LOW, 'score': 0.05},
#     ]
#     model_results['dicom_results'][instance_uid]['none'] = bbxs
#     model_results = results_processing.cap_bbx_number_per_image(model_results, max_bbxs_displayed_total,
#                                                                 max_bbxs_displayed_intermediate)
#     new_bbxs = model_results['dicom_results'][instance_uid]['none']
#     expected = [{'category': CategoryHelper.LOW, 'score': 0.1}]
#     assert new_bbxs == expected
#
#     # Scenario 5. Bbxs unsorted. There should be an AssertionError
#     bbxs = [
#         {'category': CategoryHelper.HIGH, 'score': 0.83},
#         {'category': CategoryHelper.HIGH, 'score': 0.82},
#         {'category': CategoryHelper.HIGH, 'score': 0.81},
#         {'category': CategoryHelper.HIGH, 'score': 0.85},
#     ]
#     model_results['dicom_results'][instance_uid]['none'] = bbxs

@pytest.mark.cadx
def test_T_3_U04():
    """
    Test max bounding box sizes
    """
    study_deploy_results = _get_results_example(const_deploy.RUN_MODE_CADX)
    results = study_deploy_results.results
    model_results = results_parser.get_model_results(results)
    dicom_results = results_parser.get_dicom_results(results)
    instance_uid = list(dicom_results.keys())[0]

    config = Config()
    min_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_min_relative_bbx_size_allowed']
    max_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_max_relative_bbx_size_allowed']

    # Use only the first bbox
    bbx = dicom_results[instance_uid]['none'][0]
    dicom_results[instance_uid]['none'] = [bbx]
    model_results['dicom_results'] = {instance_uid: dicom_results[instance_uid]}

    size_y, size_x = [2457, 1890]
    # The max size will be relative to size_y because it's the biggest
    max_size_expected = int(size_y * max_relative_bbx_size_allowed)
    results_parser.get_proc_info(results)[instance_uid]['original_shape'] = [size_y, size_x]

    # Bbx too big in x dimension
    coords = [1, 2000, 1801, 2451]
    bbx['coords'] = coords
    despl_x = coords[2] - coords[0] - max_size_expected
    coords_expected = [
        coords[0] + (despl_x // 2),
        coords[1],
        coords[2] - (despl_x // 2),
        coords[3],
    ]
    if coords_expected[2] - coords_expected[0] > max_size_expected:
        # Round error
        coords_expected[0] += 1

    model_results = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                     max_relative_bbx_size_allowed)
    bbx = model_results['dicom_results'][instance_uid]['none'][0]
    assert bbx['coords'] == coords_expected

    # Bbx too big in y dimension
    coords = [1, 100, 500, 2451]
    bbx['coords'] = coords
    despl_y = coords[3] - coords[1] - max_size_expected
    coords_expected = [
        coords[0],
        coords[1] + (despl_y // 2),
        coords[2],
        coords[3] - (despl_y // 2),
    ]
    if coords_expected[3] - coords_expected[1] > max_size_expected:
        # Round error
        coords_expected[1] += 1
    model_results = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                     max_relative_bbx_size_allowed)
    bbx = model_results['dicom_results'][instance_uid]['none'][0]
    assert bbx['coords'] == coords_expected

    # Bbx too big in both dimensions
    coords = [10, 20, 1700, 2000]
    bbx['coords'] = coords
    despl_x = coords[2] - coords[0] - max_size_expected
    despl_y = coords[3] - coords[1] - max_size_expected
    coords_expected = [
        coords[0] + (despl_x // 2),
        coords[1] + (despl_y // 2),
        coords[2] - (despl_x // 2),
        coords[3] - (despl_y // 2),
    ]
    if coords_expected[2] - coords_expected[0] > max_size_expected:
        # Round error
        coords_expected[0] += 1
    if coords_expected[3] - coords_expected[1] > max_size_expected:
        # Round error
        coords_expected[1] += 1
    model_results = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                     max_relative_bbx_size_allowed)
    bbx = model_results['dicom_results'][instance_uid]['none'][0]
    assert bbx['coords'] == coords_expected


@pytest.mark.cadx
def test_T_3_U05():
    """
    Test min bounding box sizes (outbounds)
    """
    results = _get_results_example(const_deploy.RUN_MODE_CADX).results

    config = Config()
    min_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_min_relative_bbx_size_allowed']
    max_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_max_relative_bbx_size_allowed']

    model_results = results_parser.get_model_results(results)
    dicom_results = results_parser.get_dicom_results(results)

    # Use only the first bbox
    instance_uid = list(dicom_results.keys())[0]
    bbx = dicom_results[instance_uid]['none'][0]
    dicom_results[instance_uid]['none'] = [bbx]
    model_results['dicom_results'] = {instance_uid: dicom_results[instance_uid]}

    size_y, size_x = [2457, 1890]
    # The max size will be relative to size_x because it's the lowest
    min_size_expected = int(size_x * min_relative_bbx_size_allowed)
    results_parser.get_proc_info(results)[instance_uid]['original_shape'] = [size_y, size_x]

    # Bbx too small in x dimension (including out of bounds)
    coords = [1, 2000, 5, 2451]
    bbx['coords'] = coords
    coords_expected = [
        0,
        coords[1],
        min_size_expected,
        coords[3],
    ]
    model_results = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                     max_relative_bbx_size_allowed)
    bbx = model_results['dicom_results'][instance_uid]['none'][0]
    assert bbx['coords'] == coords_expected

    # Bbx too small in y dimension (including out of bounds)
    coords = [1, 0, 500, 10]
    bbx['coords'] = coords
    coords_expected = [
        1,
        0,
        500,
        min_size_expected,
    ]
    model_results = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                     max_relative_bbx_size_allowed)
    bbx = model_results['dicom_results'][instance_uid]['none'][0]
    assert bbx['coords'] == coords_expected


@pytest.mark.cadx
def test_T_3_U06():
    """
    Test mixed too big/small bounding box sizes (inbounds)
    """
    results = _get_results_example(const_deploy.RUN_MODE_CADX).results

    config = Config()
    min_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_min_relative_bbx_size_allowed']
    max_relative_bbx_size_allowed = config[Config.MODULE_ENGINE, 'results_pproc_max_relative_bbx_size_allowed']

    model_results = results_parser.get_model_results(results)
    dicom_results = results_parser.get_dicom_results(results)

    # Use only the first bbox
    instance_uid = list(dicom_results.keys())[0]
    bbx = dicom_results[instance_uid]['none'][0]
    dicom_results[instance_uid]['none'] = [bbx]
    model_results['dicom_results'] = {instance_uid: dicom_results[instance_uid]}

    size_y, size_x = [2457, 1890]
    results_parser.get_proc_info(results)[instance_uid]['original_shape'] = [size_y, size_x]

    # Ok scenario (bbx size unchanged)
    coords = [1, 2000, 500, 2451]
    dicom_results[instance_uid]['none'][0]['coords'] = coords
    model_results_test = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                          max_relative_bbx_size_allowed)
    assert dicom_results[instance_uid]['none'][0]['coords'] == bbx['coords']

    # The max size will be relative to size_y because it's the biggest (and viceversa)
    max_size_expected = int(size_y * max_relative_bbx_size_allowed)
    min_size_expected = int(size_x * min_relative_bbx_size_allowed)

    # Bbx too big in x dimension and too small in y dimension (inbounds)
    coords = [1, 1000, 1801, 1010]
    dicom_results[instance_uid]['none'][0]['coords'] = coords
    despl_x = coords[2] - coords[0] - max_size_expected
    despl_y = min_size_expected - (coords[3] - coords[1])
    coords_expected = [
        coords[0] + (despl_x // 2),
        coords[1] - (despl_y // 2),
        coords[2] - (despl_x // 2),
        coords[3] + (despl_y // 2),
    ]
    if coords_expected[2] - coords_expected[0] > max_size_expected:
        # Round error
        coords_expected[0] += 1
    if coords_expected[3] - coords_expected[1] < min_size_expected:
        # Round error
        coords_expected[1] -= 1
    model_results_test = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                          max_relative_bbx_size_allowed)
    assert model_results_test['dicom_results'][instance_uid]['none'][0]['coords'] == coords_expected

    # Bbx too big in y dimension and too small in x dimension (inbounds)
    coords = [1000, 10, 1003, 2010]
    dicom_results[instance_uid]['none'][0]['coords'] = coords
    despl_x = min_size_expected - (coords[2] - coords[0])
    despl_y = coords[3] - coords[1] - max_size_expected
    coords_expected = [
        coords[0] - (despl_x // 2),
        coords[1] + (despl_y // 2),
        coords[2] + (despl_x // 2),
        coords[3] - (despl_y // 2),
    ]
    if coords_expected[3] - coords_expected[1] > max_size_expected:
        # Round error
        coords_expected[1] += 1
    if coords_expected[2] - coords_expected[0] < min_size_expected:
        # Round error
        coords_expected[0] -= 1
    model_results_test = results_processing.fix_bbxs_size(model_results, min_relative_bbx_size_allowed,
                                                          max_relative_bbx_size_allowed)

    assert model_results_test['dicom_results'][instance_uid]['none'][0]['coords'] == coords_expected


# def test_T_3_U08():
#     """
#     2D/3D bounding boxes
#     1. Overlapping boxes (a dxm and dbt)
#     2. Non-overlapping boxes (a dxm and dbt)
#     3. Missing dbt
#     4. Two boxes in dxm that overlap with one box in dbt, with dbt as the highest score
#     """
#     data_manager = DataManager()
#
#     centaur_model_version = helper_model.get_agg_real_model_version()
#     centaur_model = ModelSelector.select(centaur_model_version)
#
#     # p = os.path.join(baseline_folder, DataManager.STUDY_03_DBT_HOLOGIC,
#     #                  const_eng.CENTAUR_RESULTS_JSON)
#     # with open(p, 'r') as f:
#     #     results_orig = json.load(f)
#     study_deploy_results = _get_results_example()
#
#     results = copy.deepcopy(study_deploy_results.results)
#     model_results = results_parser.get_model_results(results)
#     dicom_results = results_parser.get_dicom_results(results)
#     metadata_df = study_deploy_results.metadata
#
#     laterality = 'R'
#     view = 'CC'
#     instance_uid_2d = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dxm_class_id()) &
#                                   (metadata_df['ImageLaterality'] == laterality) &
#                                   (metadata_df['ViewPosition'] == view)].iloc[0]['SOPInstanceUID']
#
#     instance_uid_3d = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dbt_class_id()) &
#                                   (metadata_df['ImageLaterality'] == laterality) &
#                                   (metadata_df['ViewPosition'] == view)].iloc[0]['SOPInstanceUID']
#
#     p = "{}/preprocessed_numpy/**/{}".format(data_manager.baseline_dir_centaur_output, instance_uid_3d)
#     temp_folder = glob.glob(p, recursive=True)
#     assert len(temp_folder) == 1, "Temp folder not found: " + p
#     temp_folder = os.path.dirname(temp_folder[0])
#     slice_data_path = "{}/synth/synth_slice_map.npy".format(temp_folder)
#     assert os.path.isfile(slice_data_path), "Synth mapping not found: {}" + slice_data_path
#     slice_map = np.load(slice_data_path)
#     ix = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_3d].iloc[0].name
#
#     # 1. overlapping boxes (one dxm, one dbt), test that the outcome only picks the higher score box
#     dicom_results = {k: dicom_results[k] for k in (instance_uid_2d, instance_uid_3d)}
#     dicom_results_copy = copy.deepcopy(dicom_results)
#     dicom_results = copy.deepcopy(dicom_results_copy)
#     model_results['dicom_results'] = dicom_results
#
#     coords_2d = [1, 1, 600, 600]
#     dicom_results[instance_uid_2d]['none'][0]['coords'] = coords_2d
#     dicom_results[instance_uid_2d]['none'][0]['score'] = 0.9
#     dicom_results[instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     test_slice_map = np.ones_like(slice_map) * 10
#     test_slice_map[coords_2d[1]:coords_2d[3], coords_2d[0]:coords_2d[2]] = 20 * np.ones_like(
#         test_slice_map[coords_2d[1]:coords_2d[3], coords_2d[0]:coords_2d[2]])
#
#     coords_3d = [200, 200, 700, 700]
#     dicom_results[instance_uid_3d]['none'][0]['coords'] = coords_3d
#     dicom_results[instance_uid_3d]['none'][0]['slice'] = 5
#     dicom_results[instance_uid_3d]['none'][0]['score'] = 0.1
#     dicom_results[instance_uid_3d]['none'][0]['category'] = 2
#     dicom_results[instance_uid_3d]['none'][0]['origin'] = 'dbt'
#     pixel_data = {ix: {'slice_map': test_slice_map}}
#     model_results_test = helper_results_processing.combine_dxm_dbt_boxes(model_results, metadata_df,
#                                                                          centaur_model.config, pixel_data)
#     dicom_results_test = model_results_test['dicom_results']
#     assert dicom_results_test[instance_uid_2d]['none'][0]['score'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['score'] == 0.9
#     assert dicom_results_test[instance_uid_2d]['none'][0]['slice'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['slice'] == 20
#     assert dicom_results_test[instance_uid_2d]['none'][0]['category'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['category'] == 3
#     assert dicom_results_test[instance_uid_2d]['none'][0]['coords'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['coords'] == coords_2d
#     assert dicom_results_test[instance_uid_2d]['none'][0]['origin'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['origin'] == 'dxm'
#
#     # 2. two non overlapping boxes (one dxm, one dbt), test that the outcome combines both with with correct origins and slices
#     dicom_results = copy.deepcopy(dicom_results_copy)
#     model_results['dicom_results'] = dicom_results
#
#     coords_2d = [1, 1, 600, 600]
#     dicom_results[instance_uid_2d]['none'][0]['coords'] = coords_2d
#     dicom_results[instance_uid_2d]['none'][0]['score'] = 0.9
#     dicom_results[instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     test_slice_map = np.ones_like(slice_map) * 10
#     test_slice_map[coords_2d[1]:coords_2d[3], coords_2d[0]:coords_2d[2]] = 20 * np.ones_like(
#         test_slice_map[coords_2d[1]:coords_2d[3], coords_2d[0]:coords_2d[2]])
#
#     coords_3d = [700, 700, 900, 900]
#     dicom_results[instance_uid_3d]['none'][0]['coords'] = coords_3d
#     dicom_results[instance_uid_3d]['none'][0]['slice'] = 5
#     dicom_results[instance_uid_3d]['none'][0]['score'] = 0.1
#     dicom_results[instance_uid_3d]['none'][0]['category'] = 2
#     dicom_results[instance_uid_3d]['none'][0]['origin'] = 'dbt'
#     ix = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_3d].iloc[0].name
#     pixel_data = {ix: {'slice_map': test_slice_map}}
#     model_results_test = helper_results_processing.combine_dxm_dbt_boxes(model_results, metadata_df,
#                                                                          centaur_model.config, pixel_data)
#     dicom_results_test = model_results_test['dicom_results']
#     assert are_equal([dicom_results[instance_uid_2d]['none'][0], dicom_results[instance_uid_3d]['none'][0]],
#                      dicom_results_test[instance_uid_2d]['none'])
#
#     # 3. a missing dbt, and make sure the scores don't change for the other
#     dicom_results = copy.deepcopy(dicom_results_copy)
#     model_results['dicom_results'] = dicom_results
#
#     coords_2d = [1, 1, 600, 600]
#     dicom_results[instance_uid_2d]['none'][0]['coords'] = coords_2d
#     dicom_results[instance_uid_2d]['none'][0]['score'] = 0.9
#     dicom_results[instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     pixel_data = {ix: {'slice_map': slice_map}}
#
#     dicom_results.pop(instance_uid_3d, None)
#     model_results_test = helper_results_processing.combine_dxm_dbt_boxes(model_results, metadata_df,
#                                                                          centaur_model.config, pixel_data)
#     dicom_results_test = model_results_test['dicom_results']
#
#     assert are_equal(dicom_results[instance_uid_2d]['none'], dicom_results_test[instance_uid_2d]['none'])
#
#     # 4. Two boxes in dxm that overlap with one box in dbt, with dbt as the highest score
#
#     dicom_results = copy.deepcopy(dicom_results_copy)
#     model_results['dicom_results'] = dicom_results
#
#     coords_2d = [1, 1, 300, 300]
#     dicom_results[instance_uid_2d]['none'][0]['coords'] = coords_2d
#     dicom_results[instance_uid_2d]['none'][0]['score'] = 0.2
#     dicom_results[instance_uid_2d]['none'][0]['category'] = 2
#     dicom_results[instance_uid_2d]['none'][0]['origin'] = 'dxm'
#
#     coords_2d_2 = [200, 200, 400, 400]
#     dicom_results[instance_uid_2d]['none'].append({})
#     dicom_results[instance_uid_2d]['none'][1]['coords'] = coords_2d_2
#     dicom_results[instance_uid_2d]['none'][1]['score'] = 0.3
#     dicom_results[instance_uid_2d]['none'][1]['category'] = 2
#     dicom_results[instance_uid_2d]['none'][1]['origin'] = 'dxm'
#
#     coords_3d = [1, 1, 600, 600]
#     dicom_results[instance_uid_3d]['none'][0]['coords'] = coords_3d
#     dicom_results[instance_uid_3d]['none'][0]['slice'] = 5
#     dicom_results[instance_uid_3d]['none'][0]['score'] = 0.9
#     dicom_results[instance_uid_3d]['none'][0]['category'] = 3
#     dicom_results[instance_uid_3d]['none'][0]['origin'] = 'dbt'
#     ix = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_3d].iloc[0].name
#     pixel_data = {ix: {'slice_map': slice_map}}
#     model_results_test = helper_results_processing.combine_dxm_dbt_boxes(model_results, metadata_df,
#                                                                          centaur_model.config, pixel_data)
#     dicom_results_test = model_results_test['dicom_results']
#     assert are_equal(dicom_results[instance_uid_3d]['none'],
#                      dicom_results_test[instance_uid_2d]['none'])
#     assert are_equal(dicom_results[instance_uid_3d]['none'],
#                      dicom_results_test[instance_uid_3d]['none'])
#
#
# def test_T_3_U09():
#     """
#     2D/3D bounding boxes
#     1. Test LMLO: two non overlapping boxes should appear flipped in the DBT results
#     2. Test a DxM with the same FoPUID and different shape is excluded
#     3. Test that two DxMs with the same FoPUID and shape are combined together
#     """
#     data_manager = DataManager()
#     baseline_folder = data_manager.baseline_dir_centaur_output
#
#     centaur_model_version = helper_model.get_agg_real_model_version()
#     centaur_model = ModelSelector.select(centaur_model_version)
#
#     p = os.path.join(baseline_folder, DataManager.STUDY_03_DBT_HOLOGIC,
#                      const_eng.CENTAUR_RESULTS_JSON)
#     with open(p, 'r') as f:
#         results_orig = json.load(f)
#
#     # 1. Test LMLO: two non overlapping boxes should appear flipped in the DBT results
#
#     results = copy.deepcopy(results_orig)
#     model_results = results_parser.get_model_results(results)
#     dicom_results = results_parser.get_dicom_results(results)
#     metadata_df = results_parser.get_metadata(results)
#
#     laterality = 'L'
#     view = 'MLO'
#     instance_uid_2d = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dxm_class_id()) &
#                                   (metadata_df['ImageLaterality'] == laterality) &
#                                   (metadata_df['ViewPosition'] == view)].iloc[0]['SOPInstanceUID']
#
#     instance_uid_3d_row = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dbt_class_id()) &
#                                       (metadata_df['ImageLaterality'] == laterality) &
#                                       (metadata_df['ViewPosition'] == view)]
#     instance_uid_3d = instance_uid_3d_row.iloc[0]['SOPInstanceUID']
#
#     p = "{}/preprocessed_numpy/**/{}".format(data_manager.baseline_dir_centaur_output, instance_uid_3d)
#     temp_folder = glob.glob(p, recursive=True)
#     assert len(temp_folder) == 1, "Temp folder not found: " + p
#     temp_folder = os.path.dirname(temp_folder[0])
#     slice_data_path = "{}/synth/synth_slice_map.npy".format(temp_folder)
#     assert os.path.isfile(slice_data_path), "Synth mapping not found: {}" + slice_data_path
#     slice_map = np.load(slice_data_path)
#
#     dicom_results = {k: dicom_results[k] for k in (instance_uid_2d, instance_uid_3d)}
#     dicom_results_copy = copy.deepcopy(dicom_results)
#     dicom_results = copy.deepcopy(dicom_results_copy)
#     model_results['dicom_results'] = dicom_results
#
#     ix = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_3d].iloc[0].name
#
#     coords_2d = [1, 1, 200, 200]
#     rotated_coords_2d = [slice_map.shape[1] - coords_2d[2] - 1,
#                          slice_map.shape[0] - coords_2d[3] - 1,
#                          slice_map.shape[1] - coords_2d[0] - 1,
#                          slice_map.shape[0] - coords_2d[1] - 1]
#     dicom_results[instance_uid_2d]['none'][0]['coords'] = coords_2d
#     dicom_results[instance_uid_2d]['none'][0]['score'] = 0.9
#     dicom_results[instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     test_slice_map = np.ones_like(slice_map) * 10
#     test_slice_map[coords_2d[1]:coords_2d[3], coords_2d[0]:coords_2d[2]] = 20 * np.ones_like(
#         test_slice_map[coords_2d[1]:coords_2d[3], coords_2d[0]:coords_2d[2]])
#
#     coords_3d = [1, 1, 200, 200]
#
#     dicom_results[instance_uid_3d]['none'][0]['coords'] = coords_3d
#     dicom_results[instance_uid_3d]['none'][0]['slice'] = 5
#     dicom_results[instance_uid_3d]['none'][0]['score'] = 0.1
#     dicom_results[instance_uid_3d]['none'][0]['category'] = 2
#     dicom_results[instance_uid_3d]['none'][0]['origin'] = 'dbt'
#     pixel_data = {ix: {'slice_map': test_slice_map}}
#     model_results_test = helper_results_processing.combine_dxm_dbt_boxes(model_results, metadata_df,
#                                                                          centaur_model.config, pixel_data)
#     dicom_results_test = model_results_test['dicom_results']
#
#     assert helper_preprocessor.image_must_be_rotated(instance_uid_3d_row['ImageLaterality'].values[0],
#                                                      instance_uid_3d_row['PatientOrientation'].values[0])
#     assert dicom_results_test[instance_uid_2d]['none'][0]['score'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['score'] == 0.9
#     assert dicom_results_test[instance_uid_2d]['none'][0]['slice'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['slice'] == 20
#     assert dicom_results_test[instance_uid_2d]['none'][0]['category'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['category'] == 3
#     assert dicom_results_test[instance_uid_2d]['none'][0]['coords'] == coords_2d
#     assert dicom_results_test[instance_uid_3d]['none'][0]['coords'] == rotated_coords_2d
#     assert dicom_results_test[instance_uid_2d]['none'][0]['origin'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['origin'] == 'dxm'
#
#     # 2. Test a DxM with the same FoPUID and different shape is excluded
#
#     results = copy.deepcopy(results_orig)
#     model_results = results_parser.get_model_results(results)
#     dicom_results = results_parser.get_dicom_results(results)
#     metadata_df = results_parser.get_metadata(results)
#
#     laterality = 'R'
#     view = 'CC'
#
#     instance_uid_2d = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dxm_class_id()) &
#                                   (metadata_df['ImageLaterality'] == laterality) &
#                                   (metadata_df['ViewPosition'] == view)].iloc[0]['SOPInstanceUID']
#
#     instance_uid_3d = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dbt_class_id()) &
#                                   (metadata_df['ImageLaterality'] == laterality) &
#                                   (metadata_df['ViewPosition'] == view)].iloc[0]['SOPInstanceUID']
#
#     p = "{}/preprocessed_numpy/**/{}".format(data_manager.baseline_dir_centaur_output, instance_uid_3d)
#     temp_folder = glob.glob(p, recursive=True)
#     assert len(temp_folder) == 1, "Temp folder not found: " + p
#     temp_folder = os.path.dirname(temp_folder[0])
#     slice_data_path = "{}/synth/synth_slice_map.npy".format(temp_folder)
#     assert os.path.isfile(slice_data_path), "Synth mapping not found: {}" + slice_data_path
#     slice_map = np.load(slice_data_path)
#
#     dicom_results = {k: dicom_results[k] for k in (instance_uid_2d, instance_uid_3d)}
#     dicom_results_copy = copy.deepcopy(dicom_results)
#     dicom_results = copy.deepcopy(dicom_results_copy)
#     model_results['dicom_results'] = dicom_results
#
#     ix = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_3d].iloc[0].name
#
#     coords_2d = [1, 1, 200, 200]
#     dicom_results[instance_uid_2d]['none'][0]['coords'] = coords_2d
#     dicom_results[instance_uid_2d]['none'][0]['score'] = 0.5
#     dicom_results[instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     rotated_coords_2d = [slice_map.shape[1] - coords_2d[2] - 1,
#                          slice_map.shape[0] - coords_2d[3] - 1,
#                          slice_map.shape[1] - coords_2d[0] - 1,
#                          slice_map.shape[0] - coords_2d[1] - 1]
#
#     dup_instance_uid_2d = 'dup_instance_uid_2d'
#     dup_for_uid_2d = 'dup_for_uid_2d'
#     dicom_results[dup_instance_uid_2d] = copy.deepcopy(dicom_results[instance_uid_2d])
#
#     dup_coords_2d = [100, 100, 300, 300]
#     dicom_results[dup_instance_uid_2d]['none'][0]['coords'] = dup_coords_2d
#     dicom_results[dup_instance_uid_2d]['none'][0]['score'] = 0.9
#     dicom_results[dup_instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[dup_instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     row_2d = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_2d].copy()
#     row_2d['SOPInstanceUID'] = dup_instance_uid_2d
#     row_2d['FrameOfReferenceUID'] = dup_for_uid_2d
#     metadata_df = metadata_df.append(row_2d, ignore_index=True)
#
#     coords_3d = [1, 1, 200, 200]
#     dicom_results[instance_uid_3d]['none'][0]['coords'] = coords_3d
#     dicom_results[instance_uid_3d]['none'][0]['slice'] = 5
#     dicom_results[instance_uid_3d]['none'][0]['score'] = 0.1
#     dicom_results[instance_uid_3d]['none'][0]['category'] = 2
#     dicom_results[instance_uid_3d]['none'][0]['origin'] = 'dbt'
#
#     pixel_data = {ix: {'slice_map': slice_map}}
#     model_results_test = helper_results_processing.combine_dxm_dbt_boxes(model_results, metadata_df,
#                                                                          centaur_model.config, pixel_data)
#     dicom_results_test = model_results_test['dicom_results']
#
#     assert dicom_results_test[instance_uid_2d]['none'][0]['score'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['score'] == 0.5
#     assert dicom_results_test[dup_instance_uid_2d]['none'][0]['score'] == 0.9
#     assert dicom_results_test[instance_uid_2d]['none'][0]['coords'] == coords_2d
#     assert dicom_results_test[instance_uid_3d]['none'][0]['coords'] == coords_2d
#     assert dicom_results_test[dup_instance_uid_2d]['none'][0]['coords'] == dup_coords_2d
#
#     # 3. Test that two DxMs with the same FoPUID and shape are combined together
#
#     results = copy.deepcopy(results_orig)
#     model_results = results_parser.get_model_results(results)
#     dicom_results = results_parser.get_dicom_results(results)
#     metadata_df = results_parser.get_metadata(results)
#
#     laterality = 'L'
#     view = 'CC'
#
#     instance_uid_2d = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dxm_class_id()) &
#                                   (metadata_df['ImageLaterality'] == laterality) &
#                                   (metadata_df['ViewPosition'] == view)].iloc[0]['SOPInstanceUID']
#
#     instance_uid_3d = metadata_df[(metadata_df['SOPClassUID'] == DicomTypeMap.get_dbt_class_id()) &
#                                   (metadata_df['ImageLaterality'] == laterality) &
#                                   (metadata_df['ViewPosition'] == view)].iloc[0]['SOPInstanceUID']
#
#     p = "{}/preprocessed_numpy/**/{}".format(data_manager.baseline_dir_centaur_output, instance_uid_3d)
#     temp_folder = glob.glob(p, recursive=True)
#     assert len(temp_folder) == 1, "Temp folder not found: " + p
#     temp_folder = os.path.dirname(temp_folder[0])
#     slice_data_path = "{}/synth/synth_slice_map.npy".format(temp_folder)
#     assert os.path.isfile(slice_data_path), "Synth mapping not found: {}" + slice_data_path
#     slice_map = np.load(slice_data_path)
#
#     dicom_results = {k: dicom_results[k] for k in (instance_uid_2d, instance_uid_3d)}
#     dicom_results_copy = copy.deepcopy(dicom_results)
#     dicom_results = copy.deepcopy(dicom_results_copy)
#     model_results['dicom_results'] = dicom_results
#
#     ix = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_3d].iloc[0].name
#
#     coords_2d = [1, 1, 200, 200]
#     dicom_results[instance_uid_2d]['none'][0]['coords'] = coords_2d
#     dicom_results[instance_uid_2d]['none'][0]['score'] = 0.5
#     dicom_results[instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     rotated_coords_2d = [slice_map.shape[1] - coords_2d[2] - 1,
#                          slice_map.shape[0] - coords_2d[3] - 1,
#                          slice_map.shape[1] - coords_2d[0] - 1,
#                          slice_map.shape[0] - coords_2d[1] - 1]
#
#     dup_instance_uid_2d = 'dup_instance_uid_2d'
#     dup_for_uid_2d = 'dup_for_uid_2d'
#     dicom_results[dup_instance_uid_2d] = copy.deepcopy(dicom_results[instance_uid_2d])
#
#     dup_coords_2d = [1, 1, 200, 200]
#     dicom_results[dup_instance_uid_2d]['none'][0]['coords'] = dup_coords_2d
#     dicom_results[dup_instance_uid_2d]['none'][0]['score'] = 0.9
#     dicom_results[dup_instance_uid_2d]['none'][0]['category'] = 3
#     dicom_results[dup_instance_uid_2d]['none'][0]['origin'] = 'dxm'
#     row_2d = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid_2d].copy()
#     row_2d['SOPInstanceUID'] = dup_instance_uid_2d
#     # row_2d['FrameOfReferenceUID'] = dup_for_uid_2d
#     metadata_df = metadata_df.append(row_2d, ignore_index=True)
#
#     coords_3d = [1, 1, 200, 200]
#     dicom_results[instance_uid_3d]['none'][0]['coords'] = coords_3d
#     dicom_results[instance_uid_3d]['none'][0]['slice'] = 5
#     dicom_results[instance_uid_3d]['none'][0]['score'] = 0.1
#     dicom_results[instance_uid_3d]['none'][0]['category'] = 2
#     dicom_results[instance_uid_3d]['none'][0]['origin'] = 'dbt'
#
#     pixel_data = {ix: {'slice_map': slice_map}}
#     model_results_test = helper_results_processing.combine_dxm_dbt_boxes(model_results, metadata_df,
#                                                                          centaur_model.config, pixel_data)
#     dicom_results_test = model_results_test['dicom_results']
#     assert dicom_results_test[instance_uid_2d]['none'][0]['score'] == \
#            dicom_results_test[dup_instance_uid_2d]['none'][0]['score'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['score'] == 0.9
#     assert dicom_results_test[instance_uid_2d]['none'][0]['category'] == \
#            dicom_results_test[dup_instance_uid_2d]['none'][0]['category'] == \
#            dicom_results_test[instance_uid_3d]['none'][0]['category'] == 3
#     assert dicom_results_test[instance_uid_2d]['none'][0]['coords'] == \
#            dicom_results_test[dup_instance_uid_2d]['none'][0]['coords'] == dup_coords_2d
#     assert dicom_results_test[instance_uid_3d]['none'][0]['coords'] == rotated_coords_2d
#

# def are_equal(bbxs_origin, bbxs_compared):
#     if len(bbxs_origin) != len(bbxs_compared):
#         return False
#     for bbx_o in bbxs_origin:
#         bbx = [b for b in bbxs_compared if (b['coords'] == bbx_o['coords']
#                                             and b['score'] == bbx_o['score']
#                                             and b['category'] == bbx_o['category']
#                                             and b['slice'] == bbx_o['slice']
#                                             and b['origin'] == bbx_o['origin']
#                                             )]
#         if len(bbx) != 1:
#             return False
#     return True
#
@pytest.mark.cadt
def test_T_211():
    """
    Tests if a study can be flipped from suspicious to not suspicious if the threshold were changed.
    Both DXM and DBT studies are tested
    """
    dm = DataManager()
    dm.set_baseline_params(const_deploy.RUN_MODE_CADT)
    for study in (dm.STUDY_01_HOLOGIC, dm.STUDY_03_DBT_HOLOGIC):
        baseline_results_path = os.path.join(dm.baseline_dir_centaur_output, study,
                                         const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
        baseline_study_deploy_results = StudyDeployResults.from_json(baseline_results_path)
        baseline_model_results = baseline_study_deploy_results.results
        sus_code = baseline_model_results['study_results']['total']['category']
        assert sus_code == CategoryHelper.SUSPICIOUS, f"Study {study} expected to be suspicious"
        study_type = DicomTypeMap.get_study_type(baseline_study_deploy_results.metadata)
        # Make sure we are still getting the same category with the current thresholds
        original_thresholds_version = helper_model.get_actual_version_number('cadt')
        with open(f"{const_eng.CADT_THRESHOLD_PATH}/{original_thresholds_version}.json", 'r') as f:
            original_thresholds = json.load(f)
        op_point_key = 'balanced'
        cadt_operating_points_keys = {k: v[op_point_key] for k, v in original_thresholds.items()}
        model_results = results_processing.assign_cadt_categories(baseline_model_results, study_type, cadt_operating_points_keys)

        for lat in baseline_model_results['study_results']:
            assert model_results['study_results'][lat]['category'] == baseline_model_results['study_results'][lat]['category']

        # Modify thresholds
        new_thresholds = {k: 0.999 for k in cadt_operating_points_keys}
        model_results_2 = results_processing.assign_cadt_categories(baseline_model_results,
                                                                    study_type, new_thresholds)

        # Ensure with the new thresholds the study would always be not suspicious
        for lat in model_results_2['study_results']:
            assert model_results_2['study_results'][lat]['category'] == CategoryHelper.NOT_SUSPICIOUS

def test_T_210():
    """
    Test input files are removed after processing
    """
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    try:
        f1_path = os.path.join(input_dir, 'f1')
        os.makedirs(f1_path)
        with open(os.path.join(f1_path, 'file1.dcm'), 'w') as f:
            f.write("Dummy file")

        params = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'remove_input_files': True
        }
        config = Config(overriden_params=params)
        config[Config.MODULE_DEPLOY, 'reports'] = []

        deployer = Deployer()
        deployer.initialize(config)
        deployer.deploy()
        assert not os.path.isdir(f1_path), "Study input folder was not removed"
    finally:
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)

@pytest.mark.cadt
def test_T_211_2():
    """
    Ensure the operating points can be updated in the two allowed ways for Saige-Q and the categories are changing
    accordingly.
    Note: this test could also be considered an integration test because it's using Deployer and Config objects
    initialized in a similar way it's done in the regular deploy
    """
    data_manager = DataManager()
    data_manager.set_baseline_params(const_deploy.RUN_MODE_CADT)

    p = os.path.join(data_manager.baseline_dir_centaur_output, data_manager.STUDY_03_DBT_HOLOGIC,
                                          const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(p), f"Baseline results object not found in ({p})"
    study_deploy_results = StudyDeployResults.from_json(p)

    # Simulate evaluation
    metadata_df = study_deploy_results.get_metadata(passed_checker_only=True)
    baseline_results_raw = study_deploy_results.results_raw

    cl_default_params = {'run_mode': const_deploy.RUN_MODE_CADT, 'output_dir': tempfile.mkdtemp()}
    config_default_params = ConfigFactory.VerificationInternalConfigFactory(
        optional_params_dict=cl_default_params).params_dict
    config_default = Config(overriden_params=config_default_params)

    # Initialize Deployer to replicate the conditions when Engine is evaluating
    deployer = Deployer()
    deployer.initialize(config_default)
    engine = deployer.engine
    results = engine._results_post_processing(study_deploy_results.results_raw, metadata_df)
    # Make sure the study is still Suspicious with the original conditions
    assert results_parser.get_study_results(results)['total']['category'] == CategoryHelper.SUSPICIOUS

    # Scenario 1: change operating point key
    cl_params_2 = cl_default_params.copy()
    cl_params_2['output_dir'] = tempfile.mkdtemp()
    cl_params_2['cadt_operating_point_key'] = 'high_sens'
    config_params = ConfigFactory.VerificationInternalConfigFactory(optional_params_dict=cl_params_2).params_dict
    config = Config(overriden_params=config_params)
    deployer = Deployer()
    deployer.initialize(config)
    # Ensure the thresholds changed
    assert config[Config.MODULE_ENGINE, 'cadt_operating_point_values'] != \
           config_default[Config.MODULE_ENGINE,'cadt_operating_point_values'], \
           "Expected different threshold values for high_sens operating point key"
    # Modify the total score so that the study should be flagged as NOT_SUSPICIOUS based on the config values
    results2 = baseline_results_raw.copy()
    results_parser.get_study_results(results2)['total']['score'] = \
        config[Config.MODULE_ENGINE, 'cadt_operating_point_values'][DicomTypeMap.DBT] - 0.01
    engine = deployer.engine
    results = engine._results_post_processing(results2, metadata_df)
    # Make sure the study is NOT Suspicious now
    assert results_parser.get_study_results(results)['total']['category'] == CategoryHelper.NOT_SUSPICIOUS, \
           "The study score was modified so that it was supposed to be marked as NOT SUSPICIOUS"

    # Scenario 2: change thresholds directly
    cl_params_3 = cl_default_params.copy()
    cl_params_3['output_dir'] = tempfile.mkdtemp()
    config_params = ConfigFactory.VerificationInternalConfigFactory(optional_params_dict=cl_params_3).params_dict
    config = Config(overriden_params=config_params)
    config.set_cadt_operating_points(0.99, 0.99)
    deployer = Deployer()
    deployer.initialize(config)
    # Ensure the thresholds changed
    assert config[Config.MODULE_ENGINE, 'cadt_operating_point_values'] != \
           config_default[Config.MODULE_ENGINE, 'cadt_operating_point_values'], \
           "Expected different threshold values"
    engine = deployer.engine
    results = engine._results_post_processing(study_deploy_results.results_raw, metadata_df)
    # Make sure the study is NOT Suspicious now
    assert results_parser.get_study_results(results)['total']['category'] == CategoryHelper.NOT_SUSPICIOUS, \
           "The study was supposed to be NOT SUSPICIOUS because very high thresholds were set"


def _get_results_example(run_mode):
    """
    Load the example results object used for unit tests (DBT study)
    Args:
        run_mode (str): run mode (one of the RUN_MODE_X constants defined in centaur_deploy.constants)
    Returns:
        StudyDeployResults object
    """
    data_manager = DataManager()
    data_manager.set_baseline_params(run_mode)
    baseline_folder = data_manager.baseline_dir_centaur_output
    p = os.path.join(baseline_folder, DataManager.STUDY_03_DBT_HOLOGIC,
                     const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(p), f"Path {p} not found"
    return StudyDeployResults.from_json(p)


if __name__ == "__main__":
    test_T_211_2()
    print("DONE")