import numpy as np
import json
import logging
import shutil
import tempfile
import warnings
import os
import os.path as osp
import pandas as pd
import re
import argparse

import pydicom

from centaur_deploy.deploy import Deployer
from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_test.config_factory import ConfigFactory
from centaur_test.data_manager import DataManager
from centaur_deploy.deploys.config import Config
import centaur_engine.constants as const_engine
import centaur_deploy.constants as const_deploy
import centaur_reports.constants as const_reports
import deephealth_utils.data.parse_logs as parse_logs
import deephealth_utils.misc.results_parser as results_parser
from deephealth_utils.misc.config import configs_equal
from deephealth_utils.misc.results_parser import are_equal

#
# def compare_model_results_to_baseline(study, predicted_results, raw_results=None):
#     '''
#     Test the results of a model evaluation.
#     :param ds_version: str. Dataset version. Ex: DS_01
#     :param predicted_results: dictionary of results (following results.json format)
#     :param raw_results: dictionary of raw results (following results_raw.json format)
#     '''
#     # Read baseline results
#     data_manager = DataManager()
#     results_file_path = "{}/{}/{}".format(data_manager.baseline_dir_centaur_output, study,
#                                           const_engine.CENTAUR_RESULTS_JSON)
#     assert os.path.isfile(results_file_path), "Results file path not found: {}".format(results_file_path)
#     with open(results_file_path, 'rb') as f:
#         baseline = json.load(f)
#
#     assert results_parser.are_equal(predicted_results, baseline['model_results']), \
#         "Output results different than baseline. **** Predicted: {}.\n\n**** Expected: {}".format(
#             baseline['model_results'], predicted_results)
#
#     if raw_results is not None:
#         results_file_path = "{}/{}/{}".format(data_manager.baseline_dir_centaur_output, study,
#                                               const_engine.CENTAUR_RESULTS_RAW_JSON)
#         assert os.path.isfile(results_file_path), "Results file path not found: {}".format(results_file_path)
#         with open(results_file_path, 'rb') as f:
#             baseline = json.load(f)
#
#         assert results_parser.are_equal(raw_results, baseline), \
#             "Output results different than baseline. **** Predicted: {}.\n\n**** Expected: {}".format(
#                 baseline, raw_results)


def validate_dicom_results_study(dicom_results, study_results, proc_info, metadata_df):
    """
    Ensure the results for a study are acceptable. It doesn't require the results matching exactly a baseline,
    it just checks that the results "look good" (metadata looks correct, coordinates seem reasonable...
    Args:
        dicom_results (dict): dicom_results section
        study_results (dict): study_results section
        proc_info (dict): proc_info section
        metadata_df (dataframe): metadata datafrane

    Returns:
        True if all the checks pass
    """
    # Make sure we have a result for each file
    assert set(dicom_results.keys()) == set(metadata_df.SOPInstanceUID.to_list()), \
        "Inconsistent SOPInstanceUIDs. Dicom_results: {}; Metadata_df: {}".format(set(dicom_results.keys()),
                                                                                  set(
                                                                                      metadata_df.SOPInstanceUID.to_list()))

    for sop_instance_uid in dicom_results:
        # Validate scores
        file_scores = dicom_results[sop_instance_uid]
        # Loop over the different transformations applied
        for transf_key in file_scores.keys():
            tranf_values_list = file_scores[transf_key]
            for value_d in tranf_values_list:
                # Get the coordinates for this transformation
                assert 'coords' in value_d, "Coordinates not found for SOPClassUID='{}' and tranf='{}'. Value: {}". \
                    format(sop_instance_uid, transf_key, value_d)
                coords = value_d['coords']
                # Basic checks for the coordinates
                assert isinstance(coords, list) and len(coords) == 4, "Unexpected number of coordinates: {}".format(
                    coords)
                original_shape = proc_info[sop_instance_uid]['original_shape']
                # Coordinates validation: x1 y1 x2 y2;  Shape: Y X
                # Check there are not out of bounds coords
                assert 0 <= coords[0] < original_shape[1] - 1, "Out of bounds coords: {}".format(coords)
                assert 0 < coords[2] <= original_shape[1] - 1, "Out of bounds coords: {}".format(coords)
                assert 0 <= coords[1] < original_shape[0] - 1, "Out of bounds coords: {}".format(coords)
                assert 0 < coords[3] <= original_shape[0] - 1, "Out of bounds coords: {}".format(coords)
                # Check the coordinates define a bounding box
                height = coords[2] - coords[0]
                width = coords[3] - coords[1]
                assert height > 0 and width > 0, "The coords {} do not define a bounding box".format(coords)
                # Check there is a min and max size
                biggest_size = max(original_shape[0], original_shape[1])
                min_size = biggest_size * 0.005
                max_size = biggest_size * 0.75
                assert 0 < height < original_shape[0], \
                    "Bounding box height is outside of valid bounds. Coords: {}".format(coords)
                assert 0 < width < original_shape[1], \
                    "Bounding box height is outside of valid bounds. Coords: {}".format(coords)

                if not(min_size < height < max_size):
                    warnings.warn(UserWarning("Bounding box height seems suspicious. Coords: {}".format(coords)))
                if not(min_size < width < max_size):
                    warnings.warn(UserWarning("Bounding box width seems suspicious. Coords: {}".format(coords)))

    # STUDY RESULTS
    assert study_results.keys() == {'R', 'L', 'total'}
    for lat in study_results.keys():
        assert 0. <= study_results[lat]['score'] <= 1.
        # assert isinstance(study_results[lat]['category'], int), str(study_results[lat]['category'])
    return True


def validate_study_output_folder(output_folder):
    """
    Check the output of a study processing and prediction looks ok
    :param output_folder: str. Path to the study processing output
    """
    # Ensure metadata files and results file exist
    assert osp.isfile("{}/{}".format(output_folder, const_engine.METADATA_PKL_NAME)), \
        "Missing metadata file in {}/{}".format(output_folder, const_engine.METADATA_PKL_NAME)
    assert osp.isfile("{}/{}".format(output_folder, const_engine.CENTAUR_RESULTS_JSON)), \
        "Missing results file in {}/{}".format(output_folder, const_engine.CENTAUR_RESULTS_JSON)

    with open(os.path.join(output_folder, const_engine.CENTAUR_RESULTS_JSON), 'rb') as f:
        results = json.load(f)
    # Use the "external" metadata
    metadata_df = pd.read_pickle("{}/{}".format(output_folder, const_engine.METADATA_PKL_NAME))

    # Validate study results
    validate_dicom_results_study(results['model_results']['dicom_results'],
                                 results['model_results']['study_results'],
                                 results['model_results']['proc_info'],
                                 metadata_df)




def test_full_deploy(run_mode, compare_to_baseline=True, filtered_studies=None):
    """
    Test a full deploy for a folder or a selection of studies
    Args:
        run_mode:
        compare_to_baseline:
        filtered_studies:

    Returns:

    """
    def _study_validation(study_deploy_results):
        """
        Function that will be invoked after running each study
        Args:
            study_deploy_results:

        Returns:

        """
        study_name = study_deploy_results.get_study_dir_name()
        baseline_study_deploy_results = StudyDeployResults.from_json(os.path.join(
            data_manager.baseline_dir_centaur_output,
            study_name,
            const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON))
        if compare_to_baseline:
            compare_study_deploy_results(study_deploy_results, baseline_study_deploy_results)
        else:
            # Just ensure there are not unexpected errors
            assert not study_deploy_results.has_unexpected_errors(), \
                f"Unexpeceted error for study {study_name}: {study_deploy_results.get_error_message()}"

    baseline_params = {'run_mode': run_mode}
    data_manager = DataManager(baseline_params=baseline_params)

    input_dir = data_manager.data_studies_dir

    # If studies must be filtered, copy the filtered studies to a temp input folder
    input_dir_tmp = None
    output_dir = None

    try:
        if filtered_studies is not None:
            input_dir_tmp = tempfile.mkdtemp(prefix="temp_input_folder_")
            for study in filtered_studies:
                print(f"Copying {study} to {input_dir_tmp}...")
                shutil.copytree(os.path.join(input_dir, study), f"{input_dir_tmp}/{study}")
        output_dir = tempfile.mkdtemp(prefix="temp_output_folder_")
        config_factory = ConfigFactory.VerificationInternalConfigFactory(
            input_dir=input_dir if filtered_studies is None else input_dir_tmp, output_dir=output_dir,
            optional_params_dict=baseline_params)
        deployer = Deployer()

        config = deployer.create_config_object(config_factory.params_dict)
        deployer.initialize(config)

        # For each study in the deploy, validate the results
        deployer.deploy(study_callback=_study_validation)
        if compare_to_baseline:
            compare_global_deploy_results(data_manager.baseline_dir_centaur_output, output_dir,
                                          filtered_studies=filtered_studies)
        else:
            deployer.deploy()
    finally:
        if input_dir_tmp is not None:
            shutil.rmtree(input_dir_tmp)
            print(f"Temp input folder removed ({input_dir_tmp}")
        if output_dir is not None:
            shutil.rmtree(output_dir)
            print(f"Temp output folder removed ({output_dir})")



def compare_study_deploy_results(generated_results, baseline_results):
    """
    Compare two StudyDeployResults objects (baseline and generated).
    The method fails if there is any unexpected difference

    Args:
        baseline_results (StudyDeployResults):
        generated_results (StudyDeployResults):
    """
    attrs_b = set(vars(baseline_results))
    attrs_g = set(vars(generated_results))
    assert attrs_b == attrs_g, "The keys in baseline and generated results do not match. Difference: {}".format(
        attrs_b.difference(attrs_g))

    # Compare metadata (remove a few irrelevant columns and make sure the result are sorted)
    assert baseline_results.metadata.shape == generated_results.metadata.shape, "Metadata shape do not match"
    if len(baseline_results.metadata) > 0:
        dfb = baseline_results.metadata.sort_values("SOPInstanceUID").reset_index().drop(
            ['index', 'dcm_path', 'np_paths', 'ContentDate', 'ContentTime', 'StudyTime'], axis=1)
        dfg = generated_results.metadata.sort_values("SOPInstanceUID").reset_index().drop(
            ['index', 'dcm_path', 'np_paths', 'ContentDate', 'ContentTime', 'StudyTime'], axis=1)
        for r in range(dfb.shape[0]):
            row_b = dfb.iloc[r]
            row_g = dfg.iloc[r]
            for c in range(len(row_b)):
                assert are_equal(row_b[c], row_g[c]), \
                  f"Row {r}, column {c} ({dfb.columns[c]}) in metadata do not match. Baseline:{row_b[c]}; Pred:{row_g[c]}"

    # Compare reports
    b = baseline_results.reports_generated
    g = generated_results.reports_generated
    assert b.keys() == g.keys(), f"Reports_generated keys do not match. Baseline: {b.keys()}; Generated: {g.keys()}"
    # For each report, compare the file names only (the output paths will change)
    for report_key, report_b in b.items():
        assert not ((g[report_key] is None) ^ (report_b is None)), \
            f"Only one of the '{report_key}' reports is null. Baseline: {report_b}. Generated: {g[report_key]}"
        if report_b is not None:
            if isinstance(report_b, dict):
                assert report_b.keys() == g[report_key].keys()
                for report_dict_key, report_dict_value in report_b.items():
                    assert os.path.basename(report_dict_value) == os.path.basename(g[report_key][report_dict_key])
            elif isinstance(report_b, str):
                assert os.path.basename(report_b) == os.path.basename(g[report_key])
            else:
                raise ValueError(f"Unexpected type for report {report_key}: {report_b}")

    # Compare results
    assert results_parser.are_equal(baseline_results.results_raw, generated_results.results_raw), \
        f"Results different in baseline and generated results.\nBaseline: {baseline_results.results_raw}\n" \
        f"Generated: {generated_results.results_raw}"

    assert results_parser.are_equal(baseline_results.results, generated_results.results), \
         f"Results different in baseline and generated results.\nBaseline: {baseline_results.results}\n" \
         f"Generated: {generated_results.results}"

def compare_global_deploy_results(baseline_output_folder, output_folder, filtered_studies=None):
    # Compare the global results to the baseline
    # Compare config file
    output_config = Config(config_file_path=osp.join(output_folder, const_deploy.CENTAUR_CONFIG_JSON))
    baseline_config = Config(
        config_file_path=osp.join(baseline_output_folder, const_deploy.CENTAUR_CONFIG_JSON))
    assert configs_equal(baseline_config, output_config), \
        "Output config:\n{};\n****\nBaseline config:\n{}".format(output_config, baseline_config)

    # Compare the study summary report (if available)
    csv_baseline_path = os.path.join(baseline_output_folder, const_reports.SUMMARY_REPORT_STUDY_CSV)
    csv_output_path = os.path.join(output_folder, const_reports.SUMMARY_REPORT_STUDY_CSV)
    if os.path.isfile(csv_baseline_path):
        assert os.path.isfile(csv_output_path), f"File {csv_output_path} expected"
        df_baseline = pd.read_csv(csv_baseline_path).sort_values(['ix'])
        if filtered_studies is not None:
            # Filter just the results for the filtered studies
            df_baseline = df_baseline[df_baseline.apply(
                lambda r: os.path.basename(r['input_dir']) in filtered_studies, axis=1)]
        df_output = pd.read_csv(csv_output_path).sort_values(['ix'])
        assert np.array_equal(df_baseline[['ix', 'total']].values, df_output[['ix', 'total']].values)
    else:
        assert not os.path.isfile(csv_output_path), f"File {csv_output_path} not expected"

    # Compare the dicom summary report (if available)
    csv_baseline_path = os.path.join(baseline_output_folder, const_reports.SUMMARY_REPORT_DICOM_CSV)
    csv_output_path = os.path.join(output_folder, const_reports.SUMMARY_REPORT_DICOM_CSV)
    if os.path.isfile(csv_baseline_path):
        assert os.path.isfile(csv_output_path), f"File {csv_output_path} expected"
        df_baseline = pd.read_csv(csv_baseline_path).sort_values(['ix'])
        if filtered_studies is not None:
            # Filter just the results for the filtered studies
            df_baseline = df_baseline[df_baseline.apply(
                lambda r: os.path.basename(r['input_study_dir']) in filtered_studies, axis=1)]

        df_output = pd.read_csv(csv_output_path).sort_values(['ix'])
        dfb = df_baseline['ix'].values
        dfo = df_output['ix'].values
        assert np.array_equal(dfb, dfo), \
            f"DicomSummaryReport values do not match.\nBaseline: {dfb}\nGenerated:{dfo}"

        dfb = df_baseline['score_total'].values.round(3)
        dfo = df_output['score_total'].values.round(3)
        assert np.array_equal(dfb, dfo, equal_nan=True), \
            f"DicomSummaryReport values do not match.\nBaseline: {dfb}\nGenerated:{dfo}"
    else:
        assert not os.path.isfile(csv_output_path), f"File {csv_output_path} not expected"

def _find_differences(d1, d2):
    """
    Print info about the first difference between two dictionaries
    :param d1: dictionary
    :param d2: dictionary
    :return: boolean. d1 == d2
    """
    if d1.keys() != d2.keys():
        print("D1 keys: {}; D2 keys: {}".format(d1.keys(), d2.keys()))
    for key in d1.keys():
        if isinstance(d1[key], dict):
            if not _find_differences(d1[key], d2[key]):
                return False
        elif d1[key] != d2[key]:
            print("d1[{}]={}; d2[{}]={}".format(key, d1[key], key, d2[key]))
            return False
    return True


def get_dataset_differences(ds1, ds2, allowed_diffs=None):
    """
    Compare two dicom datasets
    :param ds1: pydicom Dataset
    :param ds2: pydicom Dataset
    :param accepted_diffs: collection of dicom tags that are allowed to be different (in '(0000, 0000)' format)
    :return: list of differences found (string representation)
    """
    rep = [str(dataset).split("\n") for dataset in [ds1, ds2]]

    # # Get the differences
    if allowed_diffs is not None:
        allowed_diffs = set(allowed_diffs)
    else:
        allowed_diffs = set()

    pattern = '(\(\w{4},\s\w{4}\))\s'

    diffs = list()
    line_num = 0

    for left, right in zip(rep[0], rep[1]):
        line_num += 1
        count_diff = False

        if left != right:
            count_diff = True
            match_left = re.search(pattern, left)
            match_right = re.search(pattern, right)
            if match_left and match_right:  # If both lines are dicom fields
                field_left = match_left.group(1)
                field_right = match_right.group(1)
                if field_left == field_right and field_left in allowed_diffs:
                    count_diff = False

        if count_diff:
            diffs.append('Line {}\n{}\n{}'.format(line_num, left, right))

    return diffs


def read_result_files_fixed():
    """
    Read results files for all the studies and fix the metadata paths to convert from relative to absolute
    :return: dict. Study(str) - Results(StudyDeployResults)
    """
    data_manager = DataManager()
    data_manager.set_default_baseline_params()
    studies = data_manager.get_all_studies()
    results_dict = {}
    for study in studies:
        # Read results
        results_file_path = "{}/{}/{}".format(data_manager.baseline_dir_centaur_output, study,
                                              const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
        assert osp.isfile(results_file_path), "Results file path not found: {}".format(results_file_path)
        # Fix metadata paths that are needed to access the numpy arrays to generate images
        results = StudyDeployResults.from_json(results_file_path)
        metadata = results.metadata
        if metadata is None or metadata.empty:
            print(f"No metadata available for study {study}")
        else:
            metadata['dcm_path'] = (data_manager.data_dir + "/") + metadata['dcm_path']
            metadata['np_paths'] = (data_manager.baseline_dir + "/") + metadata['np_paths']
        results_dict[study] = results
    return results_dict


def get_testing_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # if len(logger.handlers) < 2:
    #     # Initialize console debug handler
    #     logger.setLevel(logging.DEBUG)
    #     ch = logging.StreamHandler()
    #     ch.setLevel(logging.DEBUG)
    #     logging.getLogger().setLevel(logging.DEBUG)
    #     logging.getLogger().addHandler(ch)
    return logger


def get_study_instance_uid(study_dir):
    """
    Get the StudyInstanceUID for a study reading the first file
    Args:
        study_dir (str): path to the study folder

    Returns:

    """
    f = os.path.join(study_dir, os.listdir(study_dir)[0])
    ds = pydicom.dcmread(f)
    return ds.StudyInstanceUID

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Miscelaneus")
    parser.add_argument('op', help="Operation", choices=['profile'])
    parser.add_argument('-i', '--input', help="Input file")
    parser.add_argument('-o', '--output', help="Output file")
    args = parser.parse_args()

    if args.op == 'profile':
        assert args.input is not None and args.output is not None
        df = parse_logs.get_profile_df(args.input)
        df.to_csv(args.output)
        print("{} generated".format(args.output))
