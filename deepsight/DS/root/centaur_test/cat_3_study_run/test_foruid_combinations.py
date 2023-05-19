import copy
import os
import pydicom
import tempfile
import shutil
import numpy as np

import pytest

from centaur_deploy.deploy import Deployer
from centaur_deploy.deploys.config import Config
import centaur_engine.constants as const
import deephealth_utils.data.parse_logs as parse_logs
from centaur.centaur_engine.helpers.helper_results_processing import combine_foruid_boxes, assign_cadx_categories
import centaur_deploy.constants as const_deploy
from centaur_test.data_manager import DataManager
from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_engine.helpers import helper_model
from deephealth_utils.misc.config import load_config
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap




def get_organize_dict(study_path):
    """
    get a structural dictionary that organize all file by FORUID and modality
    Args:
        study_path: path of the study

    Returns: dictionary

    """
    all_files = {}
    for file in os.listdir(study_path):
        path = os.path.join(study_path, file)
        ds = pydicom.dcmread(path)
        foruid = ds.FrameOfReferenceUID
        sd = ds.SeriesDescription

        if foruid not in all_files:
            all_files[foruid] = {}
            all_files[foruid]["BT"] = None
            all_files[foruid]["FFDM"] = None
            all_files[foruid]["CVIEW"] = None

        if "Breast Tomosynthesis Image" in sd:
            mode = "BT"

        elif "C-View" in sd:
            mode = "CVIEW"
        else:
            mode = "FFDM"

        assert all_files[foruid][mode] is None
        all_files[foruid][mode] = file
    return all_files


def check_combine(all_files, combined_dicom_results,box_to_dicom,  mode_exclude=[]):
    """
    Checks if boxes are combined as expected. DICOM boxes with the same FORUID and shape should be combined
    Args:
        all_files: structural dictionary of all the files of a study
        combined_dicom_results: model results after calling combine box function
        box_to_dicom: dictionary that maps box scores to the origin of the
        mode_exclude:(str) one of ["CVIEW", "BT" , "FFDM"]  Dicoms of which mode was the model not suppose to combine

    Returns: None

    """
    for foruid in all_files:
        boxes_foruid = None
        foruid_match = None
        for mode in all_files[foruid]:
            dicom = '.'.join(all_files[foruid][mode].split(".")[1:]).replace(".dcm", "")
            boxes = combined_dicom_results[dicom]["none"]

            if mode in mode_exclude:
                for box in boxes:
                    score = box["score"]
                    assert box_to_dicom[score] == dicom, "dicom {} has unique FORUID but have boxes from {}".format(
                        dicom, box_to_dicom[score])
            else:

                if boxes_foruid is None:
                    boxes_foruid = set([])

                    for box in boxes:
                        score = box["score"]
                        boxes_foruid.add(score)
                    foruid_match = dicom

                else:
                    for box in boxes:
                        score = box["score"]
                        assert score in boxes_foruid, "dicoms {} and {} have same FORUID however they dont have the same combined boxes".format(
                            dicom, foruid_match)


def customize_model_results_metadata(metadata, exclude, all_files):
    """
    modify metadata and results to fit what we are testing
    Args:

        metadata: Pandas DataFrame
        exclude: which modality (BT, FFDM, CVIEW) we are not combining
        all_files: structural dictionary that organize all file by FORUID and modality

    Returns: None

    """


    for foruid in all_files:
        for mode in all_files[foruid]:
            if mode in exclude:
                sop_to_exclude = all_files[foruid][mode]
                sop_to_exclude = ".".join(sop_to_exclude.split(".")[1:][:-1])
                index = metadata[metadata["SOPInstanceUID"] == sop_to_exclude].index
                metadata.at[index, 'FrameOfReferenceUID'] = sop_to_exclude


@pytest.mark.cadx
def test_T_217():
    """
    Test all combinations of BT, CVIEW and FFDM
    Args:
        study_path: path of the study

    Returns:

    """

    dm = DataManager()
    dm.set_baseline_params(const_deploy.RUN_MODE_CADX)
    study = dm.STUDY_05_HOLOGIC_COMBO_MODE
    study_path = os.path.join(const_deploy.LOCAL_TESTING_STUDIES_FOLDER, study)
    baseline_path = os.path.join(dm.baseline_dir_centaur_output, study)
    study_path_results = os.path.join(baseline_path, const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)

    study_result = StudyDeployResults.from_json(study_path_results)
    metadata = study_result.metadata
    model_results = study_result.results_raw

    cadx_version = helper_model.get_actual_version_number('cadx', 'current')
    cadx_threshold_config = load_config(const.CADX_THRESHOLD_PATH + '/' + "{}.json".format(cadx_version))
    model_results = assign_cadx_categories(model_results, metadata, cadx_threshold_config)

    model_version = helper_model.get_actual_version_number('model', 'current')
    model_config = load_config(const.MODEL_PATH + str(model_version) + '/' + const.MODEL_CONFIG_JSON)

    pixel_data = {}
    for i, row in metadata.iterrows():
        if DicomTypeMap.get_type_row(row) == DicomTypeMap.DBT:
            sop_uid = row.SOPInstanceUID
            numpy_path = os.path.join(baseline_path, sop_uid, 'slice_map.npy')
            slice_map = np.load(numpy_path)
            pixel_data[i] = {}
            pixel_data[i]['slice_map'] = slice_map

    box_to_dicom= {}
    dicom_results = model_results["dicom_results"]
    for dicom, val in dicom_results.items():
        boxes = val["none"]
        for box in boxes:
            score = box["score"]
            assert score not in box_to_dicom
            box_to_dicom[score] = dicom

    all_files = get_organize_dict(study_path)

    # test FORUID same:
    model_results_all = copy.deepcopy(model_results)
    metadata_all = copy.deepcopy(metadata)
    exclude = []
    customize_model_results_metadata(metadata_all, exclude, all_files)
    results = combine_foruid_boxes(model_results_all, metadata_all, model_config, pixel_data)
    check_combine(all_files, results["dicom_results"], box_to_dicom, exclude)
    #
    # test combine C_View and DBT:
    model_results_cview_dbt = copy.deepcopy(model_results)
    metadata_cview_dbt = copy.deepcopy(metadata)
    exclude = ["FFDM"]
    customize_model_results_metadata(metadata_cview_dbt, exclude, all_files)
    results = combine_foruid_boxes(model_results_cview_dbt, metadata_cview_dbt, model_config, pixel_data)
    check_combine(all_files, results["dicom_results"], box_to_dicom, exclude)

    # test combine C_view FFDM
    model_results_cview_ffdm = copy.deepcopy(model_results)
    metadata_cview_cview_ffdm = copy.deepcopy(metadata)
    exclude = ["BT"]
    customize_model_results_metadata(metadata_cview_cview_ffdm, exclude, all_files)
    results = combine_foruid_boxes(model_results_cview_ffdm, metadata_cview_cview_ffdm, model_config, pixel_data)
    check_combine(all_files, results["dicom_results"], box_to_dicom, exclude)

    # test combine FFDM DBT
    model_results_cview_ffdm = copy.deepcopy(model_results)
    metadata_cview_cview_ffdm = copy.deepcopy(metadata)
    exclude = ["CVIEW"]
    customize_model_results_metadata(metadata_cview_cview_ffdm, exclude, all_files)
    results = combine_foruid_boxes(model_results_cview_ffdm, metadata_cview_cview_ffdm, model_config, pixel_data)
    check_combine(all_files, results["dicom_results"], box_to_dicom,  exclude)


if __name__ == '__main__':
    test_T_217()
    print("Done!")

