import os
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import tempfile

from centaur_test.data_manager import DataManager
from centaur_deploy.deploy import Deployer
from centaur_deploy.deploys.config import Config
import centaur_deploy.constants as const_deploy
from centaur_engine.helpers.helper_results_processing import orient_bounding_box_coords
from centaur_engine.helpers.helper_preprocessor import get_orientation_change, implement_orientation
import deephealth_utils.data.parse_logs as parse_logs
import shutil


def run_deployer(config_dict):
    """
    Runs deployer based on config and return results of studies
    Args:
        config: Config object from centaur_deploy.deploys.config
    Returns: list of study results
    """

    deployer = Deployer()

    config = deployer.create_config_object(config_dict)
    config[Config.MODULE_DEPLOY, 'exit_on_error'] = True

    deployer.initialize(config)
    deployer.logger.info(parse_logs.log_line(-1, "Initialization finished. Deploying..."))
    processed_studies = deployer.deploy(return_results=True)
    return processed_studies

def compare_dicom_results(model_results1, model_results2, orientation_map_original, orientation_map_new):
    """
    comparing the results of two models
    Args:
        model_results1: results of original orientation
        model_results2: results of the new orientation
        orientation_map_original: dictionary that map the orientation of the original study
        orientation_map_new: dictionary that map the orientation of the newly oriented study

    Returns:

    """
    new_orientation_map = {}
    original_orientation_change = {}

    for key in orientation_map_new:
        new_key = ".".join(key.split(".")[1:][:-1])
        new_orientation_map[new_key] = orientation_map_new[key]

    for key in orientation_map_original:
        new_key = ".".join(key.split(".")[1:][:-1])
        original_orientation_change[new_key] = orientation_map_original[key]

    match = True
    for dicom in model_results1["dicom_results"]:
        for i, box in enumerate(model_results1["dicom_results"][dicom]["none"]):
            for key in box:
                if key == "score":
                    score1 = round(box[key], 4)
                    score2 = round(model_results2["dicom_results"][dicom]["none"][i][key], 4)
                    if score1 != score2:
                        return False

                elif key == "coords":
                    oc1 = original_orientation_change[dicom]
                    oc2 = new_orientation_map[dicom]

                    if oc2 == "keep original":
                        oc1 = "identity"
                        oc2 = "identity"

                    img_shape1 = model_results1['proc_info'][dicom]['original_shape']
                    img_shape2 = model_results2['proc_info'][dicom]['original_shape']
                    coordinate1 = orient_bounding_box_coords(box[key], img_shape1, oc1)
                    coordinate2 = orient_bounding_box_coords(model_results2["dicom_results"][dicom]["none"][i][key],
                                                             img_shape2, oc2)
                    if coordinate1 != coordinate2:
                        return False

                else:
                    if box[key] != model_results2["dicom_results"][dicom]["none"][i][key]:
                        return False
    return match



def test_T_223():
    """
    create a variation of the study_03_dbt_hologic where we cover all the possible orientations
    Returns: None

    """
    output_dir = tempfile.mkdtemp()
    try:
        dm = DataManager()
        study = dm.STUDY_03_DBT_HOLOGIC
        study_path = os.path.join(const_deploy.LOCAL_TESTING_STUDIES_FOLDER, study)
        dicoms = os.listdir(study_path)

        foruid_to_dicom = {}
        foruid_to_image_lat = {}
        dicom_to_patient_orientation = {}
        original_orientation_change = {}
        Dxms = []
        BTs = []

        for dicom in dicoms:
            path = os.path.join(study_path, dicom)
            ds = pydicom.dcmread(path)
            foruid = ds.FrameOfReferenceUID
            if foruid not in foruid_to_dicom:
                foruid_to_dicom[foruid] = [dicom]
            else:
                foruid_to_dicom[foruid].append(dicom)

            po = ds.PatientOrientation
            po = '|'.join(po)
            if 'DXm' in dicom:
                Dxms.append(dicom)
                lat = ds.ImageLaterality

                foruid_to_image_lat[foruid] = lat
                dicom_to_patient_orientation[dicom] = po
                original_orientation_change[dicom] = get_orientation_change(lat, po)
            # for some reason BT studies of study dont have ImageLaterlity
            if 'BT' in dicom:
                BTs.append(dicom)
                dicom_to_patient_orientation[dicom] = po

        for foruid, dicoms_foruid in foruid_to_dicom.items():
            for dicom in dicoms_foruid:
                if 'BT' in dicom:
                    lat = foruid_to_image_lat[foruid]
                    po = dicom_to_patient_orientation[dicom]
                    original_orientation_change[dicom] = get_orientation_change(lat, po)



        study_03_variation = os.path.join(output_dir, "study_03_variation")
        os.mkdir(study_03_variation)
        new_dicom_to_orientation_change = {}

        # make the variation of study3 that contains all possible orientation changes
        orientation_changes = ["rotate_180", "up_down_flip", "left_right_flip", "keep original"]
        orientation_to_po = {
            'R': {
                "rotate_180": ["A", "R"],
                "up_down_flip": ["P", "R"],
                "left_right_flip": ["A", "L"]
            },
            'L': {
                "rotate_180": ["P", "L"],
                "up_down_flip": ["A", "L"],
                "left_right_flip": ["P", "R"]
            },
        }

        for i in range(4):
            bt = BTs[i]
            dxm = Dxms[i]
            orientation_change = orientation_changes[i]
            new_dicom_to_orientation_change[bt] = orientation_change
            new_dicom_to_orientation_change[dxm] = orientation_change

        for dicom in dicoms:
            path = os.path.join(study_path, dicom)
            ds = pydicom.dcmread(path)
            target_orientation_change = new_dicom_to_orientation_change[dicom]
            new_dicom_path = os.path.join(study_03_variation, dicom)
            if target_orientation_change == "keep original":
                ds.save_as(new_dicom_path)
            else:
                foruid = ds.FrameOfReferenceUID
                lat = foruid_to_image_lat[foruid]
                current_orientation_change = original_orientation_change[dicom]
                pixels = ds.pixel_array
                pixel_correct = implement_orientation(pixels, current_orientation_change)
                pixels_target = implement_orientation(pixel_correct, target_orientation_change)
                target_po = orientation_to_po[lat][target_orientation_change]

                assert get_orientation_change(lat, '|'.join(target_po)) == target_orientation_change, \
                    "target_po {} does not give target orientation {}".format(target_po, target_orientation_change)

                ds.PatientOrientation = target_po
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                ds.PixelData = pixels_target.tostring()
                ds.save_as(new_dicom_path)

        centaur_out = os.path.join(output_dir, "centaur_outputs")
        os.mkdir(centaur_out)

        # run centaur on the original study
        orig_input_dir = study_path
        orig_output_dir = os.path.join(centaur_out, "original")
        os.mkdir(orig_output_dir)
        config_dict = {
            'input_dir': orig_input_dir,
            'output_dir': orig_output_dir,
            'run_mode': 'CADx',
            'save_synthetics': True,
        }
        result_original = run_deployer(config_dict)[0]



        # run centaur on the variation study
        variation_input_dir = study_03_variation
        variation_output_dir = os.path.join(centaur_out, "variation")
        os.mkdir(variation_output_dir)
        config_dict = {
            'input_dir': variation_input_dir,
            'output_dir': variation_output_dir,
            'run_mode': 'CADx'
        }
        result_variation = run_deployer(config_dict)[0]

        match = compare_dicom_results(result_original.results, result_variation.results, original_orientation_change, new_dicom_to_orientation_change)
        assert match, " the boxes of the oriented study does not match the original study"
    finally:
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    test_T_216()

