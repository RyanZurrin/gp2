import os
import shutil
import tempfile

from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_test.config_factory import ConfigFactory
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from deephealth_utils.misc import results_parser
from deephealth_utils.data.utils import dh_dcmread

import centaur_deploy.constants as const
from centaur_deploy.deploy import Deployer
from centaur_test.data_manager import DataManager


def test_T_219(input_study_folder="/root/HologicHD"):
    """
    Ensure the Hologic I2D images can be processed and produce identical results in both "DBT" and "Dxm" formats.
    TODO: this test requires to mount an external study in /root/HologicHD at the moment
    """

    assert os.path.isdir(input_study_folder), f"{input_study_folder} not found"

    root_input_dir = tempfile.mkdtemp()
    original_study_input_dir = os.path.join(root_input_dir, "original")
    modified_study_input_dir = os.path.join(root_input_dir, "modified")

    output_dir = tempfile.mkdtemp()
    #original_study_output_dir = os.path.join(output_dir, "original")
    #modified_study_output_dir = os.path.join(output_dir, "modified")

    try:
        # Copy the original study
        shutil.copytree(input_study_folder, original_study_input_dir)
        optional_params = {
            #'checks_to_ignore': ['PAC-10', 'SAC-81'],
            #'skip_study_checks': True
        }
        # Deploy the original study
        config_factory = ConfigFactory.VerificationInternalConfigFactory(input_dir=root_input_dir,
                                                                         output_dir=output_dir,
                                                                         optional_params_dict=optional_params)

        # Generate a new study converting from DBT to Dxm (or viceversa)
        os.makedirs(modified_study_input_dir)

        study_path = os.path.join(const.LOCAL_TESTING_STUDIES_FOLDER, DataManager.STUDY_01_HOLOGIC)
        dxm_shell_path = os.path.join(study_path, os.listdir(study_path)[0])
        dxm_shell = dh_dcmread(dxm_shell_path)
        for f in os.listdir(original_study_input_dir):
            p = os.path.join(original_study_input_dir, f)
            ds = dh_dcmread(p)
            if DicomTypeMap.get_image_classification(ds) == DicomTypeMap.Classification.INTELLIGENT_2D:
                arr = ds.pixel_array
                assert arr.ndim == 2, f"Expected a 2D image. Image shape: {arr.shape}"

                dxm_shell.SOPInstanceUID = ds.dh_getattribute('SOPInstanceUID')
                dxm_shell.SeriesInstanceUID = ds.dh_getattribute('SeriesInstanceUID')
                dxm_shell.ViewPosition = ds.dh_getattribute('ViewPosition')
                dxm_shell.ImageLaterality = ds.dh_getattribute('ImageLaterality')
                dxm_shell.Rows = ds.dh_getattribute('Rows')
                dxm_shell.Columns = ds.dh_getattribute('Columns')
                dxm_shell.Manufacturer = ds.dh_getattribute('Manufacturer')
                dxm_shell.ManufacturerModelName = ds.dh_getattribute('ManufacturerModelName')
                dxm_shell.PatientName = ds.dh_getattribute('PatientName')
                dxm_shell.PatientID = ds.dh_getattribute('PatientID')
                dxm_shell.StudyInstanceUID = ds.dh_getattribute('StudyInstanceUID')
                dxm_shell.FrameOfReferenceUID = ds.dh_getattribute('FrameOfReferenceUID')
                dxm_shell.StudyID = ds.dh_getattribute('StudyID')
                dxm_shell.StudyTime = ds.dh_getattribute('StudyTime')
                dxm_shell.StudyDate = ds.dh_getattribute('StudyDate')
                dxm_shell.FrameOfReferenceUID = ds.dh_getattribute('FrameOfReferenceUID')
                dxm_shell.AccessionNumber = ds.dh_getattribute('AccessionNumber')
                dxm_shell.NumberOfFrames = ds.dh_getattribute('NumberOfFrames')
                dxm_shell.PatientOrientation = ds.dh_getattribute('PatientOrientation')
                dxm_shell.PatientOrientation = ds.dh_getattribute('PatientOrientation')
                dxm_shell.PatientBirthDate = ds.dh_getattribute('PatientBirthDate')
                dxm_shell.PatientAge = ds.dh_getattribute('PatientAge')
                dxm_shell.ReferringPhysicianName = ds.dh_getattribute('ReferringPhysicianName')
                dxm_shell.FieldOfViewHorizontalFlip = ds.dh_getattribute('FieldOfViewHorizontalFlip')
                dxm_shell.FieldOfViewOrigin = ds.dh_getattribute('FieldOfViewOrigin')
                dxm_shell.FieldOfViewRotation = ds.dh_getattribute('FieldOfViewRotation')
                dxm_shell.WindowCenter = ds.dh_getattribute('WindowCenter')
                dxm_shell.WindowWidth = ds.dh_getattribute('WindowWidth')
                dxm_shell.WindowCenterWidthExplanation = ds.dh_getattribute('WindowCenterWidthExplanation')
                dxm_shell.HighBit = ds.dh_getattribute('HighBit')
                dxm_shell.PixelData = ds.PixelData
                dxm_shell.save_as(os.path.join(modified_study_input_dir, f))
            else:
                # Just copy the file as is
                shutil.copy(p, modified_study_input_dir)

        # Deploy
        deployer = Deployer()
        config = deployer.create_config_object(config_factory.params_dict)
        deployer.initialize(config)
        deployer.deploy()

        # Compare results

        modified_results_path = os.path.join(output_dir, "modified", const.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
        modified_results = StudyDeployResults.from_json(modified_results_path)
        assert modified_results.predicted, f"The study {modified_study_input_dir} could not be predicted"

        original_results_path = os.path.join(output_dir, "original", const.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
        original_results = StudyDeployResults.from_json(original_results_path)
        assert original_results.predicted, f"The study {original_study_input_dir} could not be predicted"



        print(f"Comparing results in {original_results_path} to {modified_results_path}...")

        assert results_parser.are_equal(original_results.results, modified_results.results), \
            f"Results do not match.\nOriginal:\n{original_results.results}\nModified:\n{modified_results.results}"

    finally:
        # Remove temp folders
        shutil.rmtree(root_input_dir)
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_T_219("CADt")
    print("OK!")