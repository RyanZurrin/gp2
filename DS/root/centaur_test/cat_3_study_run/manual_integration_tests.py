import os
import tempfile
import numpy as np
import cv2
import pydicom

import deephealth_utils.data.parse_logs as parse_logs
from deephealth_utils.misc import results_parser

from centaur_deploy.config import Config
from centaur_deploy.deploy import Deployer
from centaur_test.data_manager import DataManager
from centaur_engine.helpers.helper_category import CategoryHelper

def test_T_3_MI01():
    """
    Run a prediction for a dbt study in the testing dataset. No PACS - All reports.
    Generate images where bounding box coordinates are plotted on images (taken directly from ds.pixel_array)
    Create the folder manual_integration_output.
        - The images is saved in the manual_integration_output/image_plots
        - The outputs from Centaur are saved in manual_integration_output/centaur_outputs.
    No automated checks done in this function, it only asserts that the generated image was saved.
    """

    # Create ../centaur_test/cat_3_study_run/manual_integration_output/
    manual_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     'manual_integration_output')
    # Make sure it doesn't exist
    assert not os.path.isdir(manual_output_dir), 'Path \'manual_integration_output\' already exists.'
    os.makedirs(manual_output_dir)

    dm = DataManager()
    input_dir = dm.data_studies_dir
    # Just want the DBT study
    input_dir = os.path.join(input_dir, DataManager.STUDY_03_DBT_HOLOGIC)
    output_dir = tempfile.mkdtemp(prefix="temp_output_folder_")

    # Set only parameters of interest. The rest will be None
    config = Config.ProductionConfig()
    config[Config.MODULE_IO, 'input_dir'] = input_dir
    config[Config.MODULE_IO, 'output_dir'] = output_dir

    ### Run the deployer
    deployer = Deployer()
    deployer.initialize(config)
    deployer.logger.info(parse_logs.log_line(-1, "Initialization finished. Deploying..."))
    processed_studies = deployer.deploy(return_results=True)

    # Create ../centaur_test/cat_3_study_run/manual_integration_output/centaur_outputs
    centaur_output_path = os.path.join(manual_output_dir, 'centaur_outputs')
    os.makedirs(centaur_output_path)
    # Create ../centaur_test/cat_3_study_run/manual_integration_output/image_plots
    image_plots_path = os.path.join(manual_output_dir, 'image_plots')
    os.makedirs(image_plots_path)

    # Copy output of Centaur study into
    # ../centaur_test/cat_3_study_run/manual_integration_output/centaur_output
    results_dir = '/'.join(processed_studies[0].study_output_dir.split('/')[0:-1])

    os.system('cp -RT {}/ {}/'.format(results_dir, centaur_output_path))
    # Give permissions
    os.system('chmod 777 -R {}'.format(centaur_output_path))

    # Get results from Centaur
    results_dict = results_parser.load_results_json(
        os.path.join(centaur_output_path,DataManager.STUDY_03_DBT_HOLOGIC, 'results.json'))
    metadata = results_parser.get_metadata(results_dict)

    for idx, row in metadata.iterrows():
        boxes_info = results_dict['model_results']['dicom_results'][row.SOPInstanceUID]['none']
        ds = pydicom.dcmread(row.dcm_path)
        img = ds.pixel_array

        if row.SOPClassUID == '1.2.840.10008.5.1.4.1.1.13.1.3':
            # Find the frame that boxes are found on from results_dict
            # If there are multiple boxes across multiple frames, just choose the first one.
            slice_num = int(boxes_info[0]['slice'])
            img = img[slice_num]

        # Normalize to [0, 255] range
        img = np.multiply(255, (img - np.min(img)) / (np.max(img) - np.min(img)))

        for box_info in boxes_info:
            coords = tuple(map(int, box_info['coords']))
            start_point = tuple(coords[0:2])
            end_point = tuple(coords[2:4])
            # BGR
            if box_info['category'] == CategoryHelper.LOW:  # Yellow
                color = (0, 256, 256)
            elif box_info['category'] == CategoryHelper.INTERMEDIATE:  # Orange
                color = (0, 165, 256)
            elif box_info['category'] == CategoryHelper.HIGH:  # Red
                color = (0, 0, 256)
            else:
                color = (256, 0, 0)  # Blue

            # Line thickness of 3 px
            thickness = 3
            img = cv2.rectangle(img, start_point, end_point, color, thickness)

        assert cv2.imwrite(os.path.join(manual_output_dir, 'image_plots', '{}.png'.format(row.SOPInstanceUID)), img)