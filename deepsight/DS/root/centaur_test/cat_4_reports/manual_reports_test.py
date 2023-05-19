import warnings

import pytest

from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
import centaur_deploy.constants as const_deploy
from centaur_test.data_manager import DataManager
from centaur_engine.helpers import helper_model
from centaur_engine.models import model_selector
from centaur_deploy.deploy import Deployer
from centaur_deploy.deploys.config import Config
from centaur_reports.report_cadt_preview import CADtPreviewImageReport
from centaur_reports.report_most_malignant_image import MostMalignantImageReport

from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
import pydicom
import os
import numpy as np
import cv2


def get_model():
	"""
	get the current version of the model
	Returns: model object

	"""
	model_version = helper_model.get_actual_version_number('model', 'current')
	cadt_version = helper_model.get_actual_version_number('cadt', 'current')
	cadx_version = helper_model.get_actual_version_number('cadx', 'current')
	md = model_selector.ModelSelector.select(model_version, cadt_version, cadx_version)
	return md


def modify_study_deploy_results(dicom, sr, output):
	"""
	modify the study_deploy_results so that dicom indicated will have the highest/most suspicious score
	Args:
		dicom: dicom that will have the highest score
		sr: study deploy results instance
		output: output dir of the report

	Returns: None

	"""
	if not os.path.exists(output):
		os.mkdir(output)
	sr.output_dir = output
	sr.results['dicom_results'][dicom]['none'][0]['score'] =1


def convert_dcm_to_png(dcm_path, out_dir):
	"""
	convert the pixels of a dcm file to png file
	Args:
		dcm_path: path to the dcm file
		out_dir: save directory of the png file

	Returns: None

	"""
	ds = pydicom.dcmread(dcm_path)
	img = ds.pixel_array[:, :, 0]
	img = np.stack([img, img, img], axis=-1)
	img = np.multiply(255, (img - np.min(img)) / (np.max(img) - np.min(img)))
	file_name = dcm_path.split('/')[-1]
	cv2.imwrite('{}/{}.png'.format(out_dir,file_name), img)

@pytest.mark.cadt
def test_image_reports(intermediate_outputs_folder):
	"""
	Create image reports for different combinations (L R) / (DXM, BT) most suspicious dicom to be reviewed manually.
	Please note that this test is expected to save its results in a 'test_output' folder in the baseline location.
	Therefore, this test assumes that the 'baseline/run_mode__CADt/test_output' folder exists and it will create a
	'test_image_reports' subfolder there (or it will use an existing one at long as it's empty)

	Args:
		intermediate_outputs_folder (str). Output folder where the output images will be saved for manual review (fixture)

	"""
	dm = DataManager()
	dm.set_baseline_params(const_deploy.RUN_MODE_CADT)

	assert os.path.isdir(intermediate_outputs_folder), \
		f"{intermediate_outputs_folder} folder should exist to save the output results." \
		" Please mount a folder in that location or specify a different value for the" \
		" 'intermediate_results_folder' parameter"

	base_output_dir = os.path.join(intermediate_outputs_folder, "test_image_reports")
	if os.path.isdir(base_output_dir):
		warnings.warn(f"{base_output_dir} already exists. The results will be overwritten")
	else:
		os.mkdir(base_output_dir)

	# get study_results
	study = DataManager.STUDY_03_DBT_HOLOGIC
	baseline_results_path = os.path.join(dm.baseline_dir_centaur_output, study,
										 const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
	study_results = StudyDeployResults.from_json(baseline_results_path)
	# make config
	config_dict = {
		'input_dir': study_results.input_dir,
		'output_dir': study_results.output_dir,
		'run_mode': 'CADx',
		'use_disk': True,
		'algorithm_version': 'current'
	}
	deployer = Deployer()
	config = deployer.create_config_object(config_dict)

	# get model
	model = get_model()

	preview_image_generator = CADtPreviewImageReport(model, config)
	qa_image_generator = MostMalignantImageReport(model, config, draw_box=True)

	df = study_results.metadata.copy()
	df['bt_dxm'] = study_results.metadata.apply(DicomTypeMap.get_type_row, axis=1)

	tmp = df.groupby(['ImageLaterality', 'bt_dxm'])['SOPInstanceUID'].first()

	# CADt Preview images:
	output_preview = os.path.join(base_output_dir, "CADt_preview_images")
	if not os.path.exists(output_preview):
		os.mkdir(output_preview)

	for (laterality, dicom_type) in tmp.keys():
		study_results = StudyDeployResults.from_json(baseline_results_path)
		dicom = tmp[(laterality, dicom_type)]
		description = 'most_suspicous_lesion_on_{}_{}'.format(laterality, dicom_type)
		output = os.path.join(output_preview, description)
		modify_study_deploy_results(dicom, study_results, output)

		out = preview_image_generator.generate(study_results)
		convert_dcm_to_png(out['dcm_output'],output)

	# QA one box preview images

	output_qa = os.path.join(base_output_dir, "qa_preview_images")

	if not os.path.exists(output_qa):
		os.mkdir(output_qa)

	for (laterality, dicom_type) in tmp.keys():
		study_results = StudyDeployResults.from_json(baseline_results_path)
		dicom = tmp[(laterality, dicom_type)]
		description = 'most_suspicous_lesion_on_{}_{}'.format(laterality, dicom_type)
		output = os.path.join(output_qa, description)
		modify_study_deploy_results(dicom, study_results, output)

		out = qa_image_generator.generate(study_results, draw_one_bbx_only=True)
		print(out)

	print(f"All the images were saved to {base_output_dir}")
	warnings.warn("This is a manual test and it requires additional review")

