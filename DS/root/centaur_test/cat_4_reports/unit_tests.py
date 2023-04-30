import datetime
import os
import re
import shutil
import tempfile
import pandas as pd
import pydicom
import pytest
import logging

from bs4 import BeautifulSoup

import centaur_engine.constants as const_eng
import centaur_deploy.constants as const_deploy
from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_deploy.deploys.config import Config
from centaur_engine.helpers import helper_model
from centaur_reports.report_cadt_hl7 import CADtHL7Report
from centaur_test.data_manager import DataManager
from centaur_reports.report_pdf import PDFReport
from centaur_reports.report_sr import StructuredReport
from centaur_engine.models.model_selector import ModelSelector
import deephealth_utils.misc.results_parser as parser
from centaur_reports.report_summary import SummaryReport
from centaur_test import utils
from deephealth_utils.misc import results_parser
from centaur_engine.helpers.helper_category import CategoryHelper
from centaur_reports.hl7.hl7_controller import CADtHL7

@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    """
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    """
    record_xml_attribute("classname", "4_Unit")

# def test_T_4_U01():
#     """
#     Generate a PDF report with the results file in all the dataset studies.
#     This test only checks that the both the pdf and the dicom pdf are generated and the temp files are removed
#     """
#     data_manager = DataManager()
#     studies = data_manager.get_all_studies()
#     centaur_model_version = helper_model.get_agg_real_model_version()
#     centaur_model = ModelSelector.select(centaur_model_version)
#     results_dict = utils.read_result_files_fixed()
#     config = Config()
#     for study in studies:
#         output_dir = tempfile.mkdtemp()
#         try:
#             results = results_dict[study]
#             report = PDFReport(centaur_model, config)
#             results_report = report.generate(results, output_dir)
#             dicom_output = results_report['dcm_output']
#             pdf_output = results_report['pdf_output']
#             assert os.path.isfile(pdf_output), 'Report {} not found'.format(pdf_output)
#             assert os.path.isfile(dicom_output), 'Report {} not found'.format(dicom_output)
#             # Ensure temp files have been removed
#             assert not os.path.isdir(report._temp_html_folder), "Report temp files where not removed"
#
#             # # Validate some DICOM fields using the results metadata as a reference
#             # metadata = parser.get_metadata(results)
#             # dcm_report = pydicom.dcmread(dicom_output)
#             # row = metadata.iloc[0]
#             # patient_name = ' '.join(row['PatientName']['components'])
#             # assert dcm_report.PatientName == patient_name, "PatientName in the report ({}) do not match metadata ({})".\
#             #     format(dcm_report.PatientName, patient_name)
#             #
#             # for attr in ('StudyInstanceUID', 'PatientID'):
#             #     origin = row[attr]
#             #     rep = dcm_report.get(attr)
#             #     assert origin == rep, "{} attribute in the report ({}) do not match metadata ({})".\
#             #         format(attr, dcm_report.PatientName, patient_name)
#             # # Make sure the Report date is today
#             # assert dcm_report.InstanceCreationDate == datetime.datetime.today().strftime("%Y%m%d"), "Incorrect date in the Report"
#         finally:
#             shutil.rmtree(output_dir)

# def test_T_4_U02():
#     """
#     Generate a Structured report with the results file in every study.
#     This test only checks that the files are generated
#     """
#     data_manager = DataManager()
#     studies = data_manager.get_all_studies()
#     results_dict = utils.read_result_files_fixed()
#     for study in studies:
#         output_dir = tempfile.mkdtemp()
#         try:
#             results = results_dict[study]
#             report = StructuredReport()
#             results_report = report.generate(results, output_dir)
#             report_file = results_report['output_file_path']
#             assert os.path.isfile(report_file), 'Report {} not found'.format(report_file)
#
#             # Validate some DICOM fields using the results metadata as a reference
#             dcm_report = pydicom.dcmread(report_file)
#             metadata = parser.get_metadata(results)
#             row = metadata.iloc[0]
#             patient_name = row['PatientName']
#             assert dcm_report.PatientName == patient_name, "PatientName in the report ({}) do not match metadata ({})". \
#                 format(dcm_report.PatientName, patient_name)
#
#             for attr in ('StudyInstanceUID', 'PatientID'):
#                 origin = row[attr]
#                 rep = dcm_report.get(attr)
#                 assert origin == rep, "{} attribute in the report ({}) do not match metadata ({})". \
#                     format(attr, dcm_report.PatientName, patient_name)
#
#             # Make sure the Report date is today
#             assert dcm_report.ContentDate == datetime.datetime.today().strftime("%Y%m%d"), "Incorrect date in the Report"
#         finally:
#             # Remove output folder
#             shutil.rmtree(output_dir)

# def test_T_4_U03():
#     """
#     Generate a Structured Report with the results file in every study.
#     Compare the results to a baseline
#     """
#     data_manager = DataManager()
#     studies = data_manager.get_all_studies()
#     results_dict = utils.read_result_files_fixed()
#     for study in studies:
#         output_dir = tempfile.mkdtemp()
#         try:
#             # Read results
#             results = results_dict[study]
#             report = StructuredReport()
#             results_report = report.generate(results, output_dir)
#             report_file = results_report['output_file_path']
#             assert os.path.isfile(report_file), 'Report {} not found'.format(report_file)
#
#             dcm_report = pydicom.dcmread(report_file)
#             study_sopinstance = str(dcm_report.StudyInstanceUID)
#             # Compare fields with baseline
#             p = "{}/{}/{}{}{}".format(data_manager.baseline_dir_centaur_output, study,
#                                                 report.file_name_prefix, study_sopinstance, report.file_name_suffix)
#
#             assert os.path.isfile(p), "Report file not found in {}".format(p)
#             baseline_report = pydicom.dcmread(p)
#             allowed_diffs = (
#                 '(0008, 0018)', # SOP Instance UID  (report),
#                 '(0040, a121)', # Date
#                 '(0040, a122)', # Time
#                 '(0008, 0023)', # Content Date
#                 '(0008, 0033)', # Content Time
#                 '(0020, 000e)', # Series Instance UID
#                 '(0002, 0003)', # Media Storage SOP Instance UID
#             )
#             differences = utils.get_dataset_differences(dcm_report, baseline_report, allowed_diffs=allowed_diffs)
#             assert len(differences) == 0, "Differences found in Structured Report:\n{}".format("\n".join(differences))
#
#             # Make sure the Report date is today
#             assert dcm_report.ContentDate == datetime.datetime.today().strftime("%Y%m%d"), "Incorrect date in the Report"
#         finally:
#             # Remove output folder
#             shutil.rmtree(output_dir)

def test_T_176():
    """
    Generate a SummaryReport with the results file in all the dataset studies.
    """
    data_manager = DataManager()
    studies = data_manager.get_all_studies()
    output_dir = tempfile.mkdtemp()
    report = SummaryReport(output_dir)
    results_dict = utils.read_result_files_fixed()

    try:
        total_num_images = 0
        num_studies = 0
        df_studies = df_images = pd.DataFrame()
        # Read baseline results
        for study in studies:
            results = results_dict[study]
            if not results.passed_checker_acceptance_criteria():
                continue

            metadata = results.metadata
            num_studies += 1
            total_num_images += len(metadata)

            # Generate report
            results_report = report.generate(results)
            studies_path = results_report['studies_path']
            dicoms_path = results_report['dicoms_path']
            dicoms_agg_path = results_report['dicoms_agg_path']

            # Read the current content of the report
            df_studies = pd.read_csv(studies_path, index_col=0)
            df_images = pd.read_csv(dicoms_path, index_col=0)
            df_bbxs = pd.read_pickle(dicoms_agg_path)

            assert len(df_studies) == num_studies, "Current number of studies in the SummaryReport: {}. Expected: {}".\
                format(len(df_studies), num_studies)
            assert len(df_images) == total_num_images, "Current number of images in the SummaryReport: {}. Expected: {}".\
                format(len(df_images), total_num_images)
            assert len(df_bbxs['StudyInstanceUID'].unique()) == num_studies, \
                "Current number of studies in the SummaryReport: {}. Expected: {}".format(
                    len(df_bbxs['StudyInstanceUID'].unique()), num_studies)


            # Make sure that the study and the images are in the report
            study_id = metadata.iloc[0]['StudyInstanceUID']
            assert study_id in df_studies.index, "Study {} not found in the SummaryReport".format(study)

            for sop_instance_uid in metadata['SOPInstanceUID']:
                assert report.get_img_index(study_id, sop_instance_uid) in df_images.index, \
                    "Image {} not found in the Report".format(sop_instance_uid)

        # Ensure there are not null values in the Report
        nulls = df_studies.isna().count()
        for col in df_studies.columns:
            assert nulls[col] == len(df_studies), "Column '{}' in Studies Dataframe contains null values".format(col)

        nulls = df_images.isna().count()
        for col in df_images.columns:
            assert nulls[col] == len(df_images), "Column '{}' in Images Dataframe contains null values".format(col)
    finally:
        shutil.rmtree(output_dir)

# def test_T_4_U05():
#     """
#     Check that a html report does meet all the expectations including format
#     """
#     data_manager = DataManager()
#     centaur_model_version = helper_model.get_agg_real_model_version()
#     centaur_model = ModelSelector.select(centaur_model_version)
#
#     # Read results for DBT (the most complete study)
#     results_file_path = os.path.join(data_manager.baseline_dir_centaur_output, DataManager.STUDY_03_DBT_HOLOGIC, const_eng.CENTAUR_RESULTS_JSON)
#     assert os.path.isfile(results_file_path), "Results file not found in {}".format(results_file_path)
#
#     # Fix metadata paths that are needed to access the numpy arrays to generate images
#     results = results_parser.load_results_json(results_file_path)
#     metadata = results_parser.get_metadata(results)
#     metadata['dcm_path'] = (data_manager.data_studies_dir + "/") + metadata['dcm_path']
#     metadata['np_paths'] = (data_manager.baseline_dir + "/") + metadata['np_paths']
#     results['metadata'] = metadata.to_json()
#     config = Config()
#     output_dir = tempfile.mkdtemp()
#     try:
#         report = PDFReport(centaur_model, config)
#         # First, generate the original report keeping html intermediate files
#         results_report = report.generate(results, output_dir, clean_temp_files=False)
#         dicom_output = results_report['dcm_output']
#         pdf_output = results_report['pdf_output']
#         # Ensure all the files are there
#         assert os.path.isfile(pdf_output), 'Report {} not found'.format(pdf_output)
#         assert os.path.isfile(dicom_output), 'Report {} not found'.format(dicom_output)
#         assert os.path.isdir(report._temp_html_folder), "The intermediate html files folder seems to be removed ({})".\
#                                                         format(report._temp_html_folder)
#         # Parse customized texts and make sure length is not too long
#         with open(report._temp_html_report, 'r') as fp:
#             soup = BeautifulSoup(fp, 'lxml')
#         # Check patient info section
#         patient_info_div = soup.find(id="dh-patient-info-div")
#         for tag in patient_info_div.find_all("h6"):
#             assert len(tag.text.strip()) <= report.max_length_texts['patient_info'], \
#                 "Field {} has a length bigger than allowed ({}). Full text: '{}'".format(
#                     tag.attrs['id'], report.max_length_texts['patient_info'], tag.text.strip())
#
#         for tag in soup.find_all(id=re.compile("dh-findings")):
#             assert len(tag.text.strip()) <= report.max_length_texts['slice_findings'], \
#                 "Field {} has a length bigger than allowed ({}). Full text: '{}'".format(
#                     tag.attrs['id'], report.max_length_texts['slice_findings'], tag.text.strip())
#
#         # TODO: additional checks when different images are not available
#         # # Validate some DICOM fields using the results metadata as a reference
#         # metadata = parser.get_metadata(results)
#         # dcm_report = pydicom.dcmread(dicom_output)
#         # row = metadata.iloc[0]
#         # patient_name = ' '.join(row['PatientName']['components'])
#         # assert dcm_report.PatientName == patient_name, "PatientName in the report ({}) do not match metadata ({})".\
#         #     format(dcm_report.PatientName, patient_name)
#         #
#         # for attr in ('StudyInstanceUID', 'PatientID'):
#         #     origin = row[attr]
#         #     rep = dcm_report.get(attr)
#         #     assert origin == rep, "{} attribute in the report ({}) do not match metadata ({})".\
#         #         format(attr, dcm_report.PatientName, patient_name)
#         # # Make sure the Report date is today
#         # assert dcm_report.InstanceCreationDate == datetime.datetime.today().strftime("%Y%m%d"), "Incorrect date in the Report"
#     finally:
#         shutil.rmtree(output_dir)

@pytest.mark.cadt
def test_T_190():
    """
    ONLY FOR Saige-Q. Ensure the Preview image is generated for a Suspicious case and it's not generated otherwise
    """
    dm = DataManager()
    run_mode = const_deploy.RUN_MODE_CADT
    dm.set_baseline_params(run_mode)
    suspicious_studies = dm.get_suspicious_studies(run_mode)
    all_studies = dm.get_valid_studies(run_mode=run_mode)

    for study in all_studies:
        results_path = os.path.join(dm.baseline_dir_centaur_output, study)
        results_json_path = os.path.join(results_path, const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
        result_dict = StudyDeployResults.from_json(results_json_path).results

        if study in suspicious_studies:
            assert any(['DH_SaigeQ_Preview' in x for x in os.listdir(results_path)]), \
                "study {} should have produced a CADt Preview Image but did not".format(study)
            sus_code = result_dict['study_results']['total']['category']
            assert sus_code == CategoryHelper.SUSPICIOUS, "{} should be suspicious ".format(study)

        else:
            assert not any(['DH_SaigeQ_Preview' in x for x in os.listdir(results_path)]), \
                "study {} should not have produced a CADt Preview Image but did ".format(study)

            if study != dm.STUDY_02_GE:
                sus_code = result_dict['study_results']['total']['category']
                assert sus_code == CategoryHelper.NOT_SUSPICIOUS, "{} should not be suspicious ".format(study)

@pytest.mark.cadt
def test_T_196_1():
    """
    Test 1  for VER-196. Ensure that an output HL7 message for a "Suspicious" study is generated correctly.
    """
    logger = logging.getLogger("centaur." + __name__)

    # Preparing JSON result data
    data_manager = DataManager()
    data_manager.set_baseline_params(const_deploy.RUN_MODE_CADT)
    results_file_path = os.path.join(
        data_manager.baseline_dir_centaur_output,
        DataManager.STUDY_01_HOLOGIC,
        const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(results_file_path), "Results file not found in {}".format(results_file_path)

    # Converting JSON -> StudyDeployResults
    results = StudyDeployResults.from_json(results_file_path)
    results.output_dir = tempfile.mkdtemp()
    logger.debug(results.get_studyUID())
    assert results.is_completed_ok()

    # HL7 generator
    hl7_obj = CADtHL7()
    hl7_obj._study_instance_uid = results.get_studyUID()

    # Check HL7 controller from the hologic_01 case
    assert hl7_obj.get_message_control_ID(results) == ''
    assert hl7_obj.generate_obx3(results) == 'true'
    assert hl7_obj.is_suspicious(results)
    expected_prefix_obx5 = '1.2.826.0.1.3680043.9.3218.1.1.100473303.1355.1542898025071.71.0^Saige-Q: Suspicious'
    assert hl7_obj.generate_suspicious_obx5(results) == expected_prefix_obx5
    assert hl7_obj.generate_success_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5
    assert hl7_obj.generate_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5

    # Gerate HL7 Report as a text file
    config = Config()
    hl7_report = CADtHL7Report(config, logger=logger)
    hl7_report.hl7.fill_all_fields(config, results)
    hl7_report.save_to_file(results.output_dir, hl7_report.hl7)

    # Check a HL7 Report from the hologic_01 case with an expected result.
    expected_filename = 'DH_SaigeQ_hl7_1.2.826.0.1.3680043.9.3218.1.1.100473303.1355.1542898025071.71.0.txt'
    assert os.path.isfile(os.path.join(results.output_dir, expected_filename))

    str_from_file = open(os.path.join(results.output_dir, expected_filename), 'r', newline='\n').read()
    logger.debug(str_from_file.replace("\r", "\n"))

    expected_start_str = "MSH|^~\&|Saige-Q|Unknown|||"
    expected_end_str = "PID|1||PID123456789||Deep^Angelina^^^||19551115|F|||||||||||||||||||||||||\rORC|XO|AN000001" \
                       "|||CM||||||||||||||||||1\rOBX|1|TX|true||1.2.826.0.1.3680043.9.3218.1.1.100473303.1355.1542898025071.71.0^Saige-Q: Suspicious||||||F|||"
    assert expected_start_str in str_from_file
    assert expected_end_str in str_from_file
    assert hl7_report.hl7.message == str_from_file

@pytest.mark.cadt
def test_T_U196_2():
    """
    Test 2 for VER-196. Ensure that an output HL7 message for a "Suspicious" study is generated correctly.
    """
    logger = logging.getLogger("centaur." + __name__)

    data_manager = DataManager()
    data_manager.set_baseline_params(const_deploy.RUN_MODE_CADT)
    results_file_path = os.path.join(
        data_manager.baseline_dir_centaur_output,
        DataManager.STUDY_03_DBT_HOLOGIC,
        const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(results_file_path), "Results file not found in {}".format(results_file_path)

    results = StudyDeployResults.from_json(results_file_path)
    results.output_dir = tempfile.mkdtemp()
    assert results.passed_checker_acceptance_criteria()
    assert results.predicted
    assert results.reports_generated_ok()
    assert results.outputs_sent_ok()
    assert not results.has_unexpected_errors()
    assert results.is_completed_ok()

    hl7_obj = CADtHL7()
    hl7_obj._study_instance_uid = results.get_studyUID()

    assert hl7_obj.get_message_control_ID(results) == ''
    assert hl7_obj.generate_obx3(results) == 'true'
    assert hl7_obj.is_suspicious(results)
    expected_prefix_obx5 = '1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485^Saige-Q: Suspicious'
    assert hl7_obj.generate_suspicious_obx5(results) == expected_prefix_obx5
    assert hl7_obj.generate_success_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5
    assert hl7_obj.generate_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5

    # Gerate HL7 Report as a text file
    config = Config()
    hl7_report = CADtHL7Report(config, logger=logger)
    hl7_report.hl7.fill_all_fields(config, results)
    hl7_report.save_to_file(results.output_dir, hl7_report.hl7)

    # Check a HL7 Report from the hologic_01 case with an expected result.
    expected_filename = 'DH_SaigeQ_hl7_1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485.txt'
    assert os.path.isfile(os.path.join(results.output_dir, expected_filename))

    str_from_file = open(os.path.join(results.output_dir, expected_filename), 'r', newline='\n').read()
    logger.debug(str_from_file.replace("\r", "\n"))

    expected_start_str = "MSH|^~\&|Saige-Q|Unknown|||"
    expected_end_str = "PID|1||PID123456789||Deep^Angelina^^^||19551115|F|||||||||||||||||||||||||\rORC|XO|AN000003|||CM||||||||||||||||||1\rOBX|1|TX|true||1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485^Saige-Q: Suspicious||||||F|||"
    assert expected_start_str in str_from_file
    assert expected_end_str in str_from_file
    assert hl7_report.hl7.message == str_from_file

@pytest.mark.cadt
def test_T_197():
    """
    Test for VER-197. Ensure that an output HL7 message for a "" (blank) study is generated correctly.
    """
    logger = logging.getLogger("centaur." + __name__)

    data_manager = DataManager()
    data_manager.set_baseline_params(const_deploy.RUN_MODE_CADT)
    results_file_path = os.path.join(
        data_manager.baseline_dir_centaur_output,
        DataManager.STUDY_03_DBT_HOLOGIC,
        const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(results_file_path), "Results file not found in {}".format(results_file_path)

    results = StudyDeployResults.from_json(results_file_path)

    # change result to "No suspicious"
    results.results['study_results'] = {
        'total': {
            'score': 0.09766184091568,
            'category': 0,
            'category_name': 'DH-MT: ',
            'postprocessed_percentile_score': 9.0}
    }
    results.output_dir = tempfile.mkdtemp()

    hl7_obj = CADtHL7()
    hl7_obj._study_instance_uid = results.get_studyUID()

    assert hl7_obj.get_message_control_ID(results) == ''
    assert results.is_completed_ok()
    assert hl7_obj.generate_obx3(results) == 'true'
    assert not hl7_obj.is_suspicious(results)
    expected_prefix_obx5 = '1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485^Saige-Q: '
    assert hl7_obj.generate_non_suspicious_obx5(results) == expected_prefix_obx5
    assert hl7_obj.generate_success_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5
    assert hl7_obj.generate_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5

    # Gerate HL7 Report as a text file
    config = Config()
    hl7_report = CADtHL7Report(config, logger=logger)
    hl7_report.hl7.fill_all_fields(config, results)
    hl7_report.save_to_file(results.output_dir, hl7_report.hl7)

    # Check a HL7 Report from the hologic_01 case with an expected result.
    expected_filename = 'DH_SaigeQ_hl7_1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485.txt'
    assert os.path.isfile(os.path.join(results.output_dir, expected_filename))

    str_from_file = open(os.path.join(results.output_dir, expected_filename), 'r', newline='\n').read()
    logger.debug(str_from_file.replace("\r", "\n"))

    expected_start_str = "MSH|^~\&|Saige-Q|Unknown|||"
    expected_end_str = "PID|1||PID123456789||Deep^Angelina^^^||19551115|F|||||||||||||||||||||||||\rORC|XO|AN000003|||CM||||||||||||||||||1\rOBX|1|TX|true||1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485^Saige-Q: ||||||F|||"
    assert expected_start_str in str_from_file
    assert expected_end_str in str_from_file
    assert hl7_report.hl7.message == str_from_file

@pytest.mark.cadt
def test_T_198():
    """
    Test for VER-198. Ensure that an output HL7 message is generated correctly for a study that fails the acceptance criteria.
    """
    logger = logging.getLogger("centaur." + __name__)

    data_manager = DataManager()
    data_manager.set_baseline_params(const_deploy.RUN_MODE_CADT)
    results_file_path = os.path.join(
        data_manager.baseline_dir_centaur_output,
        DataManager.STUDY_02_GE,
        const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(results_file_path), "Results file not found in {}".format(results_file_path)

    results = StudyDeployResults.from_json(results_file_path)
    results.output_dir = tempfile.mkdtemp()
    assert not results.passed_checker_acceptance_criteria()
    assert not results.predicted
    assert results.reports_generated_ok()
    assert results.outputs_sent_ok()
    assert not results.has_unexpected_errors()
    assert not results.is_completed_ok()

    hl7_obj = CADtHL7()
    hl7_obj._study_instance_uid = results.get_studyUID()

    assert hl7_obj.get_message_control_ID(results) == ''
    assert not results.is_completed_ok()
    assert hl7_obj.generate_obx3(results) == 'false'

    try:  # This expression should not be run
        hl7_obj.is_suspicious(results)
    except TypeError as e:
        assert str(e) == "'NoneType' object is not subscriptable"
    except Exception as e:
        logger.error(str(e))
        raise e

    expected_prefix_obx5 = "1.2.826.0.1.3680043.9.7134.2.0.1000.1542349870.784630^Saige-Q: Error Code 002: Study does not pass acceptance criteria. Failed acceptance criteria: ['FAC-140']"
    assert hl7_obj.generate_fail_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5
    assert hl7_obj.generate_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5

    # Gerate HL7 Report as a text file
    config = Config()
    hl7_report = CADtHL7Report(config, logger=logger)
    hl7_report.hl7.fill_all_fields(config, results)
    hl7_report.save_to_file(results.output_dir, hl7_report.hl7)

    # Check a HL7 Report from the hologic_01 case with an expected result.
    expected_filename = 'DH_SaigeQ_hl7_1.2.826.0.1.3680043.9.7134.2.0.1000.1542349870.784630.txt'
    assert os.path.isfile(os.path.join(results.output_dir, expected_filename))

    str_from_file = open(os.path.join(results.output_dir, expected_filename), 'r', newline='\n').read()
    logger.debug(str_from_file.replace("\r", "\n"))

    expected_start_str = "MSH|^~\&|Saige-Q|Unknown|||"
    expected_end_str = "PID|1||PID123456789||Deep^Angelina^^^||19551115|F|||||||||||||||||||||||||\rORC|XO|AN000002|||CM||||||||||||||||||1\rOBX|1|TX|false||1.2.826.0.1.3680043.9.7134.2.0.1000.1542349870.784630^Saige-Q: Error Code 002: Study does not pass acceptance criteria. Failed acceptance criteria: ['FAC-140']||||||F|||"
    assert expected_start_str in str_from_file
    assert expected_end_str in str_from_file
    assert hl7_report.hl7.message == str_from_file


def test_T_199():
    """
    Test for VER-199. Ensure that. an output HL7 message for study that crashed (there was an unexpected error) is generated correctly.
    """
    logger = logging.getLogger("centaur." + __name__)

    data_manager = DataManager()
    data_manager.set_default_baseline_params()
    results_file_path = os.path.join(
        data_manager.baseline_dir_centaur_output,
        DataManager.STUDY_03_DBT_HOLOGIC,
        const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(results_file_path), "Results file not found in {}".format(results_file_path)

    results = StudyDeployResults.from_json(results_file_path)

    # change result to "Fail" and add some error code and messages.
    results.results = None
    results.prediction_ok = False
    results.results_raw = None
    results.generated_reports = {}
    results.reports_ok = False
    results._error_code = 1
    results._error_messages = [
        "Traceback (most recent call last):\n  File \"/root/centaur/centaur_deploy/deploys/base_deployer.py\", line 342, in deploy\n    self.run_study(study_result_info)\n  File \"/root/centaur/centaur_deploy/deploys/base_deployer.py\", line 221, in run_study\n    self.engine.evaluate()\n  File \"/root/centaur/centaur_engine/engine.py\", line 103, in evaluate\n    self.results = self._results_post_processing(self.results_raw, metadata_df)\n  File \"/root/centaur/centaur_engine/engine.py\", line 184, in _results_post_processing\n    self.config['run_mode']))\nAssertionError: Expected run_mode to be one of ['CADx', 'DEMO', 'CADt'], got None.\n"]
    results.studies_db_id = None
    results.uid = None
    results.output_dir = tempfile.mkdtemp()

    hl7_obj = CADtHL7()
    hl7_obj._study_instance_uid = results.get_studyUID()

    assert hl7_obj.get_message_control_ID(results) == ''
    assert not results.is_completed_ok()
    assert hl7_obj.generate_obx3(results) == 'false'

    # Check error message
    expected_prefix_obx5 = "1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485^Saige-Q: Error Code 001: An unexpected error occurred during study processing. Please contact DeepHealth support for more information."
    assert hl7_obj.generate_fail_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5
    assert hl7_obj.generate_obx5(results)[:len(expected_prefix_obx5)] == expected_prefix_obx5

    # Generate HL7 Report as a text file
    config = Config()
    hl7_report = CADtHL7Report(config, logger=logger)
    hl7_report.hl7.fill_all_fields(config, results)
    hl7_report.save_to_file(results.output_dir, hl7_report.hl7)

    # Check a HL7 Report from the hologic_01 case with an expected result.
    expected_filename = 'DH_SaigeQ_hl7_1.2.826.0.1.3680043.9.7134.1.2.0.23161.1505941466.143485.txt'
    assert os.path.isfile(os.path.join(results.output_dir, expected_filename))

    str_from_file = open(os.path.join(results.output_dir, expected_filename), 'r', newline='\n').read()
    logger.debug(str_from_file.replace("\r", "\n"))

    expected_start_str = "MSH|^~\&|Saige-Q|Unknown|||"
    expected_end_str = "PID|1||PID123456789||Deep^Angelina^^^||19551115|F|||||||||||||||||||||||||\rORC|XO|AN000003|||CM"
    assert expected_start_str in str_from_file
    assert expected_end_str in str_from_file
    assert hl7_report.hl7.message == str_from_file

