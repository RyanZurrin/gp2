import sys
import os
import os.path as osp
import logging
import datetime
from string import Formatter
import hl7
from hl7.client import MLLPClient

import deephealth_utils.misc.results_parser as results_parser
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
import centaur_deploy.constants as const_deploy
from centaur_deploy.deploys.config import Config
from centaur_engine.helpers.helper_category import CategoryHelper
from centaur_reports.helpers.report_helpers import read_json


class BaseHL7:
    """Class for HL7 message management. Must have a template for the HL7 message."""

    HL7_VERSION = '2.8'

    def __init__(self, hl7_template):
        self.TEMPLATE = hl7_template
        self.ATTRIBUTES = self.get_template_keywords(self.TEMPLATE)

        # Fill all attributes with a default value.
        # Example)
        # self._sending_application = ''
        # ...
        # self._deephealth_obx3 = ''
        # self._deephealth_obx5 = ''
        for key in self.ATTRIBUTES:
            self.__setattr__(key, '')

        self._logger = logging.getLogger(self.__class__.__name__)
        self._study_instance_uid = ''

    @property
    def message(self):
        return self.TEMPLATE.format(**self.to_dict())

    def pprint(self):
        return self.message.replace('\r', '\r\n')

    def to_dict(self):
        return {key: self.__getattribute__(key) for key in self.ATTRIBUTES}

    def get_obj(self):
        # print("\n!!!!!!!!! MESSAGE CONVERTING !!!!!!!!!! : \n", self.message.replace("\r", "\r\n"))
        return hl7.parse(self.message)

    @staticmethod
    def get_template_keywords(format_template):
        return [t[1] for t in Formatter().parse(format_template) if t[1] is not None]


class BaseCentaurHL7(BaseHL7):
    """Class for HL7 message. It need
      - centaur_deploy.deploys.config.Config
      - centaur_deploy.deploys.study_deploy_result.StudyDeployResults
    to fill the hl7 fields.
    """

    def __init__(self, templates):

        self.templates = templates

        super().__init__(hl7_template=templates['HL7_TEMPLATE'])

    def fill_all_fields(self, *args, **kwargs):
        """Filling all fields in the HL7 message. """
        raise NotImplementedError("This method should be implemented in a child class")

    def generate_obx3(self, results):
        raise NotImplementedError("This method should be implemented in a child class")

    def generate_obx5(self, results):
        raise NotImplementedError("This method should be implemented in a child class")

    def get_modality(self, df_metadata):
        modality = DicomTypeMap.get_study_type(df_metadata)
        return modality

    def get_value_type(self):
        """See https://hl7-definition.caristix.com/v2/HL7v2.8/Tables/0125"""
        # return 'IS'  # User defined values
        # return 'ST'  # String data
        return 'TX'  # Text data (Display)

    def get_order_control_code(self):
        """See https://hl7-definition.caristix.com/v2/HL7v2.8/Tables/0119"""
        # TODO: Understand meaning with checking the reference
        return 'XO'

    def get_order_status_code(self):
        """See https://hl7-definition.caristix.com/v2/HL7v2.8/Tables/0038"""
        return 'CM'

    def get_current_processing_id(self, config):
        """Get processing ID for current HL7 generation process.
        Please see https://www.hl7.org/documentcenter/public/wg/conf/HL7MSH.htm
        Return: str
            'P' : Production mode
            'D' : Debugging mode
            'T' : Training mode
        """
        # Production mode ## TODO: Just use P or D
        if config[Config.MODULE_DEPLOY, 'debug']:
            return 'D'
        else:
            return 'P'

    @property
    def study_instance_uid(self):
        if not self._study_instance_uid:
            raise ValueError("Not filled values yet!")
        return self._study_instance_uid

    def get_facility_name(self, config):
        return config[Config.MODULE_REPORTS, 'facility_name']

    @staticmethod
    def get_timestamp():
        return datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    def get_patient_id(self, study_metadata):
        return study_metadata['PatientID']

    def get_patient_dob(self, study_metadata):
        return study_metadata['PatientBirthDate']

    def get_patient_gender(self, study_metadata):
        return study_metadata['PatientSex']

    def get_accession_number(self, study_metadata):
        return study_metadata['AccessionNumber']

    def get_patient_first_name(self, study_metadata):
        return study_metadata['PatientFirstName']

    def get_patient_last_name(self, study_metadata):
        return study_metadata['PatientLastName']

    def get_message_control_ID(self, results):
        if results.uid is None:
            return ''
        else:
            return results.uid

    def get_date_time_observation(self, results):
        # TODO: Fix this function to use a timestamp of StudyDeployResult after the StudyDeployResult class have it.
        return self.get_timestamp()

    def is_suspicious(self, results):
        result_dict = results.results
        category = result_dict['study_results']['total']['category']
        bool_result = category >= CategoryHelper.SUSPICIOUS
        return bool_result

    def get_container_id(self, results=None):
        if results is None:
            return os.environ['HOSTNAME']
        else:
            # TODO: Fill proper a container id from the result
            return os.environ['HOSTNAME']



class CADtHL7(BaseCentaurHL7):

    def __init__(self):
        current_template_path = osp.join(osp.dirname(__file__), 'form_CADt.json')
        # templates = read_json('centaur/centaur_reports/hl7/form_CADt.json')
        templates = read_json(current_template_path)

        self.APPLICATION_NAME = templates['APPLICATION_NAME']
        self.OBX5_ERROR_TEMPLATE = templates['OBX5_ERROR_TEMPLATE']
        self.OBX5_SUSPICIOUS_TEMPLATE = templates['OBX5_SUSPICIOUS_TEMPLATE']
        self.OBX5_NON_SUSPICIOUS_TEMPLATE = templates['OBX5_NON_SUSPICIOUS_TEMPLATE']

        super().__init__(templates=templates)

    def fill_all_fields(self, config, results):
        """
        Fill all the HL7 message fields
        Args:
            config (Config): global config object
            results (StudyDeployResults): results object

        Returns:

        """
        # Get common values
        study_metadata = results.study_metadata

        # Please make assignment as a function if it's more than a line.
        self._study_instance_uid = results.get_studyUID()

        self._sending_application = self.APPLICATION_NAME
        self._sending_facility = self.get_facility_name(config)
        self._datetime_of_message = self.get_timestamp()
        self._message_control_ID = self.get_message_control_ID(results)
        self._processing_ID = self.get_current_processing_id(config)
        self._version_ID = self.HL7_VERSION
        self._application_acknowledgement_type = 'AL'

        self._patient_identifier_list = self.get_patient_id(study_metadata)
        self._patient_surname = self.get_patient_last_name(study_metadata)
        self._patient_given_name = self.get_patient_first_name(study_metadata)
        self._patient_DOB = self.get_patient_dob(study_metadata)
        self._patient_gender = self.get_patient_gender(study_metadata)
        self._order_control_code = self.get_order_control_code()
        self._accession_number = self.get_accession_number(study_metadata)
        self._order_status_code = self.get_order_status_code()
        self._value_type = self.get_value_type()
        self._deephealth_obx3 = self.generate_obx3(results)
        self._deephealth_obx5 = self.generate_obx5(results)
        self._date_time_observation = self.get_date_time_observation(results)

    def generate_obx3(self, results):
        """
        Generate obx3 segment
        Args:
            results (StudyDeployResults):  StudyDeployResults object

        Returns:

        """
        if results.is_processed_ok():
            return 'true'
        else:
            return 'false'

    def generate_obx5(self, results):
        if results.is_processed_ok():
            return self.generate_success_obx5(results)
        else:
            return self.generate_fail_obx5(results)

    def generate_success_obx5(self, results):
        if not self.is_suspicious(results):
            return self.generate_non_suspicious_obx5(results)
        else:
            return self.generate_suspicious_obx5(results)

    def generate_fail_obx5(self, results):
        if results.preprocessed and not results.passed_checker_acceptance_criteria():
            # Study failed because the Checker failed
            failed_flags = results.get_failed_acceptance_criteria()
            max_flags = 4
            error_message = "Error Code 002: Study does not pass acceptance criteria. Failed acceptance criteria: {}" \
                .format(failed_flags[:max_flags])

        else:
            # Unexpected error
            error_message = "Error Code 001: An unexpected error occurred during study processing. " \
                            "Please contact DeepHealth support for more information."

        return self.OBX5_ERROR_TEMPLATE.format(
            study_instance_uid=self.study_instance_uid,
            error_message=error_message
        )

    def is_suspicious(self, results):
        result_dict = results.results
        bool_result = result_dict['study_results']['total']['category'] > 0
        return bool_result

    def generate_non_suspicious_obx5(self, results):
        return self.OBX5_NON_SUSPICIOUS_TEMPLATE.format(
            study_instance_uid=self.study_instance_uid,
        )

    def generate_suspicious_obx5(self, results):
        return self.OBX5_SUSPICIOUS_TEMPLATE.format(
            study_instance_uid=self.study_instance_uid,
        )
