import json
import os
import traceback
import warnings
from collections import OrderedDict

import pandas as pd

import centaur_deploy.constants as const_deploy


class StudyDeployResults(object):
    def __init__(self, results_file_name=const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON):
        self.input_dir = None                       # Input dir (str)
        self.input_files = []                       # Input files (list-str)
        self.output_dir = None                      # Output dir (str)
        self._metadata = None                       # Metadata (Dataframe)
        self.study_metadata = None                  # Study-level metadata (dict)

        self._checker_validation_passed = False     # The study passed the Checker validations
        self._failed_acceptance_criteria_list = []  # List of failed acceptance criteria list-str
        self.results_raw = None                     # Prediction Results (before post-processing)
        self.results = None                         # Prediction Results (after post-processing)
        self.results_file_name = results_file_name  # Name of the default results json name

        self.reports_expected = []                  # List of keys with the reports expected to be generated for this study
        self.reports_generated = {}                 # Results for each report that should have been generated (dict of dicts)

        self.output_send_results = OrderedDict()    # Results of the output send operation for each output to be sent (OrderedDict of dicts)

        self.preprocessed = False                   # The study was preprocessed (regardless of the result of the preprocessing) (bool)
        self.predicted = False                      # A prediction was made for the study (bool)
        self.outputs_sent = False                   # Sending output process was attempted (bool)

        self._error_code = 0                        # Error code for this study (int)
        self._error_messages = []                   # List of error messages (list-str)
        self.studies_db_id = None                   # Index for StudiesDB object (str) (optional)
        self.uid = None                             # Message UID (UUID) (optional)

    @property
    def metadata(self):
        return self.get_metadata(passed_checker_only=True)

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def to_json(self):
        """
        Get a json representation of the current object
        Returns:
            str
        """
        d = vars(self).copy()
        # Dataframe is not serializable by default. Convert to json first
        if d['_metadata'] is not None:
            d['_metadata'] = d['_metadata'].to_json()
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str_or_file):
        """
        Create an instance of the current class from a json string
        Args:
            json_str_or_file (str): json that represents the object or a path to a json file

        Returns:
            StudyDeployResults (or one of the subclasses) instance
        """
        if os.path.isfile(json_str_or_file):
            with open(json_str_or_file, 'r') as f:
                dict_vars = json.load(f)
        else:
            dict_vars = json.loads(json_str_or_file)

        s = cls()
        keys = vars(s).keys()
        for key in keys:
            if key == '_metadata':
                value = dict_vars.get('_metadata')
                if value is None:
                    value = dict_vars.get('metadata')
                if value is not None:
                    value = pd.read_json(value, dtype=False)
                    s._metadata = value
            else:
                value = dict_vars[key]
                setattr(s, key, value)
        return s

    def save(self, file_path=None):
        """
        Save a json file that contains all the object info
        Args:
            file_path (str): file path. If not specified, the default outputs will be used
        """
        if file_path is None:
            file_path = self.get_default_results_file_path()
        with open(file_path, 'w') as f:
            f.write(self.to_json())

    def get_default_results_file_path(self):
        """
        Return the default expected output file path
        Returns:
            str
        """
        return os.path.join(self.output_dir, self.results_file_name)


    def is_completed_ok(self):
        """
        The study was successfully completed with no errors
        Returns:
            bool
        """
        return self.passed_checker_acceptance_criteria() and self.predicted and self.reports_generated_ok() \
               and self.outputs_sent_ok() and not self.has_unexpected_errors()

    def is_processed_ok(self):
        """
        The study was predicted successfully with no unexpected errors
        Returns:
            bool
        """
        return self.passed_checker_acceptance_criteria() and self.predicted and not self.has_unexpected_errors()

    def get_study_dir_name(self):
        """
        Get the study name based on the input folder that contains the files
        Returns:
            str (None when no input files)
        """
        return os.path.basename(self.input_dir)


    def get_metadata(self, passed_checker_only=False):
        """
        Get a copy of the metadata dataframe
        Args:
            passed_checker_only (bool): return only rows for images that passed the Checker

        Returns:
            Dataframe or None
        """
        if self._metadata is None:
            return None
        return self._metadata[self._metadata['failed_checks'].isnull()].copy() if passed_checker_only else self._metadata.copy()

    def get_error_code(self):
        """
        Get the possible error code for this study processing (0 when no errors)

        Returns:
            int
        """
        return self._error_code

    def get_error_message(self):
        """
        Get the possible error message for the processed study
        Returns:
            str
        """
        return "\n".join(self._error_messages)

    def unexpected_error(self, message=None):
        """
        There was an unexpected error when running the study
        Args:
            message (str): Error message. If None, just use the traceback
        """
        self._error_code = 1
        if message is None:
            message = traceback.format_exc()
        self._error_messages.append(message)

    def has_unexpected_errors(self):
        """
        The study failed in an uncontrolled way
        Returns:
            bool
        """
        return self.get_error_code() == 1

    def set_checker_results(self, study_accepted, failed_flags):
        """
        Set the results of the study validation.
        If failed_flags contains duplicates, they will be removed
        Args:
            study_accepted (bool): the study passed the Checker validation (note that the study could pass the Checker
                                   validations even if some flags were raised
            failed_flags (list of str): list of flags raised by the Checker
        """
        self._failed_acceptance_criteria_list = list(set(failed_flags))
        self._checker_validation_passed = study_accepted
        if not study_accepted:
            self._error_code = 2
            self._error_messages.append("The study did not meet the required acceptance criteria. "
                                        f"Failed flags: {failed_flags}")

    def passed_checker_acceptance_criteria(self):
        """
        Study was preprocessed and passed the Acceptance Criteria
        Returns:
            bool
        """
        if not self.preprocessed:
            return False
        return self._checker_validation_passed

    def get_failed_acceptance_criteria(self):
        """
        List of unique flags raised by the Checker
        Returns:
            list-str
        """
        return self._failed_acceptance_criteria_list

    def reports_generated_ok(self):
        """
        All the expected reports were generated successfully
        Returns:
            bool
        """
        if self.reports_expected is None or self.reports_generated is None:
            # These objects should not be null (they could be empty though)
            return False
        return set(self.reports_expected) == set(self.reports_generated.keys())

    def outputs_sent_ok(self):
        """
        All the outputs were sent ok
        Returns:
            bool
        """
        for result in self.output_send_results.values():
            if 'ERROR' in result:
                return False
        return True


    def get_studyUID(self):
        """
        Get the StudyInstanceUID if the metadata has been properly processed or it can be read from any dicom files
        Returns:
            str
        """
        return self.study_metadata['StudyInstanceUID']

    def get_accessionNumber(self):
        """
        Get the AcessionNumber if the metadata has been properly processed or it can be read from any input files
        Returns:
            str
        """
        # if self.metadata is None or len(self.metadata) == 0:
        #     if allow_empty:
        #         return None
        #     raise AssertionError("Metadata not available. Was the study preprocessed correctly?")
        #
        # return self.metadata.iloc[0]['AccessionNumber']
        return self.study_metadata['AccessionNumber']

    def get_category(self):
        """
        Get the final category for a study (assuming it has been properly processed, otherwise return None)
        Returns:
            int
        """
        try:
            return self.results['study_results']['total']['category']
        except:
            warnings.warn("Study category could not be found")
            return None

    def get_category_name(self):
        """
        Get the final category name for a study (assuming it has been properly processed, otherwise return None)
        Returns:
            str
        """
        try:
            return self.results['study_results']['total']['category_name']
        except:
            warnings.warn("Study category name could not be found")
            return None