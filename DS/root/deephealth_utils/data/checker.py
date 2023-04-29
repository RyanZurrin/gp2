import time

import os
import re
from inspect import getmembers, isfunction
import pandas as pd
import numpy as np

import deephealth_utils
from deephealth_utils.data.checker_functions import CheckFnSelector
from . import utils as dh_utils
from .parse_logs import log_line


class Checker:
    # CONSTANTS
    CHECKER_RESEARCH = "research"
    CHECKER_PRODUCTION = "production"
    CHECKER_DEMO = "demo"

    def __init__(self, config={}):

        self.config = config

        passes_verification, error_message = self.verify_dicom_specs()
        if not passes_verification:
            raise ValueError(error_message)

        self.checks_df = pd.read_csv(os.path.abspath(os.path.dirname(__file__)) +
                                     '/dicom_specs/checks.csv')
        self.modes_df = pd.read_csv(os.path.abspath(os.path.dirname(__file__)) +
                                    '/dicom_specs/checks_modes.csv')

        assert 'checker_mode' in self.config and self.config['checker_mode'] in self.get_allowed_modes(), \
            "checker_mode setting not found or with wrong value in config. Allowed modes: {}\nConfig: {}". \
                format(self.get_allowed_modes(), self.config)

        self._mode = self.config['checker_mode']

        unique_checks = self.modes_df['Acceptance Criteria Number'].unique()
        if 'checks_to_ignore' not in config:
            self.config['checks_to_ignore'] = []
        else:
            for check in self.config['checks_to_ignore']:
                assert check in unique_checks, "{} is not a valid check specified in checks_to_ignore".format(
                    check)

        if 'checks_to_include' not in config:
            self.config['checks_to_include'] = []
        else:
            for check in self.config['checks_to_include']:
                assert check in unique_checks, "{} is not a valid check specified in checks_to_include".format(
                    check)
        if len(set(self.config['checks_to_ignore']).intersection(set(self.config['checks_to_include']))) > 0:
            raise ValueError("ACN is in both include and exclude lists (pick one).")

        self.modes_df = self.modes_df[['Acceptance Criteria Number', self._mode]]
        for i, row in self.modes_df.iterrows():
            acn = row['Acceptance Criteria Number']
            if row[self._mode] == 0 and acn not in self.config['checks_to_include']:
                self.config['checks_to_ignore'].append(acn)

    def get_allowed_modes(self):
        return self.modes_df.columns[2:].to_list()

    @property
    def mode(self):
        return self._mode

    def verify_dicom_specs(self):
        """
        Verifies that:
        1. The CSVs under 'dicom_specs' have the expected columns in the expected order.
        2. The CSVs under 'dicom_specs' have acceptance criteria with valid prefixes.
        3. 'checks.csv' under 'dicom_specs' has valid values inthe 'Input' column.
        4. The function names under the 'Function' column in 'checks.csv' under 'dicom_specs' are actually implemented
        in deephealth_utils.data.checker_functions.
        5. The values under the ['production', 'research', 'demo'] columns in 'checks_modes.csv' under 'dicom_specs'
        are all either 0 or 1.
        6. Tags in 'common.csv', 'hologic.csv', and 'ge.csv' under 'dicom_specs' conform to the dicom tag format.
        7. Anu rules specified in 'common.csv', 'hologic.csv', and 'ge.csv' under 'dicom_specs' are validation rules
        that are in the constant deephealth_utils.data.validation.VALIDATION_RULES.
        """
        dh_utils_path = os.path.dirname(deephealth_utils.__file__)
        dicom_specs_path = os.path.join(dh_utils_path, 'data', 'dicom_specs')

        # Reading in CSVs
        checks_modes_df = pd.read_csv(os.path.join(dicom_specs_path, 'checks_modes.csv'))
        common_df = pd.read_csv(os.path.join(dicom_specs_path, 'common.csv'))
        checks_df = pd.read_csv(os.path.join(dicom_specs_path, 'checks.csv'))
        hologic_df = pd.read_csv(os.path.join(dicom_specs_path, 'hologic.csv'), keep_default_na=False)
        ge_df = pd.read_csv(os.path.join(dicom_specs_path, 'ge.csv'), keep_default_na=False)

        # Make sure first column is AcceptanceCriteria and second is Tag
        if checks_modes_df.columns[0] != 'Acceptance Criteria Number' or checks_modes_df.columns[1] != 'Tag':
            return False, 'First two columns of checks_modes.csv should be "Acceptance Criteria Number" and "Tag", respectively"'

        # Make sure columns in checks.csv have correct column names
        if list(checks_df.columns) != ['AcceptanceCriteria', 'Input', 'Parameters', 'Function']:
            return False, 'Columns of checks.csv should be "[\'AcceptanceCriteria\', \'Input\', \'Parameters\', \'Function\']"'


        # Check the acceptance criteria matches
        acceptance_criteria_lists = {'common.csv': ['-'.join(el.split('-')[0:2]) for el in np.array(common_df['AcceptanceCriteria'])],
                           'checks_modes.csv': np.array(checks_modes_df['Acceptance Criteria Number']),
                           'checks.csv': np.array(checks_df['AcceptanceCriteria']),
                           'hologic.csv': np.array(hologic_df['AcceptanceCriteria']),
                           'ge.csv': np.array(ge_df['AcceptanceCriteria'])}

        # Make sure acceptance criteria column matches form XXX-### and has an accepted prefix
        ac_regex = '^[A-Z]{2,3}-\d+$'
        accepted_categories = ['FAC', 'SAC', 'PAC', 'IAC', 'HOL', 'GE']
        for ac_list in acceptance_criteria_lists:
            for ac in acceptance_criteria_lists[ac_list]:
                if not bool(re.match(ac_regex, ac)) or \
                        not any([ac.startswith(prefix) for prefix in accepted_categories]):
                    return False, 'Invalid CSV: {} has misspecified AC: {}'.format(ac_list, ac)

        # Make sure checks.csv Inputs column are within accepted_inputs
        accepted_inputs = ['ds', 'df', 'fp']
        for input in np.array(checks_df['Input']):
            if input not in accepted_inputs:
                return False, 'Invalid CSV: checks.csv: Input must be in {}, found {} instead'.format(accepted_inputs, input)

        # Make sure functions in checks.csv are actually implemented in checker_functions.py
        # Gather implemented checker_functions from deephealth_utils.data.checker_functions
        checker_functions = [el[0] for el in getmembers(deephealth_utils.data.checker_functions) if isfunction(el[1])]
        for func in np.array(checks_df['Function']):
            if func not in checker_functions:
                return False, 'Invalid CSV: checks.csv: function specified as {} not implemented'.format(func)

        # Make sure production, research, and demo columns are only 0 or 1 (and not NaN)
        for column_name in ['production', 'research', 'demo']:
            for el in checks_modes_df[column_name].unique():
                if el not in [0, 1]:
                    return False, 'Invalid CSV: checks_modes.csv, Value {} found in {} mode column'.format(el, column_name)

        # Make sure Tags in common.csv, hologic.csv, and ge.csv conform to dicom tag format
        tag_df_dict = {'common.csv': common_df, 'hologic.csv': hologic_df, 'ge.csv': ge_df}

        for df in tag_df_dict:
            for tag in np.array(tag_df_dict[df]['Tag']):
                if not dh_utils.verify_tag(tag):
                    return False, 'Invalid CSV: {}, invalid tag found: {}'.format(df, tag)

        # Make sure Rule is in VALIDATION_RULES from validation.py
        from deephealth_utils.data.validation import VALIDATION_RULES


        for df in tag_df_dict:
            for column_name, column_data in tag_df_dict[df].iteritems():
                if column_name.endswith('-Rule'):
                    for rule in column_data:
                        if rule not in VALIDATION_RULES:
                            return False, 'Invalid CSV: {}, invalid rule found: {}'.format(df, rule)

        return True, None

    def get_check_fn(self, fn_name):
        """
        Given the name of a checker function defined under deephealth_utils.data.checker_functions, returns the function.
        :param fn_name: str. The name of the function (for example, 'patient_age_not_less_than').
        :return: function. The function matching the specified function name (for example,
        deephealth_utils.data.checker_functions.patient_age_not_less_than).
        """
        return CheckFnSelector.get_fn(fn_name)

    def get_check_input(self, input_str, input_dict):
        if input_str in input_dict:
            return input_dict[input_str]
        else:
            raise ValueError("Input not valid for checks")

    def check_file_size(self, file_path):
        """
        Implements FAC-150 as acceptance criteria.
        Exists as separate function because we want to check file size before any other check.
        """
        check = self.get_check_fn(self.checks_df[self.checks_df['AcceptanceCriteria'] == 'FAC-150']['Function'].iloc[0])
        passed, file_size = check(file_path)
        return passed, {'FAC-150': file_size}

    def check_study_size(self, file_list):
        """
        Implements SAC-120 as acceptance criteria.
        Exists as separate function because we want to check all file sizes before any other check.
        """
        check = self.get_check_fn(self.checks_df[self.checks_df['AcceptanceCriteria'] == 'SAC-120']['Function'].iloc[0])
        passed, file_size = check(file_list)
        return passed, {'SAC-120': file_size}

    @staticmethod
    def checks_that_need_pixels():

        return ['IAC-20', 'IAC-30']

    def check_file(self, file_path, return_ds=False, external_metadata=None, logger=None):
        """
        Takes in file_path and iterates through checks.

        Collects failed_checks_dict (dict).
            Key: Acceptance Criteria (eg. FAC-80)
            Value: Violating Field (if available)
        A check consists of a checker function and inputs. eval() is called to collect the required inputs.
        For example, if a checker function takes in 'ds', eval('ds') will provide 'ds'.

        Notes:
            Ignores FAC-150 since it is already done through check_file_size()
            FAC-10 checks whether file can be read by pydicom. Therefore, if it fails, we break and stop other checks.
            Acceptance criteria 'FAC-130' is actually a group of checks, so we iterate and update failed checks.
            We only want to read pixels if we have a check that requires
        """
        checks_df_dict = {check_type: self.checks_df[self.checks_df['AcceptanceCriteria'].str.contains(check_type)]
                          for check_type in ['FAC', 'IAC']}

        # Ignoring FAC-150 since we have already done check_file_size()
        checks_df_dict['FAC'] = checks_df_dict['FAC'][checks_df_dict['FAC'].AcceptanceCriteria != 'FAC-150']

        t1 = time.time()
        try:
            ds = dh_utils.dh_dcmread(file_path, external_metadata=external_metadata, stop_before_pixels=True)
        except:
            return False, {'FAC-10': file_path}, None  # If file cannot be read by Pydicom, return immediately.

        t2 = time.time()
        if logger is not None:
            logger.info(log_line(4, "Image_DicomHeaderRead:{}s;MEM:{}".format(t2 - t1, dh_utils.get_memory_used()),
                                 os.path.basename(file_path)))
        pixel_data_loaded = False

        input_dict = {
            'ds': ds,
            'fp': file_path,
            'df': None
        }
        failed_checks_dict = {}

        # Sort the checks so that checks which do not require pixels occur first.
        all_file_checks_df = pd.concat([checks_df_dict['FAC'], checks_df_dict['IAC']])
        all_file_checks_df = all_file_checks_df.sort_values(by=['AcceptanceCriteria'],
                                                            key=lambda ac: ac.isin(self.checks_that_need_pixels()),
                                                            ascending=True)

        for idx, row in all_file_checks_df.iterrows():
            acceptance_criteria = row['AcceptanceCriteria']
            if 'checks_to_ignore' in self.config:
                if acceptance_criteria in self.config['checks_to_ignore']:
                    continue
            if acceptance_criteria in self.checks_that_need_pixels() and not pixel_data_loaded:
                # Non-pixel checks happen first, so if the code reaches here, all file checks that don't require pixel
                # data have been run.
                if len(failed_checks_dict) == 0:
                    # Pre-read pixel array so that we can compute the time
                    if logger is not None:
                        t1 = time.time()
                    ds = dh_utils.dh_dcmread(file_path, external_metadata=external_metadata, stop_before_pixels=False)
                    # Check if PixelData is in ds first before trying to call ds.pixel_array
                    if hasattr(ds, 'PixelData'):
                        try:
                            _ = ds.pixel_array
                        except AttributeError as e:  # If Pydicom cannot convert due to missing DICOM attributes
                            failed_checks_dict.update({'FAC-101': e})
                    else:
                        # If PixelData is not in ds, then return FAC-100
                        return False, {'FAC-100': file_path}, None
                    if logger is not None:
                        t2 = time.time()
                        logger.info(log_line(4, "Image_DicomPixelArrayRead:{}s;MEM:{}".
                                             format(t2 - t1, dh_utils.get_memory_used()),
                                             os.path.basename(file_path)))
                    input_dict['ds'] = ds
                    pixel_data_loaded = True
                # If non-pixel checks did not pass, pixel data will not be read and pixel checks will not be run.
                else:
                    continue

            inputs = self.get_check_input(row['Input'], input_dict)
            check = self.get_check_fn(row['Function'])
            if acceptance_criteria == 'FAC-130':
                # Update failed_checks_dict with rule-based checks
                failed_checks_dict.update(check(inputs))
                # Remove ones in checks_to_ignore
                for k in [v for v in failed_checks_dict.keys() if v in self.config['checks_to_ignore']]:
                    del failed_checks_dict[k]
            else:
                if row['Parameters'] is not None and row['Parameters'] == row['Parameters']:
                    parameters = row['Parameters'].split(',')
                    passed, field = check(inputs, *parameters)
                else:
                    passed, field = check(inputs)
                if not passed:
                    failed_checks_dict.update({acceptance_criteria: field})

        file_passed = len(failed_checks_dict) == 0

        # If there are no checks that require pixels, the pixel data will not be read by default, so force a read.
        if file_passed and set(all_file_checks_df['AcceptanceCriteria']).intersection(
                set(self.checks_that_need_pixels())) == set():
            ds = dh_utils.dh_dcmread(file_path, stop_before_pixels=False)
            input_dict['ds'] = ds

        if return_ds:
            return file_passed, failed_checks_dict, input_dict['ds']

        return file_passed, failed_checks_dict, None

    def check_study(self, study_df):
        """
        Takes in a study_df and iterates through checks.

        Collects failed_checks_dict (dict).
            Key: Acceptance Criteria (eg. FAC-80)
            Value: Violating Field (if available)

        A check consists of a checker function and inputs. eval() is called to collect the required inputs.
        For example, if a checker function takes in 'ds', eval('ds') will provide 'ds'.

        Notes:
            Ignores FAC-150 since it is already done through check_file_size()
            FAC-10 checks whether file can be read by pydicom. Therefore, if it fails, we break and stop other checks.
            Acceptance criteria 'FAC-130' is actually a group of checks, so we iterate and update failed checks.
        """

        checks_df = self.checks_df[self.checks_df['AcceptanceCriteria'].str.contains('|'.join(['PAC', 'SAC']))]
        checks_df = checks_df[checks_df.AcceptanceCriteria != 'SAC-120']

        input_dict = {
            'ds': None,
            'fp': None,
            'df': study_df
        }

        failed_checks_dict = {}

        for idx, row in checks_df.iterrows():
            acceptance_criteria = row['AcceptanceCriteria']
            if acceptance_criteria not in self.config['checks_to_ignore']:

                input_type = self.get_check_input(row['Input'], input_dict)
                check = self.get_check_fn(row['Function'])
                if row['Parameters'] is not None and row['Parameters'] == row['Parameters']:
                    parameters = row['Parameters'].split(',')
                    passed = check(input_type, *parameters)
                else:
                    passed = check(input_type)
                if not passed[0]:
                    failed_checks_dict.update({acceptance_criteria: passed[1]})

        study_passed = len(failed_checks_dict) == 0

        return study_passed, failed_checks_dict
