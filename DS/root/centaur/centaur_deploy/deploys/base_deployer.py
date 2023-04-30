import argparse
import os
import sys
import time
import traceback
import gc
from collections import OrderedDict

import pandas as pd
import bcolors as bcolors
import pdb
import datetime

# Remove Tensorflow warnings
import warnings

from centaur_io.input.file_input import FileInput
from centaur_io.output.json_output import JsonOutput
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deephealth_utils.data.checker import Checker
from deephealth_utils import get_memory_used, get_study_metadata
from deephealth_utils.data.parse_logs import log_line
from deephealth_utils.misc import results_parser

from centaur_engine.engine import Engine
from centaur_engine.helpers import helper_model
from centaur_engine.models import model_selector
from centaur_engine.preprocessor import Preprocessor
import centaur_engine.helpers.helper_misc as engine_helper

from centaur_reports.report import Report
from centaur_reports.report_images import ImagesReport
from centaur_reports.report_most_malignant_image import MostMalignantImageReport
from centaur_reports.report_pdf import PDFReport
from centaur_reports.report_sr import StructuredReport
from centaur_reports.report_summary import SummaryReport
from centaur_reports.report_cadt_preview import CADtPreviewImageReport
from centaur_reports.report_cadt_hl7 import CADtHL7Report
from centaur_reports.report_msp import MSPReport

import centaur_deploy.constants as const
from centaur_deploy.deploys.config import Config
from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_deploy.heartbeat import HeartBeat


class BaseDeployer(object):

    def __init__(self):
        """
        Constructor
        """
        self._config = None
        # self.io = None
        self.input_object = None
        self.output_objects = None
        self.engine = None
        self.reports = None
        self._logger = None
        self._logger_path = None
        self._output_dir = None
        self.heartbeat = None
        # Metadata fields to process when extracting study metatada info
        self._dicom_fields_to_process = (
            "PatientName", "PatientID", "PatientFirstName", "PatientLastName", "PatientBirthDate", "PatientSex",
            "AccessionNumber", "StudyInstanceUID", "StudyDate")
        self._force_exit = False

    #region Properties

    @property
    def config(self):
        assert self._config is not None, "Deployer not initialized. Please call 'initialize' method"
        return self._config

    @property
    def logger(self):
        """
        Logger object. If not set, a default system logger will be created
        Returns:
            Logger
        """
        if self._logger is None:
            self._logger, self._logger_path = self._get_logger()
        return self._logger

    @property
    def output_dir(self):
        """
        Get the current root output dir based on the Config file
        Returns:
            str
        """
        if self._output_dir is None:
            self._output_dir = self.config[Config.MODULE_IO, 'output_dir']
        return self._output_dir

    # endregion

    # region Initialization

    def get_default_parser(self):
        checker_modes = (Checker.CHECKER_RESEARCH, Checker.CHECKER_PRODUCTION, Checker.CHECKER_DEMO)
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        parser.add_argument('--execution_id', type=str, help='Execution ID used for the container', default='0000')
        parser.add_argument('--client_root_path', type=str, help='Path to the client folder used for Config, etc.')
        # default=const.CLIENT_DIR)
        parser.add_argument('--input_dir', type=str, help='input directory', default=const.DEFAULT_INPUT_FOLDER)
        parser.add_argument('--output_dir', type=str, help='output directory', default=const.DEFAULT_OUTPUT_FOLDER)
        parser.add_argument('--checker_mode', choices=checker_modes,
                            default=Checker.CHECKER_PRODUCTION, help='Checker run mode {}'.format(checker_modes))
        parser.add_argument('--skip_existing', action='store_true',
                            help='skip studies with existing reports')
        parser.add_argument('--debug', action='store_true', help='enable debug mode?')
        # parser.add_argument('--parallelize', action='store_true', default=None, help='enable parallel mode?')
        parser.add_argument('--logging', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Set logging level (DEBUG, INFO...)')
        # parser.add_argument('--pacs', action='store_true', default=None, help='start pacs')
        # parser.add_argument('--ip', type=str, help='IP address of PACS')
        # parser.add_argument('--port', type=str, help='port of PACS')
        parser.add_argument('--exit_on_error', action='store_true',
                            help='when a study does not pass, exit with errors')
        parser.add_argument('--preprocess_only', action='store_true',
                            help='Only run dicom preprocessing (do not score the studies)')
        parser.add_argument('--reports', type=str, nargs="*", choices=self._get_valid_reports(),
                            help='Which reports to produce (overriding default ones!). Allowed values: {}'.format(
                                  BaseDeployer._get_valid_reports()))
        parser.add_argument('--model_version', type=str, help='model version')
        parser.add_argument('--checks_to_ignore', type=str, nargs="*",
                            help='additional checks to ignore. Ex: FAC-20 FAC-30')
        parser.add_argument('--checks_to_include', type=str, nargs="*",
                            help='additional checks to include. Ex: FAC-20 FAC-30')
        parser.add_argument('--save_to_disk', action='store_true', default=False,
                            help='store preprocessed dicom pixel data in disk?')
        parser.add_argument('--config_file_path', type=str,
                            help='Path to a previously created config file that will be used to parametrize')
        # parser.add_argument('--count_studies', action='store_true',
        #                     help='Use os.listdir() to count the number of studies and files before running engine')
        parser.add_argument('--save_run', action='store_true', default=False,
                            help='Save the deploy state in /root/runs')
        parser.add_argument('--load_run', type=str, default=False, help='Load the deploy state')
        parser.add_argument('--continue_log', type=str, help='Will concatenates previous log')
        parser.add_argument('--keep_temp_numpy', action='store_true', help='Keep temporary numpy arrays')
        parser.add_argument('--save_synthetics', action='store_true', help='save synthetic 3d files?')
        parser.add_argument('--keep_temp_pdf_files', action='store_true',
                            help='Keep temporary html files for pdf report')

        cadt_choices = [x.split('.')[0] for x in os.listdir(const.THRESHOLDS_PATH+'/cadt')] + ['current']
        cadx_choices = [x.split('.')[0] for x in os.listdir(const.THRESHOLDS_PATH + '/cadx')] + ['current']
        parser.add_argument('--cadt_threshold_versions', type=str, default='current',
                            help='thresholds for cadt (ex: "000")', choices=cadt_choices)
        parser.add_argument('--cadx_threshold_versions', type=str, default='current',
                            help='threholds for cadx (ex: "000")', choices=cadx_choices)
        parser.add_argument('--intended_workstation', type=str, choices=['eRad', 'ThreePalm'], default='eRad',
                            help='specify the work station on which the CAD SR will be displayed on')
        parser.add_argument('--remove_input_files', action='store_true', help='remove input files')

        return parser

    def parse_args(self):
        parser = self.get_default_parser()
        return parser.parse_args()

    def create_config_object(self, args_dict):
        """
        Create a Config object properly initialized from a dictionary (usually read from the argument parser)
        Args:
            args_dict (dictionary): dictionary of parameters

        Returns:
            Config: instance of the Config class
        """
        c = Config(overriden_params=args_dict)
        return c

    def create_input_object(self, config):
        """
        Get the object that manages the input
        Args:
            config:

        Returns:

        """
        return FileInput(config[Config.MODULE_IO, 'input_dir'], logger=self.logger)

    def create_output_objects(self, config):
        """
        Get an Ordered dictionary to handle outputs.
        Ordered dictionary is used in case a required output order is needed
        Returns:
            OrderedDict with:
                - Key (str): output type ("JSON" for regular json file, "PACS", etc.)
                - Value (Output): Instance of the Output class
        """
        return OrderedDict({
            'JSON': JsonOutput(self.logger)
        })

    def create_study_results_object(self, original_study_results=None):
        """
        Create a StudyRunResults object with the type that this class manages (either blank or as a copy)
        Args:
            original_study_results (StudyDeployResults): StudyRunResults (or any child subclass) object

        Returns:
            StudyRunResults
        """
        if original_study_results is not None:
            return original_study_results
        return StudyDeployResults()

    def initialize(self, config):
        """
        Initialize based on Config object
        Args:
            config (Config): config object
        """
        self._config = config

        self.input_object = self.create_input_object(config)

        pp = Preprocessor(self.config[Config.MODULE_ENGINE])
        pp.set_logger(self.logger)

        model_version = helper_model.get_actual_version_number('model', self.config[Config.MODULE_ENGINE, 'model_version'])
        self.config[Config.MODULE_ENGINE, 'model_version'] = model_version
        model_hashes = helper_model.get_model_file_hashes()
        self.config[Config.MODULE_ENGINE, 'model_hashes'] = model_hashes
        cadt_version_tmp = self.config[Config.MODULE_DEPLOY, 'cadt_threshold_versions']
        cadx_version_tmp = self.config[Config.MODULE_DEPLOY, 'cadx_threshold_versions']
        cadt_version = helper_model.get_actual_version_number('cadt', cadt_version_tmp)
        cadx_version = helper_model.get_actual_version_number('cadx', cadx_version_tmp)
        model = model_selector.ModelSelector.select(model_version, cadt_version, cadx_version, logger=self.logger)

        # If CADt operating points have been passed, replace the default model values with the ones specified.
        # Otherwise, just use the default model ones with the operating point key stored in the Config object
        if config[Config.MODULE_ENGINE, 'cadt_operating_point_values'] is None:
            # Pick the ones from the model
            op_point_key = config[Config.MODULE_ENGINE, 'cadt_operating_point_key']
            op_point_dxm = model.config['cadt_thresholds'][DicomTypeMap.DXM][op_point_key]
            op_point_dbt = model.config['cadt_thresholds'][DicomTypeMap.DBT][op_point_key]
            config.set_cadt_operating_points(op_point_dxm, op_point_dbt, explicitly_set=False)

        self.engine = Engine(preprocessor=pp, model=model, reuse=False,
                             save_to_ram=self.config[Config.MODULE_ENGINE, 'save_to_ram'],
                             logger=self.logger, config=self.config[Config.MODULE_ENGINE])

        self.output_objects = self.create_output_objects(config)
        rp = OrderedDict()
        reports = self.config[Config.MODULE_DEPLOY, 'reports']
        valid_reports = self._get_valid_reports()

        # assert len(reports) > 0, "At least one report class required"
        for r in reports:
            if r not in valid_reports:
                raise ValueError("Report needs to be in {} but is {}".format(valid_reports, r))

            if r == Report.SR:
                rp[Report.SR] = StructuredReport(self.config.get_algorithm_version(),\
                                                 self.config[Config.MODULE_DEPLOY, 'run_mode'],\
                                                 logger=self.logger)
            elif r == Report.PDF:
                rp[Report.PDF] = PDFReport(model, self.config, logger=self.logger)
            elif r == Report.SUMMARY:
                rp[Report.SUMMARY] = SummaryReport(self.output_dir, logger=self.logger)
            elif r == Report.MOST_MALIGNANT_IMAGE:
                rp[Report.MOST_MALIGNANT_IMAGE] = MostMalignantImageReport(model, self.config, logger=self.logger)
            elif r == Report.CADT_PREVIEW:
                rp[Report.CADT_PREVIEW] = CADtPreviewImageReport(model, self.config, logger=self.logger)
            elif r == Report.CADT_HL7:
                rp[Report.CADT_HL7] = CADtHL7Report(self.config, logger=self.logger)
            elif r == Report.MSP:
                rp[Report.MSP] = MSPReport(self.config.get_algorithm_version(), logger=self.logger)
            else:
                raise ValueError("Unexpected report type: {}".format(r))

        self.reports = rp

    # endregion

    # region Main methods

    def run_study(self, study_deploy_results):
        """
        Process, evaluate and generate reports for a study
        Args:
            study_deploy_results (StudyDeployResults): StudyDeployResults object that will be updated with the results
                                                       of the processing
        """
        self.study_chosen_for_processing(study_deploy_results)
        try:
            if not self.config[Config.MODULE_DEPLOY, 'keep_temp_numpy']:
                self.logger.info(log_line(-1, "Cleaning engine..."))
                self.engine.clean()
            # Call the garbage collector manually to avoid memory leaks with pydicom/arrays
            gc.collect()

            results_json_file = os.path.join(study_deploy_results.output_dir, const.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
            if os.path.isfile(results_json_file) and self.config[Config.MODULE_DEPLOY, 'skip_existing']:
                # The study has been already processed. Skip
                study_deploy_results.unexpected_error(f"The study {results_json_file} has been already processed")
                return study_deploy_results

            self.engine.set_file_list(study_deploy_results.input_files)
            self.engine.set_output_dir(study_deploy_results.output_dir)
            study_name = os.path.basename(study_deploy_results.input_dir)

            self.logger.info(log_line(-1, "Engine is preprocessing files..."))
            t1 = time.time()
            accepted = self.engine.preprocess(input_dir=self.config[Config.MODULE_IO, 'input_dir'])
            t2 = time.time()
            self.study_preprocessed(study_deploy_results)
            self.logger.info(log_line(4, "Study_Preprocessing:{}s;MEM:{}".format(t2 - t1, get_memory_used()), study_name))

            if not accepted:
                self.logger.info(
                    log_line(-1, "The study did not pass the validations. Prediction skipped. Failed checks: {}".
                             format(self.engine.preprocessor.get_failed_checks()), study_name))

            if self.config[Config.MODULE_DEPLOY, 'preprocess_only']:
                # Stop processing the study
                return study_deploy_results

            if accepted:
                # Evaluate the study
                self.logger.info(log_line(-1, "Engine is evaluating..."))
                t1 = time.time()
                self.engine.evaluate()
                t2 = time.time()
                self.study_evaluated(study_deploy_results)
                self.logger.info(
                    log_line(4, "Study_Evaluate:{}s;MEM:{}".format(t2 - t1, get_memory_used()), study_name))
        except:
            # Unexpected error
            study_deploy_results.unexpected_error()
            # If the study metadata could not be extracted, try to do it now
            if study_deploy_results.study_metadata is None:
                try:
                    study_deploy_results.study_metadata = get_study_metadata(
                        dicom_fields=self._dicom_fields_to_process,
                        file_list=study_deploy_results.input_files)
                except:
                    msg = "Error when extracting the study metadata for study '{}' in an 'except' block.\n".\
                            format(study_deploy_results.input_dir)
                    msg += traceback.format_exc()
                    study_deploy_results.unexpected_error(msg)
                    self.logger.error(log_line(5, msg))


        # Generate report_results
        # Calculate the reports that should be generated based on the result of the study
        report_results = {}
        reports_dict = self._filter_reports_to_send(study_deploy_results)
        study_deploy_results.reports_expected = list(reports_dict.keys())
        for name, report in reports_dict.items():
            try:
                self.logger.info(log_line(-1, "Generating a {} report...".format(name)))
                t1 = time.time()
                additional_params = {}

                if (isinstance(report, ImagesReport) or isinstance(report, MSPReport)) and \
                        self.config[Config.MODULE_ENGINE, 'save_to_ram']:
                    additional_params['ram_images'] = self.engine.preprocessor.get_pixel_data()

                if isinstance(report, StructuredReport):
                    additional_params['intended_workstation'] = self.config[Config.MODULE_REPORTS, 'intended_workstation']
                report_result = report.generate(study_deploy_results, **additional_params)
                t2 = time.time()
                self.logger.info(log_line(4, "Study_Report_{}:{}s;MEM:{}".format(name, t2 - t1, get_memory_used()),
                                          study_name))
                self.logger.info(log_line(-1, "{} generated".format(report_result)))
                report_results[name] = report_result
            except:
                exc_msg = traceback.format_exc()
                study_deploy_results.unexpected_error(exc_msg)
                report_results[name] = {'ERROR': exc_msg}
                self.logger.error(log_line(5, exc_msg, study_name))

        study_deploy_results.reports_generated = report_results
        self.reports_generated(study_deploy_results)

        if not self.config[Config.MODULE_DEPLOY, 'keep_temp_numpy'] \
           and not self.config[Config.MODULE_ENGINE, 'save_synthetics']:
            self.engine.clean()

    def deploy(self, return_results=False, study_callback=None):
        """
        Main deploy method
        Args:
            return_results (bool): the method returns a list of StudyResults objects (one per study processed)
            study_callback (function): function to call after every study is processed

        Returns:
            List-of-StudyResult
        """
        start = time.time()
        assert self.config is not None, "Config not set. Please use the 'initialize' method before deploying"

        # debug = self.config[Config.MODULE_DEPLOY, 'debug']
        exit_on_error = self.config[Config.MODULE_DEPLOY, 'exit_on_error']
        if return_results:
            processed_studies = []

        # Save the current configuration parameters
        full_config_path = os.path.join(self.config[Config.MODULE_IO, 'output_dir'], const.CENTAUR_CONFIG_JSON)
        if os.path.exists(full_config_path) and not self.config[Config.MODULE_DEPLOY, 'debug']:
            reply = str(
                input('Config file already exists in the output directory. Override? (y/n)? ')).lower().strip()
            if reply[0] != 'y':
                print('Not overwriting config.json, exiting...')
                sys.exit(1)

        with open(full_config_path, "w") as fp:
            fp.write(self.config.to_json())

        if self.config[Config.MODULE_DEPLOY, 'error_tracking']:
            error_tracking_df = pd.DataFrame(columns=['study_path', 'file_path', 'in_results', 'unexpected_errors'])

        # Log number of studies and files
        # if self.config[Config.MODULE_DEPLOY, 'count_studies']:
        #     self.logger.info(log_line('0.1.1', '{} Studies Counted'.format(self.io.count_studies())))
        #     self.logger.info(log_line('0.1.2', '{} Files Counted'.format(self.io.count_files())))

        # Start heartbeat as a thread
        if self.config[Config.MODULE_DEPLOY, "use_heartbeat"]:
            heartbeat_log_filepath = os.path.join(self.config[Config.MODULE_IO, 'output_dir'],
                                                  self.config[Config.MODULE_DEPLOY, 'heartbeat_log'])
            heartbeat_period_in_seconds = self.config[Config.MODULE_DEPLOY, 'heartbeat_period_in_seconds']
            self.heartbeat = HeartBeat(heartbeat_log_filepath, heartbeat_period_in_seconds)
            self.heartbeat.start()

        # Run studies
        try:
            for (study_name, file_list, study_id) in self.input_object.get_next_study():
                assert isinstance(file_list, list), f"Filelist for study {study_name} is not valid: {file_list}"

                study_deploy_results = self.create_study_results_object()
                study_deploy_results.input_dir = os.path.dirname(file_list[0]) if len(file_list) > 0 \
                                                 else os.path.join(self.input_object.ingress, study_name)

                study_deploy_results.output_dir = os.path.join(self.output_dir, study_name)
                study_deploy_results.input_files = file_list
                study_deploy_results.studies_db_id = study_id

                t1 = time.time()

                try:
                    file_list_dict = {}
                    assert os.path.isdir(study_deploy_results.input_dir), f"Folder {study_deploy_results.input_dir} not found"
                    assert len(file_list) > 0, f"Folder {study_deploy_results.input_dir} contains no input files"

                    self.logger.info(log_line(-1, "Running on study: {}".format(study_deploy_results.input_dir)))
                    if self.config[Config.MODULE_DEPLOY, 'error_tracking']:
                        # Insert the raw list of files to process in the error tracking dataframe
                        ix = len(error_tracking_df)
                        for f in study_deploy_results.input_files :
                            error_tracking_df.loc[ix] = [study_deploy_results.input_dir, f, False, None]
                            file_list_dict[f] = ix
                            ix += 1

                    t1 = time.time()

                    self.run_study(study_deploy_results)
                    # This line is required for compatibility with different run modes
                    study_deploy_results = self.create_study_results_object(original_study_results=study_deploy_results)
                    t2 = time.time()
                    self.logger.info(log_line(4, "Study_Total:{}s;MEM:{}".format(t2 - t1, get_memory_used()),
                                                    study_name))

                    finished_ok = study_deploy_results.is_completed_ok() \
                        if not self.config[Config.MODULE_DEPLOY, 'preprocess_only'] \
                        else study_deploy_results.passed_checker_acceptance_criteria()

                    if finished_ok \
                        and self.config[Config.MODULE_DEPLOY, 'error_tracking'] \
                        and study_deploy_results.predicted:
                            results_dict = study_deploy_results.results
                            dicom_results = results_parser.get_dicom_results(results_dict)
                            metadata = study_deploy_results.metadata
                            for sop_instance_uid in dicom_results.keys():
                                row_meta = metadata.loc[metadata['SOPInstanceUID'] == sop_instance_uid]
                                assert len(row_meta) == 1, \
                                    "Expected a single row in metadata for SOPInstanceUID={}; Got: {}".format(
                                        sop_instance_uid, row_meta)
                                file_path = row_meta.iloc[0]['dcm_path']
                                f_ix = file_list_dict[file_path]
                                error_tracking_df.loc[f_ix, 'in_results'] = True

                    if return_results:
                        processed_studies.append(study_deploy_results)

                except:
                    exc_msg = traceback.format_exc()
                    study_deploy_results.unexpected_error()

                    if self.config[Config.MODULE_DEPLOY, 'error_tracking']:
                        # Log error in error_tracking_df
                        error_tracking_df.loc[error_tracking_df['study_path'] == study_deploy_results.input_dir,
                                              'unexpected_errors'] = exc_msg

                    # if debug:
                    #     extype, value, tb = sys.exc_info()
                    #     traceback.print_exc()
                    #     pdb.post_mortem(tb)
                finally:
                    # Send outputs / Save results
                    self.send_study_output(study_deploy_results)

                    # Record in the logger any errors that happened when sending the outputs
                    for output_key, output in study_deploy_results.output_send_results.items():
                        if 'ERROR' in output:
                            self.logger.error(log_line(5, f"Output '{output_key}' failed: {output['ERROR']}", study_name))

                    self.study_output_sent(study_deploy_results)

                    study_summary = "Study_Total:{}s;MEM:{}".format(time.time() - t1, get_memory_used())

                    if study_deploy_results.is_completed_ok():
                        self.logger.info(log_line(-1, bcolors.OK))
                        self.logger.info(log_line(-1, "Study successfully completed. " + study_summary, study_name))
                    else:
                        self.logger.info(log_line(-1, bcolors.FAIL))
                        if study_deploy_results.has_unexpected_errors():
                            self.logger.error(log_line(5, study_deploy_results.get_error_message()))
                        self.logger.warning(log_line(-1, "Study finished with errors. " + study_summary, study_name))
                    self.logger.info(log_line(-1, bcolors.END))

                    if self.config[Config.MODULE_IO, "remove_input_files"]:
                        self._clean_input_files(study_deploy_results)

                    if self.config[Config.MODULE_DEPLOY, 'error_tracking']:
                        error_tracking_df.to_csv(os.path.join(self.config[Config.MODULE_IO, "output_dir"],
                                                              const.CENTAUR_ERRORS_TRACKING))
                    if study_callback is not None:
                        study_callback(study_deploy_results)

                    self.study_finished(study_deploy_results)

                    if exit_on_error and study_deploy_results.has_unexpected_errors():
                        raise Exception("Unexpected error occurred")

                    if self._force_exit:
                        # Exit was forced externally
                        break

            #### End of main deploy loop
            d = datetime.timedelta(seconds=time.time() - start)
            self.logger.info(log_line(4, "DEPLOY_TOTAL_RUN:{}s;MEM:{}".format(d, get_memory_used())))
            if return_results:
                return processed_studies
        except:
            # Unexpected error. Log exception and exit
            exc = traceback.format_exc()
            self.logger.exception(log_line(5, f"Unexpected crash in main thread: {exc}"))
            raise

    def send_study_output(self, study_deploy_results):
        """
        Send results for each defined output object.
        By default, it raises an exception with the first failed output found
        Args:
            study_deploy_results (StudyDeployResults): study results

        Returns:
            study_deploy_results.output_objects is modified in place so that the 'JSON' output contains the
            most updated information with the following data:
            OrderedDict of dicts:
                Key (str): output object id
                Value (dict): {'CODE': result_code, 'ERROR': error_message}
        """
        output_objects = self._filter_outputs_to_send(study_deploy_results)
        for key, output in output_objects.items():
            try:
                study_deploy_results.output_send_results[key] = output.send_results(study_deploy_results)
            except:
                msg = f"Unexpected error when sending output '{key}':\n"
                msg += traceback.format_exc()
                self.logger.error(log_line(5, msg))
                study_deploy_results.unexpected_error(msg)

    def stop_deploy(self):
        self._force_exit = True

    # endregion

    #region Workflow actions

    def study_chosen_for_processing(self, study_deploy_results):
        """
        A new study has been picked for processing.
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        # Create output folder for the study if it doesn't exist yet
        if study_deploy_results.output_dir is not None:
            os.makedirs(study_deploy_results.output_dir, exist_ok=True)

        # Validate the study has input files
        assert study_deploy_results.input_files is not None and len(study_deploy_results.input_files) > 0, \
            "No input files received"

        study_deploy_results.reports_expected = list(self.reports.keys())
        study_deploy_results.uid = study_deploy_results.studies_db_id

    def study_preprocessed(self, study_deploy_results, fail_if_missing_field=True):
        """
        Preprocessing finished for a study
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        study_deploy_results.preprocessed = True
        study_deploy_results.metadata = self.engine.preprocessor.get_metadata()
        study_deploy_results.set_checker_results(self.engine.preprocessor.checker_passed(),
                                                 self.engine.preprocessor.get_failed_checks())

        study_deploy_results.study_metadata = get_study_metadata(dicom_fields=self._dicom_fields_to_process,
                                                                 file_list=study_deploy_results.input_files,
                                                                 metadata_df=study_deploy_results.metadata,
                                                                 fail_if_missing_field=fail_if_missing_field)

    def study_evaluated(self, study_deploy_results):
        """
        Evaluation finished for a study
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        study_deploy_results.predicted = True
        study_deploy_results.results_raw = self.engine.results_raw
        study_deploy_results.results = self.engine.results

    def reports_generated(self, study_deploy_results):
        """
        The report_results have been generated for a study
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        pass

    def study_output_sent(self, study_deploy_results):
        """
        The outputs for the study have been sent
        Args:
            study_deploy_results (StudyDeployResults):
        """
        study_deploy_results.outputs_sent = True

    def study_finished(self, study_deploy_results):
        """
        A study has finished
        Args:
            study_deploy_results (StudyDeployResults): StudyDeployResults object with all the information for the study
        """
        pass

    #endregion

    # region Aux methods

    @classmethod
    def remove_argument(cls, arg_parser, argument):
        """
        Remove an argument from the passed ArgumentParser (if not found, do nothing)
        Args:
            arg_parser (ArgumentParser): argument parser
            argument (str): name of the argument to remove
        """
        for i in range(len(arg_parser._actions)):
            if arg_parser._actions[i].dest == argument:
                del arg_parser._actions[i]
                return

    def _get_logger(self):
        return engine_helper.create_logger(
            self.output_dir, return_path=True,
            centaur_logging_level=self.config[Config.MODULE_DEPLOY, 'logging'],
            continue_log=self.config[Config.MODULE_DEPLOY, 'continue_log'])

    @classmethod
    def _get_valid_reports(self):
        """
        Return allowed values for "reports" parameter

        Returns:
            list of str
        """
        return [Report.PDF, Report.SR, Report.SUMMARY, Report.MOST_MALIGNANT_IMAGE,
                Report.CADT_PREVIEW, Report.CADT_HL7, Report.MSP]

    def _filter_reports_to_send(self, study_deploy_results):
        """
        Filter the reports that should be sent based on the current config and the current study results
        Args:
            study_deploy_results (StudyDeployResults): current results object

        Returns:
            dictionary
        """
        # Default: just use all the configured reports
        return self.reports

    def _filter_outputs_to_send(self, study_deploy_results):
        """
        Filter the reports that should be sent based on the current config and the current study results
        Args:
            study_deploy_results (StudyDeployResults): current results object

        Returns:
            dictionary
        """
        # Default: just use all the configured reports
        return self.output_objects

    def _clean_input_files(self, study_deploy_results):
        """
        Remove the input files for a study.
        Folders are removed only if they are empty after removing the files
        Args:
            study_deploy_results (StudyDeployResults): StudyDeployResults object
        """
        study_name = "UNKNOWN"
        try:
            folders = set()
            study_name = study_deploy_results.get_study_dir_name()
            for f in study_deploy_results.input_files:
                folders.add(os.path.dirname(f))
                os.remove(f)
            for f in folders:
                if len(os.listdir(f)) == 0:
                    os.rmdir(f)
                else:
                    self.logger.warning(
                        log_line(5, f"Folder {f} in study {study_name} is not empty and it could not be removed"))
        except:
            self.logger.error(log_line(5, f"Error cleaning input files for study {study_name}:\n" + traceback.format_exc()))
    # endregion
