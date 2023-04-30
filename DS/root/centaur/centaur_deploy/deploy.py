import datetime
import json
import logging
import os
import traceback
import warnings
from collections import OrderedDict

from centaur_deploy.deploys.studies_db import StudiesDB
from centaur_engine.helpers.helper_category import CategoryHelper
from centaur_io.input.demo_file_input import DemoFileInput
from centaur_io.input.file_input import FileInput
from centaur_io.input.dir_monitor_input import DirMonitorInput
from centaur_io.input.pacs_input import PACSInput
from centaur_io.output.json_output import JsonOutput
from centaur_io.output.pacs_output import PACSOutput
from centaur_io.output.ris_output import RISOutput
from centaur_io.output.intermediate_results_file_output import IntermediateResultsFileOutput
from centaur_reports.report import Report
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from deephealth_utils.data.parse_logs import log_line
from centaur_deploy.run import Run
from centaur_deploy.deploys.base_deployer import BaseDeployer
from centaur_deploy.deploys.config import Config
import centaur_deploy.constants as const_deploy


class Deployer(BaseDeployer):

    def __init__(self):
        super().__init__()
        self.studies_db = None

    #region Initialization

    def parse_args(self):
        """
        Parse command line parameters
        Returns:
            Arguments
        """
        parser = super().get_default_parser()
        valid_run_modes = (const_deploy.RUN_MODE_CADT, const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_DEMO)

        # Add additional parameters
        parser.add_argument("--run_mode", choices=valid_run_modes, default=const_deploy.RUN_MODE_CADX,
                            help="Product type {}".format(valid_run_modes))
        parser.add_argument("--cadt_operating_point_key", choices=Config.CADT_OPERATING_POINT_KEYS,
                            help="Name of operating point to use with CADt run_mode")
        parser.add_argument("--cadt_operating_point_values", type=float, nargs=2,
                            help="Specific CADt operating points for DXM and DBT")
        parser.add_argument("--use_heartbeat", type=str, choices=('y', 'n'), help="Use heartbeat service (y|n)")
        parser.add_argument("--product_label", type=str, choices=("Saige-Q", "Saige-Dx"),
                            help="Show the product label only (do not deploy)")
        parser.add_argument('--monitor_input_dir_continuously', '-mondir', action='store_true', default=False,
                            help="Monitor the input folder constantly for new studies")
        args = parser.parse_args()
        return args

    def create_config_object(self, args_dict):
        """
        Create a Config object properly initialized from a dictionary (usually read from the argument parser).
        This method also sets some default properties of the Config object based on the run_mode and others
        Args:
            args_dict (dictionary): dictionary of parameters (normally read from argparse)

        Returns:
            Config: instance of the Config class
        """
        # Special case for backwards compatibility purposes: save_to_ram
        if 'save_to_disk' in args_dict:
            args_dict['save_to_ram'] = not args_dict['save_to_disk']
            del args_dict['save_to_disk']
        if 'run_mode' not in args_dict:
            args_dict['run_mode'] = const_deploy.RUN_MODE_CADX
        if 'load_run' not in args_dict:
            args_dict['load_run'] = False

        config = super().create_config_object(args_dict)

        if 'client_root_path' in args_dict:
            # Client config
            client_config_file = os.path.join(args_dict['client_root_path'], const_deploy.CLIENT_CONFIG_FILE_NAME)
            assert os.path.isfile(client_config_file), f"{client_config_file} not found"
            self.update_config_with_external_params(config, client_config_file)
            # The command line parameters still have priority over the client file
            config.override_default_values(args_dict)

        # Heartbeat only used by default in CADt, but it can be set manually
        if 'use_heartbeat' in args_dict:
            config[Config.MODULE_DEPLOY, "use_heartbeat"] = True if args_dict['use_heartbeat'].lower() == 'y' else False
        else:
            # Heartbeat enabled in CADt by default
            config[Config.MODULE_DEPLOY, "use_heartbeat"] = args_dict['run_mode'] == const_deploy.RUN_MODE_CADT

        # Default reports (use only when 'reports' parameter is not passed)
        if 'reports' not in args_dict:
            if args_dict['run_mode'] == const_deploy.RUN_MODE_CADT:
                config[Config.MODULE_DEPLOY, 'reports'] = [Report.SUMMARY, Report.CADT_PREVIEW, Report.CADT_HL7,
                                                           Report.MOST_MALIGNANT_IMAGE, Report.SR]
            elif args_dict['run_mode'] == const_deploy.RUN_MODE_DEMO:
                config[Config.MODULE_DEPLOY, 'reports'] = [Report.SUMMARY]
            else:
                # CADx
                config[Config.MODULE_DEPLOY, 'reports'] = [Report.SUMMARY, Report.SR, Report.PDF, Report.MSP]

        if args_dict['run_mode'] == const_deploy.RUN_MODE_DEMO:
            config[Config.MODULE_DEPLOY, 'error_tracking'] = False

        if args_dict['load_run']:
            config = Run.load_run(run_num=args.load_run,
                                  model_version=config[Config.MODULE_ENGINE, 'model_version'])
        return config

    def initialize(self, config):
        """
        Initialize all the required deploy variables based on a Config object created from the command line arguments
        Args:
            config (Config): Config object
        """
        self._output_dir = config[Config.MODULE_IO, 'output_dir']
        assert self._output_dir is not None, "output_dir is required"
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        elif not config[Config.MODULE_DEPLOY, 'debug']:
            assert len(os.listdir(self._output_dir)) == 0, "Output directory {} is not empty".format(self._output_dir)

        input_dir_exists = os.path.isdir(config[Config.MODULE_IO, 'input_dir'])
        input_file_exists = os.path.isfile(config[Config.MODULE_IO, 'input_dir'])
        assert input_dir_exists or input_file_exists, \
            f'Input folder/file {config[Config.MODULE_IO, "input_dir"]} was not found'

        if config.get_run_mode() == const_deploy.RUN_MODE_DEMO:
            self._dicom_fields_to_process = ("StudyInstanceUID",)

        super(self.__class__, self).initialize(config)

    def create_input_object(self, config):
        """
        Create an Input object based on the current config
        Args:
            config (Config): config object

        Returns:
            Subclass of Input
        """
        if config[Config.MODULE_IO, 'pacs_receive'] or self.config[Config.MODULE_IO, 'monitor_input_dir_continuously']:
            # StudiesDB and InputDirMonitor needed in these scenarios
            self.studies_db = StudiesDB(os.path.join(config[Config.MODULE_IO, 'output_dir'],
                                                     const_deploy.STUDIES_DB_FILE_NAME))
            if config[Config.MODULE_IO, 'pacs_receive']:
                # Receive studies from PACS
                input_object = PACSInput(config[Config.MODULE_IO, 'input_dir'],
                                         config[Config.MODULE_IO, 'pacs_receive_port'],
                                         config[Config.MODULE_IO, 'pacs_receive_ae'],
                                         self.studies_db,
                                         config[Config.MODULE_IO, 'new_study_threshold_seconds'],
                                         logger=self.logger)
                # Start the async DICOM receiver
                input_object.start_receiver()
            else:
                # Just listen to a folder
                input_object = DirMonitorInput(config[Config.MODULE_IO, 'input_dir'], self.studies_db,
                                               logger=self.logger)
        elif self.config.get_run_mode() == const_deploy.RUN_MODE_DEMO:
            input_object = DemoFileInput(config[Config.MODULE_IO, 'input_dir'], logger=self.logger)
        else:
            # Default behavior, just local folders
            input_object = FileInput(config[Config.MODULE_IO, 'input_dir'], logger=self.logger)

        return input_object

    def create_output_objects(self, config):
        """
        Create a dictionary of output objects
        Args:
            config (Config): config instance

        Returns:
            OrderedDict with:
                - Key (str): output type ("JSON" for regular json file, "PACS", etc.)
                - Value (Output): Instance of the Output class
        """
        output_objects = OrderedDict()
        if config[Config.MODULE_IO, 'pacs_send']:
            output_objects['PACS'] = PACSOutput(config[Config.MODULE_IO, 'pacs_send_ip'],
                                                config[Config.MODULE_IO, 'pacs_send_port'],
                                                config[Config.MODULE_IO, 'pacs_send_ae'])

        if config[Config.MODULE_IO, 'ris_send']:
            output_objects['RIS'] = RISOutput(config[Config.MODULE_IO, 'ris_send_ip'],
                                              config[Config.MODULE_IO, 'ris_send_port'])

        if config[Config.MODULE_ENGINE, 'save_synthetics']:
            output_objects['INTERMEDIATE_OUTPUTS'] = IntermediateResultsFileOutput(self.engine, save_synthetics=True)

        output_objects['JSON'] = JsonOutput(self.logger)

        return output_objects

    # endregion

    # region Main methods

    def send_study_output(self, study_deploy_results):
        """
        Send results for each defined output.
        Args:
            study_deploy_results (StudyDeployResults): study results

        Returns:
            study_deploy_results.output_objects is modified in place so that the 'JSON' output contains the
            most updated information with the following data:
            OrderedDict of dicts:
                Key (str): output object id
                Value (int,str): result_code, error_message
        """
        if self.config[Config.MODULE_DEPLOY, 'run_mode'] == const_deploy.RUN_MODE_CADT:
            study_deploy_results.output_send_results = OrderedDict({})
            output_objects = self._filter_outputs_to_send(study_deploy_results)

            k = 'PACS'
            try:
                if k in output_objects:
                    # Send the preview image to a PACS
                    if study_deploy_results.get_category() == CategoryHelper.SUSPICIOUS:
                        assert Report.CADT_PREVIEW in study_deploy_results.reports_generated, "Preview image was not generated"
                        sc_file = study_deploy_results.reports_generated[Report.CADT_PREVIEW]['dcm_output']
                        assert os.path.isfile(sc_file), f"Preview image file not found in {sc_file}"
                        study_deploy_results.output_send_results[k] = output_objects[k].send_results(sc_file)

                    assert Report.SR in study_deploy_results.reports_generated, "Structured Report was not generated"
                    sr_file = study_deploy_results.reports_generated[Report.SR]["output_file_path"]
                    assert os.path.isfile(sr_file), f"Preview image file not found in {sr_file}"
                    study_deploy_results.output_send_results[k] = output_objects[k].send_results(sr_file)

            except:
                exc_msg = traceback.format_exc()
                study_deploy_results.unexpected_error(exc_msg)
                self.logger.error(log_line(5, f"Unexpected error with output '{k}':\n{exc_msg}"))

            k = 'RIS'
            try:
                if k in output_objects:
                    # Send output to RIS
                    assert Report.CADT_HL7 in study_deploy_results.reports_generated, "HL7 output message was not generated"
                    hl7_file = study_deploy_results.reports_generated[Report.CADT_HL7]['output']
                    study_deploy_results.output_send_results[k] = output_objects[k].send_results(hl7_file)
            except:
                exc_msg = traceback.format_exc()
                study_deploy_results.unexpected_error(exc_msg)
                self.logger.error(log_line(5, f"Unexpected error with output '{k}':\n{exc_msg}"))

            # The results are saved to a file at the end of the process to capture possible errors in the other outputs
            k = 'JSON'
            try:
                if k in output_objects:
                    # Write results to a json file
                    study_deploy_results.output_send_results[k] = output_objects[k].send_results(study_deploy_results)
            except:
                exc_msg = traceback.format_exc()
                study_deploy_results.unexpected_error(exc_msg)
                self.logger.error(log_line(5, f"Unexpected error with output '{k}':\n{exc_msg}"))
        else:
            # Just default behaviour for now
            super(self.__class__, self).send_study_output(study_deploy_results)

    # endregion

    # region Workflow actions

    def study_chosen_for_processing(self, study_deploy_results):
        """
        A new study has been picked for processing
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        if self.config.get_run_mode() != const_deploy.RUN_MODE_DEMO:
            # Ensure all the files belong to the same folder
            input_folders = set([os.path.dirname(f) for f in study_deploy_results.input_files])
            assert len(input_folders) == 1, \
                "All the files should be in the same folder. Got: {}. Filelist: {}".format(input_folders,
                                                                                       study_deploy_results.input_files)

        super(self.__class__, self).study_chosen_for_processing(study_deploy_results)
        if self.studies_db:
            code = 1
            params = {
                'processing_status_code': code,
                'processing_status_text': self.studies_db.CODE_MAPPINGS[('processing_status_code', code)],
                'output_path': study_deploy_results.output_dir
            }
            self.studies_db.update_study(study_deploy_results.studies_db_id, params)

    def study_preprocessed(self, study_deploy_results):
        """
        Preprocessing finished for a study
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        super(self.__class__, self).study_preprocessed(study_deploy_results,
                                       fail_if_missing_field=self.config.get_run_mode() != const_deploy.RUN_MODE_DEMO)
        if self.studies_db:
            # Update the metadata
            params = {
                'study_instance_uid': study_deploy_results.get_studyUID(),
                'accession_number': study_deploy_results.get_accessionNumber()
            }
            self.studies_db.update_study(study_deploy_results.studies_db_id, params)


    def study_evaluated(self, study_deploy_results):
        """
        Evaluation finished for a study
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        super(self.__class__, self).study_evaluated(study_deploy_results)

    def reports_generated(self, study_deploy_results):
        """
        The reports have been generated for a study
        Args:
            study_deploy_results (StudyDeployResults): results object
        """
        super(self.__class__, self).reports_generated(study_deploy_results)
        if self.studies_db:
            params = self._get_update_results_status_params(study_deploy_results)
            self.studies_db.update_study(study_deploy_results.studies_db_id, params)

    def _get_update_results_status_params(self, study_deploy_results):
        """
        Get the parameters required to update StudiesDB "results_status" fields
        Args:
            study_deploy_results (StudyDeployResults):

        Returns:
            dictionary of field (str)-value
        """
        results_status_code = study_deploy_results.get_error_code()
        results_error_message = None

        if study_deploy_results.has_unexpected_errors():
            results_error_message = study_deploy_results.get_error_message()
        elif not study_deploy_results.passed_checker_acceptance_criteria():
            results_error_message = "|".join(study_deploy_results.get_failed_acceptance_criteria())

        params = {
            'results_status_code': results_status_code,
            'results_status_text': self.studies_db.CODE_MAPPINGS[('results_status_code', results_status_code)],
            'results_error_message': results_error_message,
        }
        return params

    def study_output_sent(self, study_deploy_results):
        """
        The outputs for the study have been sent
        Args:
            study_deploy_results (StudyDeployResults):
        """
        super(self.__class__, self).study_output_sent(study_deploy_results)

    def study_finished(self, study_deploy_results):
        """
        A study has finished. All the required info should be available in the study_deploy_results
        Args:
            study_deploy_results (StudyDeployResults): StudyDeployResults object with all the information for the study
        """
        super(self.__class__, self).study_finished(study_deploy_results)

        if self.studies_db:
            # Update again results_status in the event because it could happen that is has not been saved yet
            results_status_params = self._get_update_results_status_params(study_deploy_results)

            processing_status_code = 2  # Study finished

            # if len([o for o in self.output_objects if o.is_clinical]) == 0:
            #     # There was no any clinical output channel
            #     sending_status_code = -1
            # Check if there has been any error messages in the sending output
            sending_error_message = "|".join(output_results['ERROR'] for output_results
                                             in study_deploy_results.output_send_results.values()
                                             if 'ERROR' in output_results)
            sending_status_code = 0 if sending_error_message == "" else 1
            params = {
                'processing_status_code': processing_status_code,
                'processing_status_text': self.studies_db.CODE_MAPPINGS[('processing_status_code', processing_status_code)],
                'sending_status_code': sending_status_code,
                'sending_status_text': self.studies_db.CODE_MAPPINGS[('sending_status_code', sending_status_code)],
                'sending_error_message': sending_error_message,
                'timedate_processed': datetime.datetime.utcnow()
            }
            params.update(results_status_params)
            self.studies_db.update_study(study_deploy_results.studies_db_id, params)

    #endregion

    #region Aux methods

    def update_config_with_external_params(self, config, client_config_file):
        """
        Update a Config object using the parameters from a external file.
        Args:
            config (Config): original Config object
            client_config_file (str): path to the client config file

        Returns:
            (None). The Config object will be updated in place
        """
        assert os.path.isfile(client_config_file), f"Client Config file not found in {client_config_file}"
        with open(client_config_file, 'r') as f:
            client_params = json.load(f)

        if 'PACS_RECEIVE_PORT' in client_params and 'PACS_RECEIVE_AE' in client_params:
            config.set_pacs_receive_config(client_params['PACS_RECEIVE_PORT'], client_params['PACS_RECEIVE_AE'])
        else:
            warnings.warn('PACS RECEIVE params incomplete!')

        if 'PACS_SEND_IP' in client_params and 'PACS_SEND_PORT' in client_params and 'PACS_SEND_AE' in client_params:
            config.set_pacs_send_config(client_params['PACS_SEND_IP'], client_params['PACS_SEND_PORT'],
                                        client_params['PACS_SEND_AE'])
        else:
            warnings.warn('PACS SEND params incomplete!')

        if 'RIS_SEND_IP' in client_params and 'RIS_SEND_PORT' in client_params:
            config.set_ris_send_config(client_params['RIS_SEND_IP'], client_params['RIS_SEND_PORT'])
        else:
            warnings.warn('RIS SEND params incomplete!')

        if 'FACILITY_NAME' in client_params:
            config.set_param('facility_name', client_params['FACILITY_NAME'])

        # for param, value in client_params.items():
        #     # RIS and PACS params will be set in a "group"
        #     if not param.startswith("RIS_") and not param.startswith("PACS_"):
        #         config.set_param(param.lower(), value)
        # required_params = {#'CLIENT_DIR', 'SUPPORT_DIR',
        #                    #'INPUT_DIR', 'OUTPUT_DIR',
        #                    'PACS_RECEIVE_PORT', 'PACS_RECEIVE_AE',
        #                    'PACS_SEND_IP', 'PACS_SEND_PORT', 'PACS_SEND_AE',
        #                    'RIS_SEND_IP', 'RIS_SEND_PORT'}
        # for param, value in client_params.items():
        #     config.set_param(param.lower(), value)
        #     if param in required_params:
        #         required_params.remove(param)
        #
        # assert len(required_params) == 0, "The following required params were not found: {}".format(required_params)

    def _filter_reports_to_send(self, study_deploy_results):
        """
        Get the reports that should be sent based on the current config and the current study results
        Args:
            study_deploy_results (StudyDeployResults): current results object

        Returns:
            dictionary
        """
        filtered_reports_keys = set(self.reports.keys())

        # Shared actions in every run_mode
        if not study_deploy_results.passed_checker_acceptance_criteria():
            # Remove all the reports
            filtered_reports_keys.clear()
        elif Report.MSP in filtered_reports_keys \
             and DicomTypeMap.get_study_type(study_deploy_results.metadata) != DicomTypeMap.DBT:
                # This report should be generated only for DBT studies
                filtered_reports_keys.remove(Report.MSP)

        # CADt actions
        if self.config[Config.MODULE_DEPLOY, 'run_mode'] == const_deploy.RUN_MODE_CADT:
            if Report.CADT_HL7 in self.reports:
                # HL7 report should be always generated even if the study didn't pass the acceptance criteria
                filtered_reports_keys.add(Report.CADT_HL7)

            if Report.CADT_PREVIEW in filtered_reports_keys \
                and study_deploy_results.get_category() == CategoryHelper.NOT_SUSPICIOUS:
                    # This report should be generated only for Suspicious cases
                    filtered_reports_keys.remove(Report.CADT_PREVIEW)

        # Return a dictionary of reports_keys/report_objects with the filtered keys only
        return {key: self.reports[key] for key in filtered_reports_keys}

    def _filter_outputs_to_send(self, study_deploy_results):
        """
        Filter the reports that should be sent based on the current config and the current study results
        Args:
            study_deploy_results (StudyDeployResults): current results object

        Returns:
            dictionary
        """
        # Default: just use all the configured reports
        filtered_keys = list(self.output_objects.keys())

        if 'INTERMEDIATE_OUTPUTS' in filtered_keys and not study_deploy_results.predicted:
            # Intermediate results will not be saved if no prediction was made
            filtered_keys.remove('INTERMEDIATE_OUTPUTS')

        if self.config[Config.MODULE_DEPLOY, 'run_mode'] == const_deploy.RUN_MODE_CADT \
                and 'PACS' in filtered_keys \
                and not study_deploy_results.predicted:
            filtered_keys.remove('PACS')

        # Return just filtered keys
        return OrderedDict({key: self.output_objects[key] for key in filtered_keys})

    #endregion

if __name__ == '__main__':
    deployer = Deployer()
    args = deployer.parse_args()

    if 'product_label' in args:
        # Just show a Product label
        print(Config.get_product_label(args.product_label))
    else:
        # Regular deploy
        args_dict = vars(args)
        config = deployer.create_config_object(args_dict)
        deployer.initialize(config)

        if args.save_run:
            run = Run(deploy_config=config)
            run.save_run()

        deployer.deploy()
