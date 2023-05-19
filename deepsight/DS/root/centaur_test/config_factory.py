import os
import tempfile

import centaur_deploy.constants as const_deploy
from deephealth_utils.data.checker import Checker

class ConfigFactory:
    """
    Class that builds different dictionaries of parameters used to create a Config object.
    """
    def __init__(self, params=None):
        """
        The constructor should be invoked internally only
        """
        self._params = {} if params is None else params

    @property
    def params_dict(self):
        return self._params

    def _set_internal_config(self, input_dir, output_dir, additional_params=None):
        """
        Config object for an internal run (basic properties)
        Args:
            input_dir (str): input dir
            output_dir (str): base output dir
            additional_params (dict): additional params

        Returns:
            dictionary
        """
        params = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'use_heartbeat': 'n',
            'pacs_send': False,
            'ris_send': False
        }
        if additional_params is not None:
            params.update(additional_params)
        self._params.update(params)

    @classmethod
    def VerificationInternalConfigFactory(cls, input_dir=const_deploy.DEFAULT_INPUT_FOLDER,
                                          output_dir=const_deploy.DEFAULT_OUTPUT_FOLDER, optional_params_dict=None):
        """
        Get a ConfigFactory instance for verification tests Unit/Integration/System
        Args:
            input_dir (str): input dir
            output_dir (str): base output dir
            optional_params_dict (dict): optional params

        Returns:
            dictionary
        """
        factory = ConfigFactory()
        factory._set_internal_config(input_dir, output_dir, additional_params=optional_params_dict)
        config_dict = {}
        temp_preprocessed_numpy_folder = os.path.join(output_dir, "preprocessed_numpy")
        config_dict["preprocessed_numpy_folder"] = temp_preprocessed_numpy_folder
        config_dict['checker_mode'] = Checker.CHECKER_PRODUCTION
        config_dict['error_tracking'] = False
        factory._params.update(config_dict)
        return factory

    @classmethod
    def VerificationPACSConfigFactory(cls, optional_params_dict=None):
        temp_folder = tempfile.mkdtemp()
        print(f"{temp_folder} created")
        input_folder = os.path.join(temp_folder, "input")
        os.makedirs(input_folder)
        output_folder = os.path.join(temp_folder, "output")
        os.makedirs(output_folder)

        params = {
            'input_dir': input_folder,
            'output_dir': output_folder,
            #'run_mode': run_mode,
            'pacs_receive': True,
            'pacs_receive_port': 29999,
            'pacs_receive_ae': "test_ae",
        }
        if optional_params_dict is not None:
            params.update(optional_params_dict)
        factory = ConfigFactory(params=params)
        return factory


    @classmethod
    def ResearchConfigFactory(cls, input_dir, output_dir, run_mode, **kwargs):
        factory = ConfigFactory()
        factory._set_internal_config(input_dir, output_dir,
                                     additional_params={'checker_mode': Checker.CHECKER_RESEARCH, 'run_mode': run_mode})

        factory._params.update(kwargs)

        return factory


    def set_production_without_pixel_info(self):
        params = {
            'checker_mode': Checker.CHECKER_PRODUCTION,
            'process_pixel_data': False,
            'skip_study_checks': True
        }
        self._params.update(params)

