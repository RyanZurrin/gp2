import json
import logging
import os
import pandas as pd

import centaur_deploy.constants as const_deploy
from deephealth_utils.data.checker import Checker


class Config(object):
    """
    Centralize all the config parameters used for any of the Centaur components
    """
    # MODULES
    MODULE_DEPLOY = "DEPLOY"
    MODULE_ENGINE = "ENGINE"
    MODULE_IO = "IO"
    MODULE_REPORTS = "REPORTS"
    MODULES = (MODULE_DEPLOY, MODULE_ENGINE, MODULE_IO, MODULE_REPORTS)
    CADT_OPERATING_POINT_KEYS = ('high_spec', 'balanced', 'high_sens')

    def __init__(self, config_file_path=None, overriden_params={}):
        """
        Constructor
        Args:
            config_file_path (str): if not None, path to a Config file to initialize params
            overriden_params (dict): if existing, override default config params (and potentially config file params)
        """
        self._params = {}
        for module in Config.MODULES:
            self._params[module] = {}
        # self[Config.MODULE_DEPLOY, 'run_mode'] = mode
        self._initialize()

        if config_file_path is not None:
            assert os.path.isfile(config_file_path), f"Config file not found in {config_file_path}"
            with open(config_file_path, 'r') as f:
                params = json.load(f)
            params.update(overriden_params)
        else:
            params = overriden_params

        if len(params) > 0:
            self.override_default_values(params)
            self._paths_to_absolute()
        self._sync_params()

    # region Public Class methods

    @classmethod
    def from_json_file(cls, json_file_path):
        """
        Generate a Config object from a json file
        Args:
            json_file_path (str): Path to the json file
        Returns:
            Config object
        """
        assert os.path.isfile(json_file_path), "File {} not found".format(json_file_path)
        with open(json_file_path, 'r') as fp:
            return cls.from_json(fp.read())

    @classmethod
    def from_json(cls, json_str):
        """
        Create a Config object from a json string
        :param json_str: str. Json string
        :return: Config object
        """
        params = json.loads(json_str)
        for module, module_params in params.items():
            for mod_param, mod_value in module_params.items():
                if mod_param.endswith("__DF"):
                    # Parameter was a dataframe
                    df = pd.read_json(mod_value)
                    # Get the "real name" of the parameter and save the dataframe instead of the string json representation
                    params[module][mod_param[:-4]] = df
                    # Remove "prefixed" parameter
                    del params[module][mod_param]
        c = Config()
        c._params = params
        return c

    @classmethod
    def get_algorithm_version(cls):
        """
        Get the current algorithm version.

        Returns:
            str
        """
        assert os.path.isfile(const_deploy.ALGORITHM_VERSION_PATH), \
            f"Algorithm version file not found in {const_deploy.ALGORITHM_VERSION_PATH}"
        with open(const_deploy.ALGORITHM_VERSION_PATH, 'r') as f:
            return f.read()

    @classmethod
    def get_product_label(cls, product):
        """
        Get a label with the description of the product
        Args:
            product (str): Saige-Q, Saige-Dx, etc.

        Returns:
            str
        """
        if product == "Saige-Q":
            label = const_deploy.LABEL_SAIGE_Q
        elif product == "Saige-Dx":
            label = const_deploy.LABEL_SAIGE_DX
        else:
            raise ValueError(f"Product not valid ({product})")

        algorithm_version = cls.get_algorithm_version()
        final_label = ""
        for line in label.split('\n'):
            # Replace [VERSION_PLACEHOLDER] with the real version and add the missing spaces to complete the line
            if "[VERSION_PLACEHOLDER]" in line:
                ix = line.index("[VERSION_PLACEHOLDER]")
                num_spaces_before = len(line) - ix - len("[VERSION_PLACEHOLDER]") - 2
                num_spaces_now = num_spaces_before + (len("[VERSION_PLACEHOLDER]") - len(algorithm_version))
                line_to_add = line[:ix] + algorithm_version + (" " * num_spaces_now) + "||\n"
                final_label += line_to_add
            else:
                final_label += line + "\n"
        return final_label

    # def copy(self):
    #     """
    #     Create a deep copy of the object
    #     """
    #     c = Config(self.__create_key)
    #     for mod in self.MODULES:
    #         c[mod] = copy.deepcopy(self[mod])
    #     return c

    # endregion

    # region Public methods

    def set_input_dir(self, value):
        self[Config.MODULE_IO, 'input_dir'] = os.path.realpath(value)

    def set_output_dir(self, value):
        self[Config.MODULE_IO, 'output_dir'] = os.path.realpath(value)

    def set_pacs_receive_config(self, port, ae):
        """
        Set the configuration for the PACS where the input studies are received
        Args:
            port (int): port number
            ae (str): DICOM Application Entity (AE)
        """
        self[Config.MODULE_IO, 'pacs_receive'] = True
        self[Config.MODULE_IO, 'pacs_receive_port'] = port
        self[Config.MODULE_IO, 'pacs_receive_ae'] = ae

    def set_pacs_send_config(self, ip, port, ae):
        """
        Set the configuration for the PACS where the output images are going to be sent
        Args:
            ip (str): IP address
            port (int): port number
            ae (str): DICOM Application Entity (AE)
        """
        self[Config.MODULE_IO, 'pacs_send'] = True
        self[Config.MODULE_IO, 'pacs_send_ip'] = ip
        self[Config.MODULE_IO, 'pacs_send_port'] = port
        self[Config.MODULE_IO, 'pacs_send_ae'] = ae

    def get_run_mode(self):
        """
        Get the current run mode
        Returns:
            str (normally one of the values in centaur_deploy.constants.RUN_MODE_X or None)
        """
        return self[Config.MODULE_DEPLOY, 'run_mode']

    def set_ris_send_config(self, ip, port):
        """
        Set the configuration for an external RIS where results will be sent
        Args:
            ip (str): IP address
            port (int): port number
        """
        self[Config.MODULE_IO, 'ris_send'] = True
        self[Config.MODULE_IO, 'ris_send_ip'] = ip
        self[Config.MODULE_IO, 'ris_send_port'] = port

    def set_param(self, param_name, value):
        """
        Set a value using just a key name
        Args:
            param_name (str): param name
            value (object): param value
        """
        modules = self.search_key_modules(param_name)
        assert len(modules) > 0, f"Parameter {param_name} was not found in any Config section"
        self.override_default_values({param_name: value})

    def override_default_values(self, params_dict):
        """
        Override default values using a dictionary.
        This method is useful to create a Config object using CLI parameters
        :param params_dict: dictionary of parameters
        """
        for key, value in params_dict.items():
            if value is not None:
                modules = self.search_key_modules(key)
                for module in modules:
                    self[module, key] = value
    def to_json(self):
        """
        Get a json string that represents the config
        :return: str. Json string
        """
        string_params = {}
        for module, params in self._params.items():
            string_params[module] = {}
            for mod_param, mod_value in params.items():
                if isinstance(mod_value, pd.DataFrame):
                    # Convert Dataframes to a json format
                    string_params[module][mod_param + "__DF"] = mod_value.to_json()
                else:
                    # Otherwise, just use the string representation
                    string_params[module][mod_param] = mod_value
        return json.dumps(string_params)

    def to_json_file(self, output_file_path):
        """
        Write this object to a json file
        :param output_file_path: str. Output path (it will overwrite any existing files!)
        """
        js = self.to_json()
        with open(output_file_path, 'w') as fp:
            fp.write(js)


    def search_key_modules(self, key):
        """
        Search in all the modules to the passed key object.
        It returns a list of the modules where key is present
        :param key: str. Key to search
        :return: list of str (modules)
        """
        modules_found = []
        for module in self.MODULES:
            if key in self[module]:
                modules_found.append(module)
        return modules_found

    def disable_image_flags(self):
        """
        Disable all the Checker flags that are directly related to the pixel data information
        """
        checks_to_ignore = Checker.checks_that_need_pixels()
        if "checks_to_ignore" in self[Config.MODULE_ENGINE]:
            self[Config.MODULE_ENGINE, "checks_to_ignore"].extend(checks_to_ignore)
        else:
            self[Config.MODULE_ENGINE, "checks_to_ignore"] = checks_to_ignore

    def set_cadt_operating_points(self, dxm, dbt, explicitly_set=False):
        """
        Set values for the operating points (thresholds) both in DXM and DBT modalities.
        If the values were not read using an operating point key (high_sens, balanced, high_spec) and a thresholds file
        (regular scenario), cadt_operating_point_key will be set to None to avoid confusion
        Args:
            dxm (float): operating point in DXM modality
            dbt (float): operating point in DBT modality
            explicitly_set (bool): the values were set explicitly, meaning they were not read using an
                                    Operating Point key (high_sens, balanced, high_spec) + thresholds file
        """
        assert isinstance(dxm, float) and isinstance(dbt, float), f"Expected 2 float values. Got: {dxm}, {dbt}"
        self[Config.MODULE_ENGINE, 'cadt_operating_point_values'] = {'dxm': dxm, 'dbt': dbt}

        # If the values were explicitly set, cadt_operating_point_key is set to None to avoid confusion
        if explicitly_set:
            print("WARNING: default operating points will be overwritten!")
            self[Config.MODULE_ENGINE, 'cadt_operating_point_key'] = None

    # endregion

    # region Internal methods

    def _initialize(self):
        """
        Default configuration
        """
        # External params
        self[Config.MODULE_DEPLOY, 'execution_id'] = None

        self[Config.MODULE_DEPLOY, 'run_mode'] = None
        self[Config.MODULE_DEPLOY, 'skip_existing'] = False
        self[Config.MODULE_DEPLOY, 'logging'] = logging.INFO
        self[Config.MODULE_DEPLOY, 'debug'] = False
        self[Config.MODULE_DEPLOY, 'preprocess_only'] = False
        self[Config.MODULE_DEPLOY, 'exit_on_error'] = False
        self[Config.MODULE_DEPLOY, 'parallelize'] = False
        self[Config.MODULE_DEPLOY, 'reports'] = []
        self[Config.MODULE_DEPLOY, 'count_studies'] = False
        self[Config.MODULE_DEPLOY, 'continue_log'] = None
        self[Config.MODULE_DEPLOY, 'keep_temp_numpy'] = False
        self[Config.MODULE_DEPLOY, 'use_heartbeat'] = False
        self[Config.MODULE_DEPLOY, 'heartbeat_log'] = "heartbeat.log"
        self[Config.MODULE_DEPLOY, 'heartbeat_period_in_seconds'] = 60
        self[Config.MODULE_DEPLOY, 'error_tracking'] = True
        self[Config.MODULE_DEPLOY, 'cadt_threshold_versions'] = 'current'
        self[Config.MODULE_DEPLOY, 'cadx_threshold_versions'] = 'current'

        self[Config.MODULE_IO, 'input_dir'] = const_deploy.DEFAULT_INPUT_FOLDER
        self[Config.MODULE_IO, 'output_dir'] = const_deploy.DEFAULT_OUTPUT_FOLDER
        self[Config.MODULE_IO, 'pacs_receive'] = False
        self[Config.MODULE_IO, 'pacs_receive_port'] = 0
        self[Config.MODULE_IO, 'pacs_receive_ae'] = 'DH_Receive'
        self[Config.MODULE_IO, 'pacs_send'] = False
        self[Config.MODULE_IO, 'pacs_send_ip'] = "127.0.0.1"
        self[Config.MODULE_IO, 'pacs_send_port'] = 0
        self[Config.MODULE_IO, 'pacs_send_ae'] = 'DH_Send'
        self[Config.MODULE_IO, 'ris_send'] = False
        self[Config.MODULE_IO, 'ris_send_ip'] = "127.0.0.1"
        self[Config.MODULE_IO, 'ris_send_port'] = 0
        self[Config.MODULE_IO, 'new_study_threshold_seconds'] = 180
        self[Config.MODULE_IO, 'remove_input_files'] = False
        self[Config.MODULE_IO, 'monitor_input_dir_continuously'] = False

        self[Config.MODULE_ENGINE, 'model_version'] = "current"
        self[Config.MODULE_ENGINE, 'checker_mode'] = Checker.CHECKER_PRODUCTION
        self[Config.MODULE_ENGINE, 'skip_study_checks'] = False
        self[Config.MODULE_ENGINE, 'process_pixel_data'] = True
        self[Config.MODULE_ENGINE, 'checks_to_ignore'] = []
        self[Config.MODULE_ENGINE, 'checks_to_include'] = []
        self[Config.MODULE_ENGINE, 'save_to_ram'] = True
        # self[Config.MODULE_ENGINE, 'reuse_ds'] = True
        self[Config.MODULE_ENGINE, 'preprocessed_numpy_folder'] = None
        # Results post processing
        self[Config.MODULE_ENGINE, 'results_pproc_max_bbxs_displayed_total'] = 4
        self[Config.MODULE_ENGINE, 'results_pproc_max_bbxs_displayed_intermediate'] = 2
        # Approx. percentile 1/99 in op2
        self[Config.MODULE_ENGINE, 'results_pproc_min_relative_bbx_size_allowed'] = 0.02
        self[Config.MODULE_ENGINE, 'results_pproc_max_relative_bbx_size_allowed'] = 0.35
        self[Config.MODULE_ENGINE, 'save_synthetics'] = False
        self[Config.MODULE_ENGINE, 'cadt_operating_point_key'] = 'balanced'
        self[Config.MODULE_ENGINE, 'cadt_operating_point_values'] = None        # Dictionary of float ('Dxm', 'Dbt')

        self[Config.MODULE_REPORTS, 'keep_temp_pdf_files'] = False

        self[Config.MODULE_REPORTS, 'temp_html_files_dir'] = "temp_html_files"
        self[Config.MODULE_REPORTS, 'temp_imgs_filename_template'] = "{modality}-{laterality}-{view}.png"
        self[Config.MODULE_REPORTS, 'facility_name'] = "Unknown"
        self[Config.MODULE_REPORTS, 'intended_workstation'] = "eRad"

    def _paths_to_absolute(self):
        """
        Convert to absolute paths the possibly relative paths (input/output) in the config
        """
        if 'input_dir' in self[Config.MODULE_IO] and self[Config.MODULE_IO, 'input_dir'] is not None:
            self[Config.MODULE_IO, 'input_dir'] = os.path.realpath(self[Config.MODULE_IO, 'input_dir'])
        if 'output_dir' in self[Config.MODULE_IO] and self[Config.MODULE_IO, 'output_dir'] is not None:
            self[Config.MODULE_IO, 'output_dir'] = os.path.realpath(self[Config.MODULE_IO, 'output_dir'])

        self._sync_params()

    def _sync_params(self):
        """
        Copy some of the settings to child sections so that they can be used independently
        """
        # TODO: include a sync mechanism for settings that can be in more than one module? Or just make them "global"?)
        self[Config.MODULE_ENGINE, 'parallelize'] = self[Config.MODULE_DEPLOY, 'parallelize']
        self[Config.MODULE_ENGINE, 'run_mode'] = self[Config.MODULE_DEPLOY, 'run_mode']

    def __getitem__(self, item):
        """
        Return a parameter value (or None if it's not found).
        Two options:
            - config[MODULE_X, param]: return single parameter or None if not found
            - config[MODULE_X]: return a dictionary with all the parameters for a module
        :param item: str or 2-tuple
        :return: parameter value or dictionary with all the parameters in a module
        """

        if not isinstance(item, tuple):
            # Single key. Return all the params in a module
            assert item in Config.MODULES, \
                "Expected format: \"config[Config.MODULE_X, 'param'] = value\" or just \"config[Config.MODULE_X]\"" \
                ", where 'MODULE_X' in {}".format(Config.MODULES)
            return self._params[item]

        # Return a single param or raise error if it's not found
        assert len(item) == 2 and item[0] in Config.MODULES and isinstance(item[1], str), \
            "Expected format: \"config[Config.MODULE_X, 'param'] = value\" or just \"config[Config.MODULE_X]\"" \
            ", where 'MODULE_X' in {}".format(Config.MODULES)
        module = item[0]
        param = item[1]
        if param in self._params[module]:
            return self._params[module][param]
        raise ValueError("Parameter {} not found".format(item))

    def __setitem__(self, key, value):
        """
        Set the value for a parameter or all parameters of a section at once.
        Two options:
            - config[MODULE_X, param] = value: set single parameter or None if not found
            - config[MODULE_X] = dict: Set dictionary with all the parameters for a module
        :param key: str or 2-tuple
        :param value: object
        """
        if not isinstance(key, tuple):
            # Set all the parameters for a module at once
            assert key in Config.MODULES, "Key must be one of {}".format(Config.MODULES)
            assert isinstance(value, dict), "Expected a dictionary of parameters for the module '{}'".format(key)
            self._params[key] = value
        else:
            assert len(key) == 2 and key[0] in Config.MODULES, \
                "Expected format: \"config[Config.MODULE_X, 'param'] = value\", where 'MODULE_X' in {}".format(
                    Config.MODULES)
            self._params[key[0]][key[1]] = value

    def __eq__(self, other):
        """
        Return True if 2 config instances are the same
        :param other:
        :return:
        """
        if not isinstance(other, type(self)):
            return False
        for mod in self.MODULES:
            if self[mod] != other[mod]:
                return False
        return True

    def __str__(self):
        return json.dumps(self._params)

    # endregion