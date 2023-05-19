import logging

from deephealth_utils import get_memory_used
from deephealth_utils.data.parse_logs import log_line


class BaseModel:
    """BaseModel class: Abstract class for Models.
    Methods: __call__(inputs): Take inputs and returns outputs.
    """

    def __init__(self, config, logger=None):
        self._config = config
        # self.model_config = _config['model']
        self._logger = logger
        self._pixel_data = None

    @property
    def logger(self):
        if self._logger is None:
            # Initialize default system logger
            self._logger = logging.getLogger()
        return self._logger

    def set_logger(self, logger):
        self._logger = logger

    def __call__(self, inputs):
        raise NotImplementedError('The __call__() method is not implemented in the BaseModel child object.')

    @property
    def config(self):
        return self.get_config()

    def get_config(self):
        return self._config

    def get_version(self):
        """
        Model version or None if not found in config file
        :return: str
        """
        if 'version' not in self._config:
            return None
        return self._config['version']

    def set_pixel_data(self, pixel_data):
        if hasattr(self, 'models'):
            for key, model in self._models.items():
                model.set_pixel_data(pixel_data)
        else:
            self._pixel_data = pixel_data

    def log_time(self, message, duration, additional_info=""):
        self.logger.info(log_line(4, "{}:{}s;MEM:{}".format(message, duration, get_memory_used()), str(additional_info)))
