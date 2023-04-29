import logging
import traceback

from deephealth_utils.data.parse_logs import log_line


class Output(object):
    RESULT_OK = {'CODE': 0}

    def __init__(self, logger=None):
        self._logger = logger
        self._is_clinical = False

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger()
        return self._logger

    # @property
    # def is_clinical(self):
    #     """
    #     The output is a clinical channel (PACS, RIS, etc.)
    #     Returns:
    #         bool
    #     """
    #     return self._is_clinical

    def set_logger(self, logger):
        self._logger = logger

    def send_results(self, results):
        """
        Send the results of the study
        Args:
            results (object): results in the expected format

        Returns:
            dictionary:
                {
                    'CODE': int (0 = OK)
                    'ERROR': str (in case of success this key will not be present)
                }
        """
        raise NotImplementedError("This must be implemented in a child class")

    def result_error(self, code=1, error_message=None):
        """
        Default error message
        Args:
            code (int): error code to return (default: 1)
            error_message (str): error message (default: last exception traceback)

        Returns:
            dictionary
            {
                'CODE': int
                'ERROR': str (optional)
            }
        """
        if error_message is None:
            error_message = traceback.format_exc()
        self.logger.warning(log_line(5, "Error when sending output:\n" + error_message))
        return {'CODE': code, 'ERROR': error_message}

