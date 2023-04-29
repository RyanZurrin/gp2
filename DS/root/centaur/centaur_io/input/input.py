import logging

class Input(object):

    def __init__(self, ingress=None, logger=None):
        """
        Constructor
        Args:
            ingress (str): input source (usually a string but it could be other type)
            logger (Logger): logger object
        """
        self.ingress = ingress
        self._logger = logger

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger()
        return self._logger

    def set_logger(self, logger):
        self._logger = logger

    def get_next_study(self):
        raise NotImplementedError("Need to implement get_next_study()")

