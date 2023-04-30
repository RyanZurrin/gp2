import logging
import os


class Report(object):
    """ Report class: Creates a Report object for
    Methods:
    Notes:
    """
    # Report types constants
    PDF = 'pdf'
    SR = 'sr'
    SUMMARY = 'summary'
    IMAGES = 'images'
    MOST_MALIGNANT_IMAGE = 'most_malignant_image'
    CADT_PREVIEW = 'cadt_preview'
    CADT_HL7 = 'cadt_hl7'
    MSP = 'msp'

    def __init__(self, algorithm_version, logger=None):
        """
        Constructor
        Args:
            algorithm_version (str): algorithm version (ex: 1.1.0)
            logger (Logger): global logger object
        """
        self._product = 'UNKNOWN'
        self._report_prefix = 'UNKNOWN'
        self._file_extension = 'UNKNOWN'
        self._file_name_template = 'DH_{product}_{report_prefix}_{study_uid}.{file_extension}'
        self._algorithm_version = algorithm_version
        self._logger = logger

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger()
        return self._logger

    def generate(self, study_deploy_results, **kwargs):
        """
        Generate a report for a study.
        The logic for this method should be implemented in children classes
        Args:
            study_deploy_results (StudyDeployResults): Object that contains the results of the study
            **kwargs: additional parameters

        Returns:
            str or dictionary of str. Key/s-Path/s to the reports generated
        """
        raise NotImplementedError("This method should be implemented in a child class")

    def get_output_file_name(self, study_instance_uid):
        """
        Get the default file name for a report based on the default template
        Args:
            study_instance_uid (str): StudyInstanceUID (or any other id that may be determined)

        Returns:
            str
        """
        return self._file_name_template.format(product=self._product.replace('-', ''), report_prefix=self._report_prefix,
                                               study_uid=study_instance_uid, file_extension=self._file_extension)


    def exists(self, study_instance_uid, output_dir):
        return os.path.isfile(os.path.join(output_dir, self.get_output_file_name(study_instance_uid)))
