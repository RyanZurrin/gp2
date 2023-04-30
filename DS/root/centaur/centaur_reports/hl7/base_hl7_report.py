import os


from centaur_reports.report import Report


class HL7Report(Report):
    PREFIX = 'DH_hl7_'
    SUFFIX = '.txt'
    REPORT_TYPE = 'hl7'

    def __init__(self, config, logger=None):
        """
        Constructor
        Args:
            config (centaur_deploy.deploys.config.Config): Config object
            logger (Logger): global logger object
        """
        super().__init__(config.get_algorithm_version(), logger)

        # Overwrite class name
        self._file_name_prefix = self.PREFIX
        self._file_name_suffix = self.SUFFIX
        self.report_type = self.REPORT_TYPE

        # Get Centaur configuration
        self._config = config

        self._hl7 = None

    @property
    def hl7(self):
        return self._hl7  # to prevent set hl7 obj

    def save_to_file(self, output_dir, hl7_obj):
        """
        Save an hl7 object in a local file in the output folder
        Args:
            output_dir (str): output dir
            hl7_obj (HL7): HL7 object

        Returns:
            str. Path to the saved file
        """
        study_instance_uid = hl7_obj.study_instance_uid
        file_fullpath = os.path.join(output_dir, self.get_output_file_name(study_instance_uid))
        with open(file_fullpath, 'w', newline="\r") as f:
            f.write(hl7_obj.message)
        return file_fullpath
