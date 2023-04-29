import os

from centaur_reports.report import Report
from centaur_reports.helpers.srwriter import SRWriter
from centaur_reports.constants import RUN_MODE_CADT, RUN_MODE_CADX, RUN_MODE_PRODUCT_MAP


class StructuredReport(Report):
    def __init__(self, algorithm_version, run_mode=RUN_MODE_CADX, logger=None):
        super(StructuredReport, self).__init__(algorithm_version, logger=logger)
        self._report_prefix = 'SR'
        self._file_extension = 'dcm'
        self.run_mode = run_mode
        assert self.run_mode in [RUN_MODE_CADT, RUN_MODE_CADX], 'invalid run_mode {}'.format(run_mode)
        self._product = RUN_MODE_PRODUCT_MAP[run_mode]

        self.srwriter = None

    def generate(self, study_deploy_results, intended_workstation='eRad', **kwargs):
        """
        Generate a DICOM Structured Report
        Args:
            study_deploy_results (StudyDeployResults): Object that contains the results of the study
            intended_workstation (str): indicate on which work station will the CAD SR be displayed.

        Returns:
            dictionary of str ('output_file_path'): Path to the generated report
        """
        self.srwriter = SRWriter(study_deploy_results, intended_workstation, self._algorithm_version, self.run_mode)
        self.srwriter.populate_sr()

        file_name = self.get_output_file_name(study_deploy_results.get_studyUID())
        output_file_path = os.path.join(study_deploy_results.output_dir, file_name)
        self.srwriter.save(output_file_path)

        return {'output_file_path': output_file_path}

