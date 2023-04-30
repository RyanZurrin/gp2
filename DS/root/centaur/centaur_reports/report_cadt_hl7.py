from centaur_reports.hl7.base_hl7_report import HL7Report
from centaur_reports.hl7.hl7_controller import CADtHL7
import centaur_reports.constants as const

class CADtHL7Report(HL7Report):
    """A HL7 Report class for Saige-Q service. It generate reports in HL7 report using a class of BaseCentaurHL7.
    The `__init__` is called at the initialization of the application.
    The `generate` is called when generating report after study processed.
    """
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self._product = const.SAIGE_Q
        self._report_prefix = 'hl7'
        self._file_extension = 'txt'

        self._hl7 = CADtHL7()

    def generate(self, results, **kwargs):
        self.hl7.fill_all_fields(self._config, results)

        # Output result files
        file_path = self.save_to_file(results.output_dir, self.hl7)

        return {'output': file_path}
