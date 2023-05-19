import os

from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_io.output.output import Output
from deephealth_utils.data.parse_logs import log_line


class JsonOutput(Output):
    """
    Default Output JSON file object.
    It saves a StudyDeployResults object in the default path directory
    """
    def __init__(self, logger=None):
        """
        Constructor
        Args:
            logger (Logger): logging object (optional)
        """
        super().__init__(logger=logger)

    def send_results(self, results):
        """
        Write the results to a json file
        Args:
            results (StudyDeployResults): results object
        Return:
            dictionary:
                {
                    'CODE': int (0 = OK)
                    'ERROR': str (optional)
                }
        """
        try:
            assert isinstance(results, StudyDeployResults), "StudyDeployResults object expected"
            os.makedirs(results.output_dir, exist_ok=True)
            results.save()
            self.logger.info(log_line(-1, "Study results file saved ({})".format(results.get_default_results_file_path()),
                                      results.get_study_dir_name()))
            return self.RESULT_OK
        except:
            return self.result_error()


