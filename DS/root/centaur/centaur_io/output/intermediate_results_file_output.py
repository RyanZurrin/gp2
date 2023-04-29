import os

from centaur_io.output.json_output import JsonOutput


class IntermediateResultsFileOutput(JsonOutput):
    """Class created for legacy intermediate files for develop purposes"""
    def __init__(self, engine, save_results_post_processed=False, save_results_raw=False, save_synthetics=False,
                 logger=None):
        super().__init__(logger)
        self.engine = engine
        self.save_results_post_processed = save_results_post_processed
        self.save_results_raw = save_results_raw
        self.save_synthetics = save_synthetics

    def send_results(self, results):
        """
        Save intermediate files based on the current configuration
        Args:
            results (StudyDeployResults): results object

        Returns:
            dictionary:
                {
                    'CODE': int (0 = OK)
                    'ERROR': str (optional)
                }
        """
        try:
            if not os.path.isdir(results.output_dir):
                os.makedirs(results.output_dir)

            self.engine.save_intermediate_files(save_results_post_processed=self.save_results_post_processed,
                                                save_results_raw=self.save_results_raw,
                                                save_synthetics=self.save_synthetics)
            return self.RESULT_OK
        except:
            return self.result_error()

