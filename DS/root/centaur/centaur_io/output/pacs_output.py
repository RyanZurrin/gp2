import os
import subprocess
import traceback

from centaur_io.output.output import Output
from deephealth_utils.data.parse_logs import log_line


class PACSOutput(Output):

    def __init__(self, ip, port, ae, timeout_seconds=120, verbose=False, logger=None):
        """
        Constructor.
        Send dicom files to a PACS server
        Args:
            ip (str): IP address for PACS
            port (int): port for PACS
            ae (str): AE for PACS
            timeout_seconds (int): timeout (in seconds)
            verbose (bool): use verbose info when sending the files
            logger (Logger): logging object
        """

        super(self.__class__, self).__init__(logger=logger)
        # self._is_clinical = True
        self.ip = ip
        self.port = port
        self.ae = ae
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose

    def send_results(self, results):
        """
        Send a DICOM file (or a list of DICOM files) to a PACS
        Args:
            results (str or list-str) : file path/s

        Returns:
            dictionary
            {
                'CODE': int
                'ERROR': str (optional)
            }
        """
        if isinstance(results, str):
            # Single file
            results = [results]

        file_path = ""
        verbose = "-v" if self.verbose else ""
        try:
            for file_path in results:
                assert os.path.isfile(file_path), f"File not found ({file_path})"

                cmd_str = f"dcmsend {verbose} -to {self.timeout_seconds} {self.ip} {self.port} +rd -aec {self.ae} \"{file_path}\""
                return_code = subprocess.call(cmd_str, shell=True)
                if return_code != 0:
                    # Error sending the file
                    return self.result_error(code=return_code,
                                             error_message=self.get_dcmtk_error_message(return_code))
        except:
            # Unexpected error
            error_msg = traceback.format_exc()
            self.logger.error(log_line(5, error_msg))
            return self.result_error(error_message=f"Unexpected error when sending file '{file_path}'")

        return self.RESULT_OK

    def get_dcmtk_error_message(self, code):
        """
        Get an output error message when the dcmsend command failed
        Args:
            code (int): error code

        Returns:
            str
        """
        if code == 21:
            # no report produced
            return f"File to send was not found by DCMTK: dcmsend code {code}"
        elif code in (22, 23):
            # report is invalid
            return f"File does not have the correct format: dcmsend code {code}"
        elif code == 43:
            # cannot write to output
            return f"File could not be written in the PACS: dcmsend code {code}"
        elif code in (61, 62, 65):
            # network error
            return f"Network communication error: dcmsend code {code}"
        return f"Unknown error: dcmsend code {code}"




