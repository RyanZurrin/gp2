import os
import traceback

import hl7
from hl7.client import MLLPClient

from centaur_io.output.output import Output


class RISOutput(Output):
    def __init__(self, ip, port, logger=None):
        super(self.__class__, self).__init__(logger=logger)
        # self._is_clinical = True
        self.ip = ip
        self.port = port

    def send_results(self, results):
        """
        Send HL7 message to a RIS
        Args:
            results (str): Path to a file that contains an HL7 message properly formatted

        Returns:
            dictionary
            {
                'CODE': int (1=file not found; 2=HL7 message could not be parsed; 3=other)
                'ERROR': str (optional)
            }
        """
        if not os.path.isfile(results):
            return self.result_error(code=1, error_message=f"File not found ({results})")

        try:
            with open(results, 'r') as f:
                hl7_msg = hl7.parse(f.read())
        except:
            error_msg = traceback.format_exc()
            return self.result_error(code=2, error_message=f"File {results} could not be converted to a HL7 message.\n"
                                                           f"{error_msg}")
        try:
            with MLLPClient(self.ip, self.port) as client:
                client.send_message(hl7_msg)
                return self.RESULT_OK
        except:
            error_msg = traceback.format_exc()
            return self.result_error(code=3,
                                     error_message=f"Message could not be sent to {self.ip}/{self.port}: {error_msg}")