import os
import subprocess
import re


class DicomSender:
    def __init__(self, port, config_file_path=None):
        self.config_file_path = os.path.realpath(os.path.dirname(__file__) + "/storescu.cfg") \
                                if config_file_path is None else config_file_path
        self.port = port

    def send_study(self, study_path):
        """
        Send a DICOM study to a local listener
        Args:
            study_path (str): study folder path
        """
        # command = f"storescu -v +sd 127.0.0.1 -xf {self.config_file_path} Centaur {self.port} {study_path}"
        command = f"dcmsend 127.0.0.1 {self.port} +sd {study_path}"
        response_proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = response_proc.stdout.read().decode()
        print(f"Output command {command}:\n{output}")

        assert not re.search("^E: .*$", output, flags=re.MULTILINE), \
            "Unexpected error in storescu response:\n {}".format(output)
        assert not "Error" in output, "It seems there's been an error in the storescu command.\n" \
                                       f"Command: {command}\n" \
                                       f"Output: {output}"

