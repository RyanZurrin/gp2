import os
import subprocess

from deephealth_utils.data.parse_logs import log_line
from centaur_io.input.dir_monitor_input import DirMonitorInput

# Receive studies via a DICOM listener that copies them to an input folder
class PACSInput(DirMonitorInput):

    def __init__(self, input_dir, port, ae, studies_db, new_study_threshold_seconds,
                 logger=None, scp_config_file_path=None, sleep_for_seconds=None):
        """
        Constructor.
        Args:
            input_dir (str): input folder that will be constantly monitored
            port (int): PACS port
            ae (str): PACS AE
            studies_db (StudiesDB): StudiesDB object to track received studies
            new_study_threshold_seconds (int): number of seconds to wait for the last received file to insert a new
                                               study into StudiesDB
            logger (Logger): logger object
            scp_config_file_path (str): path to a file that contains required SCP configuration for the PACS listener
            sleep_for_seconds (int): number of seconds to sleep if no new studies are found in the input dir.
                                     By default, SLEEP_IF_NO_NEW_STUDIES_SECONDS constant will be used.
        """

        super().__init__(input_dir, studies_db, logger=logger,
                         new_study_threshold_seconds=new_study_threshold_seconds,
                         sleep_for_seconds=sleep_for_seconds)
        self.pacs_receive_port = port
        self.pacs_receive_ae = ae
        self.scp_config_file_path = \
            os.path.realpath(__file__ + "/../../pacs_helpers/storescp.cfg") if scp_config_file_path is None \
                                                                            else scp_config_file_path
        self._receiver = None

    @property
    def receiver(self):
        return self._receiver

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._receiver is not None:
            print("Terminating storescp process (pid:{})...".format(self._receiver.pid))
            self._receiver.terminate()

    def start_receiver(self):
        """
        Start DICOM receiver (DCMTK).
        Returns:
            subprocess POpen
        """

        args = ["storescp", "-su", "+B", "-uf",
                "-od",  self.input_dir,
                "--fork",   # Read single files in parallel
                # "-xcs","\"python {} -d #p\"".format(self.receiver_py_path),
                #"-xcr", "\"python {} -d #p\"".format(self.receiver_py_path),
                # "-tos", str(self.scp_timeout),
                 "-xf", self.scp_config_file_path, "Centaur",
                "--aetitle", self.pacs_receive_ae,
                str(self.pacs_receive_port)]
        self._receiver = subprocess.Popen(args, shell=False)
        self.logger.info(log_line(-1, "Dicom receiver was started in port={} pid={}".format
                (self.pacs_receive_port, self._receiver.pid)))
        return self._receiver
