########################################################
# DEPRECATED CLASS !!
########################################################
import warnings
warnings.warn("This is a deprecated class", DeprecationWarning)

import logging
import os
import time
from threading import Thread
import datetime

from deephealth_utils.data.parse_logs import log_line

import centaur_io.constants as constants



class InputDirMonitorThread(Thread):
    def __init__(self, input_dir, studies_db, new_study_threshold_seconds,
                 sleep_for_seconds=10, use_transfer_complete_file=False, logger=None):
        """
        Constructor
        Args:
            input_dir (str): input dir to monitor where the new studies will be received
            studies_db (StudiesDB): StudiesDB object created by the Deployer (also see centaur_io/db.py)
            new_study_threshold_seconds (int): number of seconds to wait for the last received file to insert a new study into the db
        """
        super().__init__()
        self.daemon = True
        self.name = "DHInputDirMonitor"
        self.input_dir = input_dir
        self.studies_db = studies_db
        self.new_study_threshold_seconds = new_study_threshold_seconds
        self.sleep_for_seconds = sleep_for_seconds
        self.daemon = True  # Make sure thread is stopped when parent process is stopped
        self.__stop = False
        self.use_transfer_complete_file = use_transfer_complete_file
        self._logger = logger

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger()
        return self._logger

    def run(self):
        """
        Listen for existing folders in self.input_dir. For each folder, try to insert it in StudiesDB if it has not
        been inserted yet
        """
        self.logger.info(log_line(-1, f"Listening for new studies in {self.input_dir}..."))
        try:
            while True:
                # Name of subdir in input_dir is Study Instance UID
                folder_names = self.get_folder_names(self.input_dir)
                if len(folder_names) > 0:
                    self.logger.info(log_line(-1, f"Found {len(folder_names)} new studies to process"))
                    for folder_name in folder_names:
                        study_dir = os.path.join(self.input_dir, folder_name)
                        self.logger.info(log_line(-1, f"Processing {study_dir}..."))
                        if self.use_transfer_complete_file:
                            # Search for transfer_complete_file to assume the study transfer is complete
                            if os.path.isfile(os.path.join(study_dir, constants.TRANSFER_COMPLETE_FILE_NAME)):
                                self.logger.info(log_line(-1, f"{constants.TRANSFER_COMPLETE_FILE_NAME} found."
                                                              f" Attempting to insert {study_dir} in DB..."))
                                inserted = self.studies_db.insert_if_not_exist(study_dir)
                                self.logger.info(log_line(-1, f"{study_dir} inserted in DB: {inserted}"))
                            else:
                                self.logger.info(log_line(-1,
                                                 f"{constants.TRANSFER_COMPLETE_FILE_NAME} not found yet in {study_dir}"))
                        else:
                            # Search for the most recent file and assume the study has been completely transferred if the
                            # newest file has been modified less than new_study_threshold_seconds seconds ago
                            latest_timestamp = self.get_latest_timestamp_in_dir(study_dir)
                            if latest_timestamp is None:
                                self.logger.warning(log_line(5, f"Files not found in folder {folder_name}"))
                                continue
                            self.logger.info(log_line(-1, f"Latest Timestamp found: {latest_timestamp}"))
                            current_timestamp = self.get_current_utctime()

                            diff_in_seconds = (current_timestamp - latest_timestamp).seconds
                            if diff_in_seconds > self.new_study_threshold_seconds:
                                self.logger.info(log_line(-1, f"Attempting to insert {study_dir} in DB..."))
                                # Insert the study in the StudiesDB (if we didn't do it already)
                                inserted = self.studies_db.insert_if_not_exist(study_dir)
                                self.logger.info(log_line(-1, f"{study_dir} inserted in db: {inserted}"))
                            else:
                                self.logger.info(log_line(-1, f"Study is too recent ({diff_in_seconds} seconds old)"))
                else:
                    # No new folders in input_dir, just sleep for some seconds
                    self.logger.info(log_line(-1, f"No new studies found"))
                self.logger.info(log_line(-1, f"Sleeping for {self.sleep_for_seconds} seconds..."))
                time.sleep(self.sleep_for_seconds)

                if self.__stop:
                    self.logger.warning(log_line(5, "Input monitor stopped"))
                    return
        except:
            self.logger.exception("Unexpected error in the input monitor thread", exc_info=True)
            raise

    def stop(self):
        """ This method does not kill this thread (is_alive() = True)
            But, it will stop "run"
        """
        self.__stop = True

    @staticmethod
    def get_current_utctime():
        """ Get current time in UTC
        """
        return datetime.datetime.utcnow()

    @classmethod
    def get_latest_timestamp_in_dir(cls, dir):
        """ Get latest timestamp in dir
        :param dir:
        :return: latest timestamp in dir
        """
        timestamps = []
        for file in os.listdir(dir):
            filepath = "{}/{}".format(dir, file)
            # Get timestamp (in UTC) of file
            timestamp = os.path.getmtime(filepath)  # modification time
            timestamp = datetime.datetime.utcfromtimestamp(timestamp)
            timestamps.append(timestamp)
        if len(timestamps) > 0:
            latest_timestamp = max(timestamps)
            return latest_timestamp
        return None

    @classmethod
    def get_folder_names(cls, dir):
        """
        Get a list of subdirectory names in 'dir'

        :param dir:
        :return: a list of subdirectory names
        """
        folder_names = []
        for folder_name in os.listdir(dir):
            if folder_name.startswith("."):
                continue
            folder_path = os.path.join(dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            folder_names.append(folder_name)
        return folder_names