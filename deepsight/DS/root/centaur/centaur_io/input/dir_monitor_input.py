import datetime
import glob
import os
import time
import traceback

import centaur_io.constants as const
from centaur_io.input.input import Input

from deephealth_utils.data.parse_logs import log_line
# Monitor an input folder permanently searching for new studies

class DirMonitorInput(Input):

    def __init__(self, input_dir, studies_db, logger=None,
                 new_study_threshold_seconds=None, sleep_for_seconds=None):
        """
        Constructor
        Args:
            input_dir (str): input folder that will be monitored constantly
            studies_db (StudiesDB): StudiesDB object to track received studies
            logger (Logger): logger object
            sleep_for_seconds (int): number of seconds to sleep if no new studies are found in the input dir.
                                     By default, SLEEP_IF_NO_NEW_STUDIES_SECONDS constant will be used.
            new_study_threshold_seconds (int): number of seconds to wait for the last received file to insert a new
                                               study into StudiesDB.
                                               When None, the study will be inserted in StudiesDB
                                               only when a ".transfer_complete" file is found
        """
        super().__init__(ingress=input_dir, logger=logger)

        self._studies_db = studies_db
        self._sleep_for_seconds = sleep_for_seconds if sleep_for_seconds is not None \
                                 else const.SLEEP_IF_NO_NEW_STUDIES_SECONDS
        self.new_study_threshold_seconds = new_study_threshold_seconds

    @property
    def studies_db(self):
        """
        Get StudiesDB object.
        Returns:
            StudiesDB
        """
        return self._studies_db

    @property
    def input_dir(self):
        """
        Input directory.
        Alias property for self.ingress (just for clarity purposes).
        Returns:
            str
        """
        return self.ingress

    @property
    def sleep_for_seconds(self):
        """
        Number of seconds to sleep when the input folder is empty
        Returns:
            int
        """
        return self._sleep_for_seconds

    @property
    def use_transfer_complete_file(self):
        """
        Require a .transfer_complete file to start processing the study.
        Returns:
            bool
        """
        return self.new_study_threshold_seconds is None

    def get_next_study(self):
        """
        Iterator that reads studies that have not been processed yet from StudiesDB.
        If no studies are found, the process sleeps for some seconds

        Returns:
            study_name (str), file_list (list-str), index (int)
        """
        self.logger.info(log_line(-1, "Starting Input Dir Monitor. Waiting for new studies..."))
        while True:
            next_study = self._studies_db.get_oldest_unstarted_study()
            if next_study is None:
                self.logger.debug(log_line(-1, "No new studies found in DB. Checking the input folder..."))
                num_new_studies = self.check_new_studies()
                # Check for new studies in the input directory
                if num_new_studies == 0:
                    self.logger.debug(log_line(-1, f"No new studies found in the input dir. "
                                                   f"Sleeping for {self.sleep_for_seconds} seconds..."))
                    time.sleep(self.sleep_for_seconds)
            else:
                # A new study is ready for processing
                self.logger.debug(log_line(-1, f"A new study was read from StudiesDB:\n{next_study}"))
                input_path = next_study['input_path']
                try:
                    assert os.path.isdir(input_path), f"Folder does not exist: {input_path}"
                    file_list = []
                    for f in glob.glob(input_path + "/*"):
                        if os.path.basename(f).startswith("."):
                            # Ignore hidden files
                            continue
                        assert os.path.isfile(f), f"{f} does not exist or it does not seem to be a regular file."
                        file_list.append(f)

                    study_name = os.path.basename(input_path)
                    # Get the StudiesDB index
                    index = int(next_study.name)
                    self.logger.info(
                        log_line(-1, f"New study read from StudiesDB: {study_name}; Ix: {index}; Filelist: {file_list}"))
                    yield study_name, file_list, index
                except GeneratorExit:
                    # Exit the loop if the generator is stopped externally
                    self.logger.warning(log_line(-1, "Studies generator stopped externally"))
                    break
                except:
                    # IO Error. Mark the study as failed in StudiesDB so that it's not processed again
                    error_message = "Unexpected I/O error when processing the following path: " \
                                    f"{input_path};{traceback.format_exc()}"
                    self.logger.error(log_line(5, error_message))
                    self.studies_db.mark_study_as_failed(input_path, error_message)


    def check_new_studies(self):
        """
        Listen for existing folders in self.input_dir. For each folder, try to insert it in StudiesDB if it has not
        been inserted yet

        Returns:
            int. Number of new studies inserted in StudiesDB
        """
        num_new_studies = 0
        self.logger.debug(log_line(-1, f"Searching for new studies in {self.input_dir}..."))
        # Name of subdir in input_dir is Study Instance UID
        folder_names = self.get_folder_names(self.input_dir)
        if len(folder_names) > 0:
            self.logger.debug(log_line(-1, f"Found {len(folder_names)} studies to look at in the input folder"))
            for folder_name in folder_names:
                study_dir = "{}/{}".format(self.input_dir, folder_name)
                self.logger.debug(log_line(-1, f"Looking at folder {study_dir}..."))
                if self.use_transfer_complete_file:
                    # Search for transfer_complete_file to assume the study transfer is complete
                    if os.path.isfile(os.path.join(study_dir, const.TRANSFER_COMPLETE_FILE_NAME)):
                        self.logger.debug(log_line(-1, f"{const.TRANSFER_COMPLETE_FILE_NAME} found."
                                                      f" Attempting to insert {study_dir} in DB..."))
                        inserted = self.studies_db.insert_if_not_exist(study_dir)
                        if inserted:
                            num_new_studies += 1
                            self.logger.info(log_line(-1, f"{study_dir} INSERTED in StudiesDB"))
                        else:
                            self.logger.debug(log_line(-1, f"{study_dir} was NOT inserted in StudiesDB"))
                    else:
                        self.logger.debug(log_line(-1,
                                         f"{const.TRANSFER_COMPLETE_FILE_NAME} not found yet in {study_dir}"))
                else:
                    # Search for the most recent file and assume the study has been completely transferred if the
                    # newest file has been modified less than new_study_threshold_seconds seconds ago
                    latest_timestamp = self.get_latest_timestamp_utc_in_dir(study_dir)
                    if latest_timestamp is None:
                        self.logger.warning(log_line(5, f"Files not found in folder {folder_name}"))
                        continue
                    self.logger.debug(log_line(-1, f"Latest Timestamp found: {latest_timestamp}"))
                    current_timestamp = self.get_current_utctime()

                    diff_in_seconds = (current_timestamp - latest_timestamp).seconds
                    if diff_in_seconds > self.new_study_threshold_seconds:
                        self.logger.debug(log_line(-1, f"Attempting to insert {study_dir} in DB..."))
                        # Insert the study in the StudiesDB (if we didn't do it already)
                        inserted = self.studies_db.insert_if_not_exist(study_dir)
                        if inserted:
                            num_new_studies += 1
                            self.logger.info(log_line(-1, f"{study_dir} INSERTED in StudiesDB"))
                        else:
                            self.logger.debug(log_line(-1, f"{study_dir} was NOT inserted in StudiesDB"))
                    else:
                        self.logger.debug(log_line(-1, f"{diff_in_seconds}<{self.new_study_threshold_seconds}."
                                                        " Nothing to do yet"))
        return num_new_studies


    @staticmethod
    def get_current_utctime():
        """
        Get current time in UTC
        Returns:
            datetime
        """
        return datetime.datetime.utcnow()

    @classmethod
    def get_latest_timestamp_utc_in_dir(cls, folder):
        """
        Get the added time for the most recent file in a folder
        Args:
            folder (str): folder path
        Returns:
            datetime. Most recent date in utc time
        """
        timestamps = []
        for file_name in os.listdir(folder):
            filepath = "{}/{}".format(folder, file_name)
            # Get timestamp (in UTC) of file
            timestamp = os.path.getatime(filepath)  # added time
            timestamp = datetime.datetime.utcfromtimestamp(timestamp)
            timestamps.append(timestamp)
        if len(timestamps) > 0:
            latest_timestamp = max(timestamps)
            return latest_timestamp
        return None

    @classmethod
    def get_folder_names(cls, folder):
        """
        Get a list of subdirectory names in 'folder'

        Args:
            folder (str): folder path

        Returns:
            str-list. List of subdirectory names
        """

        folder_names = []
        for folder_name in os.listdir(folder):
            if folder_name.startswith("."):
                continue
            folder_path = os.path.join(folder, folder_name)
            if not os.path.isdir(folder_path):
                continue
            folder_names.append(folder_name)
        return folder_names