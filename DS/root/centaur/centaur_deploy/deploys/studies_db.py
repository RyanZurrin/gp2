import datetime
import logging
import sqlite3
import os.path as osp
import warnings

import pandas as pd
import centaur_io.constants as const


class StudiesDB(object):
    CODE_MAPPINGS = {
        ('processing_status_code', 0): "Processing of study has not started",
        ('processing_status_code', 1): "Processing of study in progress",
        ('processing_status_code', 2): "Processing of study completed",

        ('results_status_code', -1): "Study output not completed",
        ('results_status_code', 0): "Study successfully processed",
        ('results_status_code', 1): "Study was not successfully processed due to an unexpected error",
        ('results_status_code', 2): "Study was not processed due to failure of acceptance criteria",

        ('sending_status_code', -1): "Outputs not attempted to send yet",
        ('sending_status_code', 0): "Outputs successfully received by client clinical systems",
        ('sending_status_code', 1): "Outputs not successfully received by client clinical systems",
    }

    def __init__(self, db_path, code_mappings=None):
        """
        Constructor.
        If the DB file does not exist, it will create the StudiesDB table
        Args:
            db_path (str): Path to the DB file
        """
        self._db_path = db_path
        self._studies_table = const.STUDIES_TABLE
        self._code_mappings = self.CODE_MAPPINGS if code_mappings is None else code_mappings
        self._initialize()

    #region Public operations

    @property
    def studies_table(self):
        return self._studies_table

    def get_all(self):
        """
        Get all the rows in the database in a Dataframe
        Returns:
            Dataframe
        """
        return self._run_query(f"SELECT * FROM {self.studies_table}")

    def insert_study(self, input_path):
        """
        Insert a new study
        Args:
            input_path (str): input study path
        """
        sql = f"INSERT INTO {self.studies_table} (" \
                "input_path, timedate_received,"\
                "processing_status_code, processing_status_text,"\
                "results_status_code,results_status_text," \
                "sending_status_code,sending_status_text" \
                " ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
        params = (input_path, datetime.datetime.utcnow(),
                  0, self._code_mappings[('processing_status_code', 0)],
                  -1, self._code_mappings[('results_status_code', -1)],
                  -1, self._code_mappings[('sending_status_code', -1)],
                  )
        self._run_non_query(sql, params=params)

    def get_unstarted_studies(self):
        """
        Get a dataframe with all the studies that have been inserted but not processed yet
        Returns:
            Dataframe
        """
        query = f"SELECT * FROM {self._studies_table} WHERE processing_status_code=0"
        return self._run_query(query)

    def get_oldest_unstarted_study(self):
        """
        Get a Series with the oldest study that has not been processed yet (or None if no pending studies)
        Returns:
            pandas Series or None
        """
        query = f"SELECT * FROM {self._studies_table} WHERE processing_status_code=0 ORDER BY timedate_received LIMIT 1"
        df = self._run_query(query)
        if len(df) == 0:
            return None
        return df.iloc[0]

    def update_study(self, study_id, params_dict):
        """
        Update A study object.
        The method ensures that only and only one row is updated
        Args:
            study_id (int): study id in the database
            params_dict (dictionary): dictionary of values to update. Each entry in the dictionary should have the field
                                      to update as key.

        """
        sql = f"UPDATE {self.studies_table} SET "
        params_sql = {}
        for field, value_ in params_dict.items():
            sql += f"{field}=:{field},"
            params_sql[field] = value_
        sql = sql[:-1] + f" WHERE id={study_id}"
        self._run_non_query(sql, params_sql, single_row_operation=True)

    def query_by_input_path(self, input_path):
        """
        Search for a study given an input path
        Args:
            input_path (str): path to the study

        Returns:
            Dataframe. All the rows with the given input path (usually just one)
        """
        query = f"SELECT * FROM {self.studies_table} WHERE input_path = ?;"
        rows = self._run_query(query, params=input_path)
        return rows

    def insert_if_not_exist(self, input_path):
        """
        Insert a new study in the database if it doesn't exist already.
        Args:
            input_path (str): input path for a study

        Returns:
            bool. True if the study was inserted in DB, or False otherwise (the study was already in DB)
        """

        # query by input_path
        rows = self.query_by_input_path(input_path)

        # Check whether study (saved in 'input_path') is present in StudiesDB
        # if absent, insert a row the the study
        if len(rows) == 0:
            self.insert_study(input_path)
            return True
        return False

    def mark_study_as_failed(self, input_path, error_message):
        """
        Mark a study as failed because of an uncontrolled error, so that it is not processed again.
        IMPORTANT: The state will be modified only when the study has not been started
        Args:
            input_path (str): Path to the input study
            error_message (str): error message saved to 'results_error_message'
        """

        study = self.query_by_input_path(input_path)
        assert len(study) != 0, f"Study {input_path} not found in StudiesDB"
        assert len(study) == 1, f"Study {input_path} is duplicated in StudiesDB"
        study = study.iloc[0]

        if study['processing_status_code'] == 0:
            processing_status_code = 2     # Study finished processing
            results_status_code = 2        # Unexpected error happened

            params = {
                'processing_status_code': processing_status_code,
                'processing_status_text': self.CODE_MAPPINGS[('processing_status_code', processing_status_code)],
                'results_status_code': results_status_code,
                'results_status_text': self.CODE_MAPPINGS[('results_status_code', results_status_code)],
                'results_error_message': error_message,
                'timedate_processed': datetime.datetime.utcnow()
            }

            self.update_study(study.name, params)
        else:
            warnings.warn(f"The study {input_path} was not marked as failed because "
                          f"processing_status_code=={study['processing_status_code']}")

    def reset_db(self):
        """
        Removes all the information from the studies table
        """
        sql = f"DELETE FROM {self.studies_table}"
        self._run_non_query(sql)

    #endregion

    #region Protected aux methods
    def _create_connection(self):
        """
        Create a new connection to the DB
        Returns:
            sqlite3 connection
        """
        return sqlite3.connect(self._db_path)

    def _initialize(self):
        """
        Create a new Database or load an existing one.
        If the DB file does not exist yet, ensure the folder where it's going to be stored does
        """
        if osp.isfile(self._db_path):
            logging.getLogger().warning("StudiesDB already existing!")
        else:
            # The db does not exist yet. Check the folder where it's going to be stored does exist
            folder = osp.dirname(self._db_path)
            assert osp.isdir(folder), f"Folder {folder} does not exist. The database cannot be created"

            # Create the database
            self._create_studies_table()
            logging.getLogger().info("StudiesDB was created")

    def _create_studies_table(self):
        """
        Create the Studies table
        """
        sql = f"""CREATE TABLE IF NOT EXISTS {self.studies_table}
                    (id INTEGER PRIMARY KEY,
                    input_path TEXT,
                    output_path TEXT,
                    study_instance_uid TEXT,
                    accession_number TEXT,
                    timedate_received DATETIME,
                    timedate_processed DATETIME,
                    processing_status_code INTEGER,
                    processing_status_text TEXT,
                    results_status_code INTEGER,
                    results_status_text TEXT,
                    results_error_message TEXT,
                    sending_status_code INTEGER,
                    sending_status_text TEXT,
                    sending_error_message TEXT                
                    );"""
        self._run_non_query(sql)

    def _run_non_query(self, sql, params=None, single_row_operation=False):
        """
        Execute a SQL statement that does not return any value (insertions, updates, etc.)
        Args:
            sql (str): sql instruction to execute
            params (tuple-str): parameters
            single_row_operation (bool): when True, assert that only one row was updated
        Returns:
            (int) Number of affected rows
        """
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            if params is None:
                c = cursor.execute(sql)
            else:
                if not isinstance(params, tuple) and not isinstance(params, list) and not isinstance(params, dict):
                    params = (params,)
                c = cursor.execute(sql, params)
            if single_row_operation:
                # Assert that only one row was updated
                assert c.rowcount == 1, "{} rows would be affected by this change (single row expected)".format(c.rowcount)
            conn.commit()
            return c.rowcount
        finally:
            conn.close()

    def _run_query(self, query, params=None):
        """
        Execute a SQL query that returns a dataframe
        Args:
            query (str): query to execute
            params (tuple-str): parameters
        Returns:
             pandas dataframe
        """
        conn = self._create_connection()
        try:
            if params is not None and \
               (not isinstance(params, tuple) and not isinstance(params, list)):
                params = (params,)
            results = pd.read_sql_query(query, conn, params=params, index_col='id')
            return results
        finally:
            conn.close()

    #endregion

