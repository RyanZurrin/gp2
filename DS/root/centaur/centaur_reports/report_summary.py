import datetime
import json
import numpy as np
import os
import pandas as pd
import copy

import deephealth_utils.misc.results_parser as results_parser
import deephealth_utils.data.parse_logs as parse_logs
import centaur_reports.constants as constants
from .report import Report


class SummaryReport(Report):
    """
    Report that aggregates the Study results and the individual Dicom results
    """
    def __init__(self, save_dir, logger=None):
        # Please note that the algorithm version (first parameter) is not expected to be used in this report
        super(SummaryReport, self).__init__(None, logger=logger)

        self.save_dir = save_dir
        self._studies_file_name = constants.SUMMARY_REPORT_STUDY_CSV
        self._dicoms_file_name = constants.SUMMARY_REPORT_DICOM_CSV
        self._dicoms_aggregated_file_name = constants.SUMMARY_REPORT_DICOM_AGGREGATED_PKL
        self.df_studies = None
        self.df_dicoms = None

        self.csv_header_studies = ['ix', 'StudyInstanceUID', 'output_dir', 'input_dir', 'total', 'L', 'R', 'total_category',
                                   'total_category_name', 'L_category', 'L_category_name', 'R_category', 'R_category_name',
                                   'timestamp']

        self.df_dicoms_aggregated = None
        self.csv_header_dicoms = ['ix', 'StudyInstanceUID', 'SOPInstanceUID', 'SOPClassUID', 'laterality', 'view',
                                  'output_dir', 'input_study_dir', 'input_file_name',
                                  'score_unflipped', 'score_flipped', 'score_total', 'timestamp']
        self._initialize()

    def _initialize(self):
        # if len(self.save_dir) == 0:
        #     raise ValueError("Save directory not set")
        studies_path = os.path.join(self.save_dir, self._studies_file_name)
        dicoms_path = os.path.join(self.save_dir, self._dicoms_file_name)
        if not os.path.isfile(studies_path) or not os.path.isfile(dicoms_path):
            # self._initialize_dataframes()
            self.df_studies = pd.DataFrame(columns=self.csv_header_studies[1:])
            self.df_studies.index.name = self.csv_header_studies[0]
            self.df_dicoms = pd.DataFrame(columns=self.csv_header_dicoms[1:])
            self.df_dicoms.index.name = self.csv_header_dicoms[0]
            return

        # Load only the columns used right now (ignore the rest)
        self.df_studies = pd.read_csv(studies_path, index_col=self.csv_header_studies[0],
                                      usecols=lambda x: x if x in self.csv_header_studies else False)
        # Insert possibly missing columns
        for col in [c for c in self.csv_header_studies[1:] if c not in self.df_studies.columns]:
            self.df_studies[col] = None

        # Same thing for df_dicoms
        self.df_dicoms = pd.read_csv(dicoms_path, index_col=self.csv_header_dicoms[0],
                                     usecols=lambda x: x if x in self.csv_header_dicoms else False)
        for col in [c for c in self.csv_header_dicoms[1:] if c not in self.df_dicoms.columns]:
            self.df_dicoms[col] = None

    def exists(self, studyInstanceUID, output_dir):
        """
        A study is already present in the report
        :param studyInstanceUID: str. Study UID
        :param output_dir: not used (inherited from the Report parent class)
        :return: bool
        """
        return studyInstanceUID in self.df_studies.index

    def get_img_index(self, study_uid, sop_instance_uid):
        """
        Build an index for a particular image in the dicoms_df
        :param study_uid: str. StudyInstanceUID for the study
        :param sop_instance_uid: str. SOPInstanceUID for the image
        :return: str. Built index
        """
        return "{}__{}".format(study_uid, sop_instance_uid)

    def generate(self, study_deploy_results, **additional_params):
        """
        It adds the data of a study to the existing dataframes
        Args:
            study_deploy_results (StudyDeployResults): Object that contains the results of the study

        Returns:
            dictionary of str ('studies_path', 'dicoms_path', 'dicoms_agg_path'): Paths to the generated reports
        """
        timestamp = datetime.datetime.now().isoformat()

        metadata = copy.copy(study_deploy_results.metadata)
        L_score = R_score = L_cat = R_cat = np.nan
        L_cat_name = R_cat_name = np.nan

        study_results = study_deploy_results.results['study_results']
        total_score = study_results['total']['score']
        total_cat = study_deploy_results.get_category()
        total_cat_name = study_deploy_results.get_category_name()
        if 'L' in study_results:
            L_score = study_results['L']['score']
            L_cat = study_results['L']['category']
            L_cat_name = study_results['L']['category_name']

        if 'R' in study_results:
            R_score = study_results['R']['score']
            R_cat = study_results['R']['category']
            R_cat_name = study_results['R']['category_name']


        study_id = study_deploy_results.get_studyUID()
        self.df_studies.loc[study_id] = [study_id, study_deploy_results.output_dir, study_deploy_results.input_dir,
                                         total_score, L_score, R_score,
                                         total_cat, total_cat_name, L_cat, L_cat_name, R_cat, R_cat_name, timestamp]

        for sop_id, dicom in study_deploy_results.results['dicom_results'].items():
            agg_scores = []
            if len(dicom['none']) > 0:
                scores_none = np.max([box['score'] for box in dicom['none']])
            else:
                scores_none = np.nan
            agg_scores.append(scores_none)
            if 'lr_flips' in dicom and len(dicom['lr_flips']) > 0:
                scores_flip = np.max([box['score'] for box in dicom['lr_flips']])
                agg_scores.append(scores_flip)
            else:
                scores_flip = np.nan
            rows = metadata[metadata['SOPInstanceUID'] == sop_id]
            assert len(rows) == 1, "Only one row expected for SOPInstanceUID={}; Got: {}".format(sop_id, rows)
            row = rows.iloc[0]
            lat = row['ImageLaterality']
            view = row['ViewPosition']
            class_id = row['SOPClassUID']
            input_file_name = row['dcm_path']
            ix = self.get_img_index(study_id, sop_id)
            self.df_dicoms.loc[ix] = [
                study_id,
                sop_id,
                class_id,
                lat,
                view,
                study_deploy_results.output_dir,
                study_deploy_results.input_dir,
                input_file_name,
                scores_none,
                scores_flip,
                np.mean(agg_scores),
                timestamp
            ]

        df = results_parser.parse_results(study_deploy_results, include_study_df=False)
        if self.df_dicoms_aggregated is None:
            self.df_dicoms_aggregated = df
        else:
            self.df_dicoms_aggregated = pd.concat([self.df_dicoms_aggregated, df])
            self.df_dicoms_aggregated = self.df_dicoms_aggregated.set_index(np.arange(len(self.df_dicoms_aggregated)))

        studies_path = os.path.join(self.save_dir, self._studies_file_name)
        dicoms_path = os.path.join(self.save_dir, self._dicoms_file_name)
        dicoms_agg_path = os.path.join(self.save_dir, self._dicoms_aggregated_file_name)
        self.df_studies.to_csv(studies_path)
        self.df_dicoms.to_csv(dicoms_path)
        self.df_dicoms_aggregated.to_pickle(dicoms_agg_path)
        return {'studies_path': studies_path, 'dicoms_path': dicoms_path, 'dicoms_agg_path': dicoms_agg_path}
