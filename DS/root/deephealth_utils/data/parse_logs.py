import re
import warnings
import pandas as pd
import os
import ast
import glob

'''
Helper Functions
'''
FILE_PREFIXES = ['FAC', 'IAC', 'GE', 'HOL']
STUDY_PREFIXES = ['SAC', 'PAC']


def get_error_checking_dfs(error_tracking_df, output_dir, bad_entries_only=False):
    """
    Checks whether there are any studies or files that encounter unexpected errors and/or have no predictions

    Args:
        error_tracking_df (DataFrame): DF indicating whether a given file has predictions and whether it encountered
                                       any unexpected errors.
        output_dir (str): The path to the directory containing Centaur outputs.
        bad_entries_only (bool): If True, return only files and studies that encounter unexpected errors and/or have
                                 no predictions. Otherwise, return all merged files/studies.

    References:
        centaur_reports.constants

    Returns:
        2-DataFrame tuple. file-level and study-level DataFrames. Contents depend on the bad_entries_only argument.
    """
    import centaur_reports.constants as const_reports

    def get_log_file_path(output_dir):
        f = glob.glob(output_dir + "/*.log")
        assert len(f) == 1, "Found {} log files in {} folder".format(len(f), output_dir)
        return f[0]
    log_path = get_log_file_path(output_dir)
    reader = LogReader(log_path)
    dicom_checks_df = reader.get_file_df()
    dicom_checks_df['file_path'] = dicom_checks_df['study'] + '/' + dicom_checks_df['filename']
    study_checks_df = reader.get_study_df()
    study_checks_df = study_checks_df.rename(columns={'filename': 'study_path'})
    dicom_summary_df = pd.read_csv(os.path.join(output_dir, const_reports.SUMMARY_REPORT_DICOM_CSV)).rename(columns={'input_file_name': 'file_path'})
    study_summary_df = pd.read_csv(os.path.join(output_dir, const_reports.SUMMARY_REPORT_STUDY_CSV)).rename(columns={'input_dir': 'study_path'})
    box_summary_df = pd.read_pickle(os.path.join(output_dir, const_reports.SUMMARY_REPORT_DICOM_AGGREGATED_PKL))

    file_merge_df = pd.merge(error_tracking_df, dicom_checks_df.loc[:, ['file_path', 'passes_checker', 'failed_checks']], on='file_path', how='left')
    file_merge_df = pd.merge(file_merge_df, dicom_summary_df.drop(columns=['ix', 'output_dir', 'timestamp']).rename(columns={'score_unflipped': 'file_score_unflipped',
                                                                                                                        'score_flipped': 'file_score_flipped',
                                                                                                                        'score_total': 'file_score_total'}), on='file_path', how='left')
    file_merge_df = pd.merge(file_merge_df, box_summary_df.drop(columns=['StudyInstanceUID', 'timestamp']).rename(columns={'score': 'box_score'}), on='SOPInstanceUID', how='left')
    bad_files = file_merge_df[(file_merge_df['unexpected_errors'].notnull()) | ((file_merge_df['passes_checker'] == True) & (file_merge_df['in_results'] == False))]

    # If there are unexpected errors, all files in the study will have the same value for 'unexpected_errors'
    study_error_tracking_df = error_tracking_df.groupby('study_path').apply(lambda df: pd.Series({'in_results': df['in_results'].max(),
                                                                                                  'unexpected_errors': df['unexpected_errors'].values[0]})).reset_index()
    study_merge_df = pd.merge(study_error_tracking_df, study_checks_df, on='study_path', how='left')
    study_merge_df = pd.merge(study_merge_df, study_summary_df.drop(columns=['ix', 'output_dir', 'timestamp']).rename(columns={'total': 'study_score', 'L': 'left_score', 'R': 'right_score',
                                                                                                                               'total_category': 'study_category'}), on='study_path', how='left')
    bad_studies = study_merge_df[(study_merge_df['unexpected_errors'].notnull()) | ((study_merge_df['passes_checker'] == True) & (study_merge_df['in_results'] == False))]

    if bad_entries_only:
        return bad_files, bad_studies
    else:
        return file_merge_df, study_merge_df

def log_line(code, input, study_dir=None):
    '''
    :param code:
        -1: Do not parse
        0: Beginning of a study
        1.1: Study successful passed all checks
        1.2.1: Study failed checks since no files passed checks
        1.2.2: Study failed checks, and checks failed are logged
        2: Message regarding processing of a file
        3.1: File finished checks, succeeded
        3.2: File finished checks, failed
        4: Timestamp logging
        5: Error
    :param input:
        Message or data to be stored
    '''
    if str(code) == '5':
        # Usually an exception. Remove all the line breaks
        input = input.replace("\n", ";   ")

    return '{} @@ {} @@ {}'.format(code, input, study_dir)

def parse_path(full_path):

    path_arr = full_path.split('/')

    root = '/'.join(path_arr[0:-2])
    study = path_arr[-2]
    file = path_arr[-1]

    return root, study, file

def parse_dictionary(s):

    try:
        return ast.literal_eval(s)
    except:
        temp_dict = {"other":s}
        return temp_dict


def combine_logs(log_paths):
    """
    Combine multiple log files in one.

    Args:
        log_paths (str-list): list of full paths to log files
    References:
        centaur_engine.helpers

    Returns:
        str. Path to the combined log file

    """
    # Combine all logs into one
    from centaur_engine.helpers import helper_misc

    logger, log_path = helper_misc.create_logger('logs', return_path=True)
    print('Joining logs to to {}'.format(log_path))

    with open(log_path, "wb") as outfile:
        for f in log_paths:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
    return log_path

def _passes_checks(row):
    return int(len(row['failed_checks']) == 0)

def get_profile_df(log_file_path):
    """
    Read a log file and get all the profiling lines in a dataframe
    :param log_file_path: str. Path to the Centaur log file
    :return: Dataframe with ['event', 'time_seconds', 'memory_bytes', 'data', 'timestamp']
    """
    with open(log_file_path, 'r') as f:
        full_text = f.read()
    df = pd.DataFrame(columns=['event', 'time_seconds', 'memory_bytes', 'data', 'timestamp'])

    i = 0
    for m in re.finditer("^(?P<timestamp>.*) @@ (.*@@ ){2,2}4 @@ (?P<event>.*):(?P<time>.*)s;MEM:(?P<mem>.*) @@ (?P<data>.*)$", full_text,
                         re.MULTILINE):
        df.loc[i] = [m.group('event'), m.group('time'), m.group('mem'), m.group('data'), m.group('timestamp')]
        i += 1

    # Adjust last row (DEPLOY TOTAL RUN)
    row = df.iloc[-1]
    ev = row['event']
    if ev.startswith('DEPLOY_TOTAL_RUN'):   # Sanity check
        ix = ev.index(':')
        event = ev[:ix]
        time_ = ev[ix+1:]
        df.loc[row.name, 'event'] = event
        seconds = row['time_seconds'].split('.')[0]
        total_time = "{}:{}".format(time_, seconds)
        df.loc[row.name, 'data'] = total_time
        # Compute the total number of seconds
        spl = total_time.split(':')
        total_seconds = int(spl[0]) * 3600 + int(spl[1]) * 60 + int(spl[2])
        df.loc[row.name, 'time_seconds'] = str(total_seconds)
    return df


VALID_CODES = ['-1', '0', '0.1.1', '0.1.2', '1.1', '1.2.1', '1.2.2', '2', '3.1', '3.2', '4', '5']

'''
Log Reader Object
'''
class LogReader:
    '''
    Parser object for logs.

    __init__ takes in log_path

    Example:
        reader = LogReader(log_path)
        reader.iter_lines()
        file_df = reader.get_file_df()
        study_df = reader.get_study_df()
        reader.count_study_ac()
        reader.count_file_ac()
        reader.list_rejects('FAC-20')
        rejected_studies = reader.rejected_studies()
        rejected_files = reader.rejected_files()

    '''
    def __init__(self, log_path):

        with open(log_path) as fp:
            self.lines = fp.readlines()

        self.p_vars = {}  # variables needed for one process_id

        self.n_studies = 0
        self.n_files = 0

        self.dict = {}

        current_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.ac_table = pd.read_csv(os.path.join(current_folder, "dicom_specs/checks_modes.csv"))

        self.iter_lines()
        self.file_df = self.get_file_df()
        self.study_df = self.get_study_df()

    def add_process_id(self, p_id):

        if p_id not in self.p_vars:

            self.p_vars[p_id] = {'dict': {},
                                 'current_study': None,
                                 'current_file': None,
                                 'passed': None,
                                 'failed_checks': None,
                                 'n_studies': 0,
                                 'n_files': 0}

        return

    def triage_line(self, line):
        """
        Each line contains a code which indicates the following:
            -1: Do not parse
            0: Beginning of a study
            0.1.1: Number of studies
            0.1.2: Number of files
            1.1: Study successful passed all checks
            1.2.1: Study failed checks since no files passed checks
            1.2.2: Study failed checks, and checks failed are logged
            2: Message regarding processing of a file
            3.1: File finished checks, succeeded
            3.2: File finished checks, failed
        List of valid codes found in constant VALID_CODES defined above.
        """
        line_split = [part.strip(' ').strip('\n') for part in line.split('@@')]
        if len(line_split) != 6:
            warnings.warn("Ignored line in log parsing (wrong format): {}".format(line))
            return

        time, module, p_id, code, message, study_dir = line_split

        self.add_process_id(p_id)

        if code not in VALID_CODES:
            raise ValueError("Invalid code used in line {}".format(line))

        line_triaged = True

        if module == 'deploy.py':
            if code.startswith('0'):
                if code.startswith('0.1'):
                    if code.startswith('0.1.1'):
                        self.p_vars[p_id]['n_studies'] = int(message.split(' ')[0])
                    elif code.startswith('0.1.2'):
                        self.p_vars[p_id]['n_files'] = int(message.split(' ')[0])
                    else:
                        line_triaged = False
                else:
                    line_triaged = False
            else:
                line_triaged = False

        elif module == 'preprocessor.py':
            if code.startswith('0'):  # Beginning of a study
                self.p_vars[p_id]['current_study'] = study_dir
            elif code.startswith('1'):  # Study level checks
                self.p_vars[p_id]['current_study'] = study_dir
                if code.startswith('1.1'):
                    self.p_vars[p_id]['passed'] = True
                    self.p_vars[p_id]['failed_checks'] = {}
                elif code.startswith('1.2'):  # Study fails (broad category)
                    self.p_vars[p_id]['passed'] = False
                    if code.startswith('1.2.1'):  # No files left to do study checks
                        self.p_vars[p_id]['failed_checks'] = {'SAC-NA': None}
                    elif code.startswith('1.2.2'):  # There are study checks that failed
                        self.p_vars[p_id]['failed_checks'] = parse_dictionary(message)
                    else:
                        line_triaged = False
                else:
                    line_triaged = False
                self.p_vars[p_id]['current_file'] = None  # Resets current_file to None to begin new study
                self.update_dict(p_id, 'study')
            elif code.startswith('2'):  # File level messages
                _, _, file = parse_path(message)
                self.p_vars[p_id]['current_study'] = study_dir
                self.p_vars[p_id]['current_file'] = file
            elif code.startswith('3'):  # File level checks
                self.p_vars[p_id]['current_study'] = study_dir
                if code.startswith('3.1'):  # File checks all pass
                    self.p_vars[p_id]['passed'] = True
                    self.p_vars[p_id]['failed_checks'] = {}
                elif code.startswith('3.2'):  # Some file checks have failed
                    self.p_vars[p_id]['passed'] = False
                    self.p_vars[p_id]['failed_checks'] = parse_dictionary(message)
                else:
                    line_triaged = False
                self.update_dict(p_id, 'file')
            else:
                line_triaged = False

        if not line_triaged and code not in ("-1", "4", "5"):
            raise Exception('Line not triaged: {}'.format(line))

    def update_dict(self, p_id, file_or_study):

        # New study to add to dictionary
        if self.p_vars[p_id]['current_study'] is not None and \
                self.p_vars[p_id]['current_study'] not in self.p_vars[p_id]['dict']:
            self.p_vars[p_id]['dict'][self.p_vars[p_id]['current_study']] = {}

        # New file to add to dictionary
        if self.p_vars[p_id]['current_file'] is not None \
                and self.p_vars[p_id]['current_file'] \
                not in self.p_vars[p_id]['dict'][self.p_vars[p_id]['current_study']]:
            self.p_vars[p_id]['dict'][self.p_vars[p_id]['current_study']][self.p_vars[p_id]['current_file']] = {}

        # Add failed checks (if any) from study to dictionary
        if self.p_vars[p_id]['current_study'] is not None and file_or_study == 'study' \
                and self.p_vars[p_id]['failed_checks'] is not None:
            study_dict = dict()
            study_dict['Passed'] = self.p_vars[p_id]['passed']
            study_dict['FailedChecks'] = self.p_vars[p_id]['failed_checks']
            self.p_vars[p_id]['dict'][self.p_vars[p_id]['current_study']]['Status'] = study_dict

        # Add failed checks (if any) from file to dictionary
        if self.p_vars[p_id]['current_file'] is not None and file_or_study == 'file':
            file_dict = dict()
            file_dict['Passed'] = self.p_vars[p_id]['passed']
            file_dict['FailedChecks'] = self.p_vars[p_id]['failed_checks']
            self.p_vars[p_id]['dict'][self.p_vars[p_id]['current_study']][self.p_vars[p_id]['current_file']] = file_dict

    def iter_lines(self):
        for self.line_num, line in enumerate(self.lines):
            self.triage_line(line)

        # Combine dictionaries from p_ids

        for p_id in self.p_vars:
            self.dict.update(self.p_vars[p_id]['dict'])
            self.n_studies += self.p_vars[p_id]['n_studies']
            self.n_files += self.p_vars[p_id]['n_files']

    def map_ac(self, ac):
        if '|' in ac:  # If ac is a tuple, split it and recurse
            return '|'.join([self.map_ac(each_ac) for each_ac in ac.split('|')])
        else:
            match = self.ac_table[self.ac_table['Acceptance Criteria Number'] == ac]['Tag']
            if len(match) == 0:
                return
            else:
                return match.iloc[0]

    def get_df(self, level):
        mapping = dict()
        for study in self.dict:
            if level == 'file':
                for dicom in [d for d in self.dict[study] if d != 'Status']:
                    failed_checks = self.dict[study][dicom]['FailedChecks']
                    mapping[dicom] = dict()
                    mapping[dicom]['study'] = study
                    failed_ac_list = list(failed_checks.keys())
                    mapping[dicom]['failed_checks'] = '|'.join(failed_ac_list)
                    for ac in failed_checks:
                        if failed_checks[ac] is None:
                            failed_checks[ac] = 'No info'
                        mapping[dicom][ac] = failed_checks[ac]
            elif level == 'study':
                if 'Status' in self.dict[study]:  # Some examples not completed yet if log is streaming
                    failed_checks = self.dict[study]['Status']['FailedChecks']
                    mapping[study] = dict()
                    failed_ac_list = list(failed_checks.keys())
                    mapping[study]['failed_checks'] = '|'.join(failed_ac_list)
                    for ac in failed_checks.keys():
                        if failed_checks[ac] is None:
                            failed_checks[ac] = 'No info'
                        mapping[study][ac] = failed_checks[ac]
            else:
                raise ValueError('Only file and study are accepted as levels')

        df = pd.DataFrame.from_dict(mapping, orient='index')
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'filename'}, inplace=True)
        df['failed_checks'] = df['failed_checks'].astype(str)
        df = pd.DataFrame(df)
        df['passes_checker'] = df.apply(_passes_checks, axis=1)
        return df


    def get_file_df(self, save_path=None):

        df = self.get_df('file')

        if save_path is not None:
            df.to_pickle(save_path)

        return df

    def get_study_df(self, save_path=None):

        df = self.get_df('study')

        if save_path is not None:
            df.to_pickle(save_path)

        return df

    def turn_readable(self, df):
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'Acceptance Criteria'}, inplace=True)
        df['Tag'] = df['Acceptance Criteria'].apply(self.map_ac)
        if 'failed_checks' in df.columns:
            df.rename(columns={'failed_checks': 'Counts'}, inplace=True)
        else:
            df.rename(columns={0: 'Counts'}, inplace=True)
        df.sort_values(by='Counts', ascending=False, inplace=True)
        df.reset_index(level=0, inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        return df

    def summarize_studies(self):
        df = pd.DataFrame(self.study_df.failed_checks.value_counts())
        return self.turn_readable(df)


    def summarize_files(self):
        df = pd.DataFrame(self.file_df.failed_checks.value_counts())
        return self.turn_readable(df)


    def count_study_ac(self):
        df = pd.DataFrame(self.study_df[[col for col in self.study_df.columns if
                                         any([prefix in col for prefix in STUDY_PREFIXES])]].count())
        return self.turn_readable(df)

    def count_files_ac(self):
        df = pd.DataFrame(self.file_df[[col for col in self.file_df.columns if
                                         any([prefix in col for prefix in FILE_PREFIXES])]].count())
        return self.turn_readable(df)

    def list_rejects(self, acceptance_criteria):
        if any(dcm_tag in acceptance_criteria for dcm_tag in FILE_PREFIXES):
            df = self.file_df
        elif any(dcm_tag in acceptance_criteria for dcm_tag in STUDY_PREFIXES):
            df = self.study_df
        return df[df['failed_checks'].str.contains(acceptance_criteria)][acceptance_criteria].value_counts()

    def rejected_studies(self, ignore_criteria=[]):
        df = self.study_df
        if len(ignore_criteria) > 0:
            df = df[~df.failed_checks.str.contains('|'.join(ignore_criteria))]
        return df[df.failed_checks.str.len() > 0]['filename']

    def rejected_files(self, ignore_criteria=[]):
        df = self.file_df
        if len(ignore_criteria) > 0:
            df = df[~df.failed_checks.str.contains('|'.join(ignore_criteria))]
        return df[df.failed_checks.str.len() > 0]['filename']
