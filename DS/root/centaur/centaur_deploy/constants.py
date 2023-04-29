import os

if os.environ.get('CENTAUR_ROOT_PATH') is None:
    ROOT_PATH = '/root/'
else:
    ROOT_PATH = os.environ.get('CENTAUR_ROOT_PATH')

DEFAULT_OUTPUT_FOLDER = ROOT_PATH + "output/"
DEFAULT_INPUT_FOLDER = ROOT_PATH + "input/"
CLIENT_DIR = ROOT_PATH + "client_dir/"
CLIENT_CONFIG_FILE_NAME = "config.json"

MODEL_PATH = ROOT_PATH + 'models/'
THRESHOLDS_PATH = ROOT_PATH + 'threshold_versions/'
RETINANET_PATH = ROOT_PATH + 'keras-retinanet/'
DHUTILS_PATH = ROOT_PATH + 'deephealth_utils/'
DEPLOY_RUNS_PATH = ROOT_PATH + 'runs/'

LOCAL_TESTING_DATA_ROOT_FOLDER = ROOT_PATH + 'test_datasets/'
LOCAL_TESTING_STUDIES_FOLDER = LOCAL_TESTING_DATA_ROOT_FOLDER + 'data/studies/'
LOCAL_TESTING_BASELINE_FOLDER = LOCAL_TESTING_DATA_ROOT_FOLDER + 'baseline/centaur_output/'


CENTAUR_CONFIG_JSON = 'centaur_config.json'
CENTAUR_STUDY_DEPLOY_RESULTS_JSON = 'results_full.json'
CENTAUR_COMMAND_TXT = 'deploy_command.txt'
CENTAUR_ERRORS_TRACKING = 'centaur_errors_tracking.csv'
CENTAUR_COMMIT_FN = 'repo_commits.json'
STUDIES_DB_FILE_NAME = 'studies.db'
ALGORITHM_VERSION_PATH = ROOT_PATH + "algorithm_version.txt"

# DEPLOY MODES
RUN_MODE_CADT = 'CADt'
RUN_MODE_CADX = 'CADx'
RUN_MODE_DEMO = 'DEMO'

# LABELS
LABEL_SAIGE_Q = \
""" 
 ==================================================================
||        Saige-Q Screening Mammogram Workflow Software           ||
||        DeepHealth, Inc.                                         ||
||        Version [VERSION_PLACEHOLDER]                           ||
||        UDI: (01)00860005144006(10)200                          ||
||        Copyright © 2021 by DeepHealth, Inc.                     ||
||        Bringing the Best Doctor in the World to Every Patient  ||
 ==================================================================
"""
LABEL_SAIGE_DX = \
""" 
 ==================================================================
||        Saige-Dx Screening Mammogram Workflow Software          ||
||        DeepHealth, Inc.                                         ||
||        Version [VERSION_PLACEHOLDER]                           ||
||        UDI: (XX) XXXXXXXXXXXXXX(XX)XXX                         ||
||        Copyright © 2021 by DeepHealth, Inc                     ||
||        Bringing the Best Doctor in the World to Every Patient  ||
 ==================================================================
"""