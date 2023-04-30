import os
from centaur_deploy.constants import MODEL_PATH, ROOT_PATH

TMP_NP_PATH = ROOT_PATH + 'temp_files/'

MODEL_CONFIG_JSON = 'config.json'
CADT_THRESHOLD_PATH = ROOT_PATH + 'threshold_versions/cadt/'
CADX_THRESHOLD_PATH = ROOT_PATH + 'threshold_versions/cadx/'

METADATA_PKL_NAME = 'metadata.pkl'

SYNTHETIC_IM_FILENAME = 'frame_synth.npy'
SYNTHETIC_SLICE_MAP_FILENAME = 'synth_slice_map.npy'

MODEL_VERSION = 'current'

DEMO_METADATA_FILENAME = 'study_metadata.pkl'

CATEGORY_MAPPINGS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/category_mappings.json')
