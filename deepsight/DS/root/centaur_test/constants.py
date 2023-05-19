STUDY_GT_DF = "study_gt.csv"
FILE_GT_DF = "file_gt.csv"
AGG_BOX_DF = 'agg_box_preds.pkl'

BASELINE_BENCHMARK_COMPARISON_LOGNAME = 'comparison.log'
BASELINE_COMPARISON_RUN_BENCHMARK_SAVE_DIR = 'current_run_benchmark_summary'
BASELINE_OLD_BASELINE_RUN_BENCHMARK_SAVE_DIR = 'old_baseline_benchmark_summary'

CENTAUR_TEST_DATA_FOLDER = "test_datasets/data"
CENTAUR_TEST_BASELINE_FOLDER = "test_datasets/baseline"

TEST_DATA_FOLDER = "/data/dh_dcm_testing"
TEST_BASELINE_FOLDER = "/data/dh_test_results/baselines"

# FOLDER CONSTANTS
STUDIES_FOLDER = "studies"
CENTAUR_OUTPUT_FOLDER = "centaur_output"
# AGG_RESULTS_FOLDER = "agg_results"
TEST_OUTPUT_FOLDER = "test_output"
TEST_REPORTS_FOLDER = "test_reports"
BM_OUTPUT_FOLDER = "bm_output"
BM_REPORTS_FOLDER = "bm_reports"

FILELIST = "filelist"

# BASELINE CHECKS
CUDA_MIN_VERSION = "10.0.0"
NVIDIA_DRIVERS_MIN_VERSION = "418.126.02"
MIN_MEMORY_GB = 32
MIN_MEMORY_GPU_GB = 12
BASELINE_MAX_RUN_TIME_SECONDS = 1000     # For reference: AWS p2.xlarge = 320s for 2 dxm and 1 dbt
BASELINE_MAX_MEMORY_GB = 20
MIN_HARD_DRIVE_STORAGE_GB = 100


INDEX_3_MONTH_LABEL = 'index_3_months'
INDEX_1_YEAR_LABEL = 'index_1_year'
PRE_INDEX_LABEL = 'pre_index'

CONFIRMED_NEGATIVE_LABEL = 'confirmed_negative'
UNCONFIRMED_NEGATIVE_LABEL = 'unconfirmed_negative'

BENIGN_LABEL = 'benign'

LABEL_MAP = {
    INDEX_3_MONTH_LABEL: 1,
    INDEX_1_YEAR_LABEL: 1,
    PRE_INDEX_LABEL: 1,
    CONFIRMED_NEGATIVE_LABEL: 0,
    UNCONFIRMED_NEGATIVE_LABEL: 0,
    BENIGN_LABEL: 0
}


class BM01:
    FILE_CHECKS = 'file_checks'
    STUDY_CHECKS = 'study_checks'
    PASS_FAIL_STATS = 'pass_fail_stats'


class BM02:
    STUDY_AUCS = 'study_aucs'
    LABEL_DATA = 'label_data'


class BM03:
    STUDY_LOCALIZATION_AUC = 'study_localization'
    STUDY_HITS = 'study_hits'
    LABEL_DATA = 'label_data'


class BM04:
    SUSPICION_LEVELS = 'suspicion_levels'
    SCORE_PERCENTILES = 'score_percentiles'
    LABEL_SCORES = 'label_scores'


class BM05:
    BOX_STATS = 'box_stats'


class BM06:
    STUDY_LOCALIZATION_AUC = 'study_localization'
    STUDY_HITS = 'study_hits'
    HIT_PARAMS = 'hit_params'
    LABEL_DATA = 'label_data'
    SLICE_THICKNESS = 1
    Z_HIT_RANGE = 15


REPORT_BASE_FILE = 'report_base.html'