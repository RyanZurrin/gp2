import os
import shutil
import tempfile

import pytest

from centaur_deploy.deploys.config import Config

from centaur_deploy.deploy import Deployer
from centaur_test.config_factory import ConfigFactory
from centaur_test.data_manager import DataManager

import centaur_test.utils as utils

"""
Basic pipeline tests that run a study end to end and compare the results to the baseline
"""

@pytest.mark.basic_pipeline
def test_T_BP_Dxm(run_mode):
    """
    Full deploy for the first DXM study in the dataset. No PACS - All reports.
    The results are compared to the baseline
    """
    study = DataManager.STUDY_01_HOLOGIC
    utils.test_full_deploy(run_mode, compare_to_baseline=True, filtered_studies=[study])
    # run_pipeline_single_study_with_deployer(study, run_mode)

@pytest.mark.basic_pipeline
def test_T_BP_Dbt(run_mode):
    """
    Full deploy for the first DBT study in the dataset. No PACS - All reports.
    The results are compared to the baseline
    """
    study = DataManager.STUDY_03_DBT_HOLOGIC
    utils.test_full_deploy(run_mode, compare_to_baseline=True, filtered_studies=[study])

@pytest.mark.basic_pipeline
def test_T_BP_Dxm_run_only(run_mode):
    """
    Full deploy for the first DXM study in the dataset without comparing to baseline (just run)
    """
    study = DataManager.STUDY_01_HOLOGIC
    utils.test_full_deploy(run_mode, compare_to_baseline=False, filtered_studies=[study])

@pytest.mark.basic_pipeline
def test_T_BP_DBT_run_only(run_mode):
    """
    Full deploy for the first DXM study in the dataset without comparing to baseline (just run)
    """
    study = DataManager.STUDY_03_DBT_HOLOGIC
    utils.test_full_deploy(run_mode, compare_to_baseline=False, filtered_studies=[study])


if __name__=="__main__":
    test_T_BP_Dxm('CADx')
    print("OK!")