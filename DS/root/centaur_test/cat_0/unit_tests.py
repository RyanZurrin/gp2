import os
import tempfile
import sys
import subprocess
import pytest
import json
import copy
from centaur_deploy.run import Run
from centaur_deploy.deploy import Deployer
from centaur_deploy.config import Config
import centaur_deploy.constants as const_deploy
import deephealth_utils.misc.config as dh_config

from centaur_test.data_manager import DataManager

###################################################
# AUX FUNCTIONS
###################################################

@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    '''
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    '''
    record_xml_attribute("classname", "0_Unit")

###################################################
# DEPLOY RUN TEST FUNCTIONS
###################################################

def test_T_0_U01():
    '''
    Deploy Run tests
    '''
    data_manager = DataManager()

    config = Config.ProductionConfig()
    config[Config.MODULE_IO, 'input_dir'] = data_manager.get_input_dir(DataManager.STUDY_01_HOLOGIC)
    config[Config.MODULE_IO, 'output_dir'] = tempfile.mkdtemp()
    deployer = Deployer()
    deployer.initialize(config)

    run = Run(deploy_config=config)
    run_num, run_dir = run.save_run()
    print(run_dir)

    # verify that:
    # 1. the config parameters are correctly saved
    saved_config = Config.from_json_file(os.path.join(run_dir, const_deploy.CENTAUR_CONFIG_JSON))
    assert config == saved_config, "Saved config json is not correctly saved"
    # 2. conda packages are correctly saved
    current_packages = subprocess.check_output('conda list -e', shell=True).decode('utf-8')
    with open(os.path.join(run_dir, 'conda_packages.txt'), 'r') as fp:
        run_packages = fp.read()
    assert current_packages.strip() == run_packages.strip(), "Conda packages are not correctly saved"
    # 3. verify that the deploy command is the same
    python_cmd = 'python ' + ' '.join(sys.argv)
    with open(os.path.join(run_dir, const_deploy.CENTAUR_COMMAND_TXT), 'r') as fp:
        deploy_command = fp.read()
    assert python_cmd == deploy_command, "Shell command is not correctly saved"
    # 4. verify that the repo commits are the same
    git_commits = dh_config.load_config(os.path.join(run_dir, const_deploy.CENTAUR_COMMIT_FN))
    current_commits = Run.get_git_commits()
    assert git_commits == current_commits, "Git commits are not correctly saved"

    # Use the saved run, then edit errors into the config and test that load_run identities the errors
    git_commits_copy = copy.deepcopy(git_commits)
    git_commits_copy['centaur']['commit'] = '00000a9f1d691fa8eede3ee160b8b033d6800000'
    with open(os.path.join(run_dir, const_deploy.CENTAUR_COMMIT_FN), 'w') as fp:
        json.dump(git_commits_copy, fp)
    with pytest.raises(ValueError):
        Run.load_run(run_num=run_num, model_version=config[Config.MODULE_ENGINE, 'model_version'])

    git_commits_copy = copy.deepcopy(git_commits)
    git_commits_copy['centaur']['branch'] = 'wrong'
    with open(os.path.join(run_dir, const_deploy.CENTAUR_COMMIT_FN), 'w') as fp:
        json.dump(git_commits_copy, fp)
    with pytest.raises(ValueError):
        Run.load_run(run_num=run_num, model_version=config[Config.MODULE_ENGINE, 'model_version'])

    config_copy = copy.deepcopy(config)
    config[Config.MODULE_DEPLOY, 'run_mode'] = 'wrong_test'
    config.to_json_file(os.path.join(run_dir, const_deploy.CENTAUR_CONFIG_JSON))
    with pytest.raises(ValueError):
        Run.load_run(run_num=run_num, model_version=config[Config.MODULE_ENGINE, 'model_version'])