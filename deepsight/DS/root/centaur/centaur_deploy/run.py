import sys

import centaur_deploy.constants as const
import glob
import json
import numpy as np
import os
import subprocess
from centaur_deploy.deploys.config import Config



class Run(object):

    def __init__(self, deploy_config={}):
        self.deploy_config = deploy_config
        self.python_cmd = 'python ' + ' '.join(sys.argv)

    def get_next_run(self, base_dir, prefix):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        runs = glob.glob(os.path.join(base_dir, prefix + '*'))
        try:
            run_num = int(np.sort(runs)[-1].split(prefix)[1].split('/')[0]) + 1
        except:
            run_num = 0
        run_name = '{}{:04d}'.format(prefix, run_num)
        save_dir = os.path.join(base_dir, run_name + '/')
        if os.path.exists(save_dir):
            raise Exception('Run number already exists!')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return run_num, save_dir

    @staticmethod
    def get_git_commits():
        git_commits = {}
        for repo in ['centaur', 'deephealth_utils', 'keras', 'keras_retinanet']:
            git_path = const.ROOT_PATH + repo

            git_prefix = 'git --git-dir={} --work-tree={}'.format(os.path.join(git_path, '.git'), git_path)

            subprocess.call('{} config core.fileMode false --global'.format(git_prefix), shell=True)

            commit = subprocess.check_output(
                '{} rev-parse --verify HEAD'.format(git_prefix), shell=True).decode(
                'utf-8').strip()

            branch = subprocess.check_output(
                '{} rev-parse --abbrev-ref HEAD'.format(git_prefix), shell=True).decode(
                'utf-8').strip()

            uncommitted = subprocess.check_output('{} diff --name-only HEAD'.format(git_prefix), shell=True).decode(
                'utf-8').strip()

            unpushed = subprocess.check_output(
                '{} log --branches={} --not --remotes'.format(git_prefix, branch), shell=True).decode(
                'utf-8').strip()

            if len(unpushed) > 1 or len(uncommitted) > 1:
                raise ValueError(
                    "Git repo {} has unpushed or uncommited changes:\nUncommited:\n{}\nUnpushed:\n{}\n" \
                    "Commit and push them before running Centaur.".format(repo, uncommitted, unpushed))


            git_commits[repo] = {
                'commit': commit,
                'branch': branch
            }
        return git_commits

    @staticmethod
    def load_run(run_num='', model_version=''):
        run_dir = os.path.join(const.DEPLOY_RUNS_PATH, 'run_{:04d}'.format(int(run_num)))
        if not os.path.exists(run_dir):
            raise IOError("Load run dir {} does not exist.".format(run_dir))

        # load the deploy config
        deploy_config = Config.from_json_file(os.path.join(run_dir, const.CENTAUR_CONFIG_JSON))

        # print out the exact deploy command to run
        with open(os.path.join(run_dir, 'deploy_command.txt'), 'r') as fp:
            python_command = fp.read()
        print("Python command used to run:\n{}".format(python_command))

        # check the python requirements.txt to make sure they match
        current_packages = subprocess.check_output('conda list -e', shell=True).decode('utf-8')
        with open(os.path.join(run_dir, 'conda_packages.txt'), 'r') as fp:
            run_packages = fp.read()
        if current_packages.strip() != run_packages.strip():
            raise ValueError("Conda packages do not match. Check {} for correct packages.".format(
                os.path.join(run_dir, 'conda_packages.txt')))

        # load the model configs saved
        # check the model hashes
        run_model_path = os.path.join(run_dir, 'models')
        centaur_model_path = os.path.join(const.MODEL_PATH, model_version)

        for root, dirs, files in os.walk(centaur_model_path):
            for name in files:
                rel_dir = os.path.relpath(root, centaur_model_path)
                run_path = os.path.join(run_model_path, rel_dir, name)
                source_path = os.path.join(centaur_model_path, rel_dir, name)
                if name.endswith('.json'):
                    # subprocess.call('cp {} {}'.format(source_path, dest_path), shell=True)
                    json_diff = subprocess.call('diff {} {}'.format(source_path, run_path), shell=True)
                    if json_diff != 0:
                        raise ValueError("Config file {} does not match.".format(source_path))
                elif name.endswith('.hdf5'):
                    sha1 = subprocess.check_output('sha1sum {}'.format(source_path), shell=True)
                    sha1 = sha1.decode('utf-8').strip().split(' ')[0]
                    with open(os.path.join(run_path + '.txt'), 'r') as fp:
                        run_sha1 = fp.read()
                    if run_sha1.strip() != sha1.strip():
                        raise ValueError("SHA1 hash for model weights {} does not match.".format(source_path))

        # check the git repo commits match

        git_commits = Run.get_git_commits()
        with open(os.path.join(run_dir, const.CENTAUR_COMMIT_FN), 'r') as fp:
            run_git_commits = json.load(fp)

        for repo, values in run_git_commits.items():
            if repo not in git_commits:
                raise ValueError("Repo {} not found in current state".format(repo))
            if git_commits[repo]['branch'] != values['branch']:
                raise ValueError("Branch {} in repo {} not found in current state".format(values['branch'], repo))
            if git_commits[repo]['commit'] != values['commit']:
                raise ValueError("Commit {} in repo {} not found in current state".format(values['commit'], repo))

        # notify GPU/CPU differences

        return deploy_config

    def save_run(self):
        # create a new run directory
        run_num, run_dir = self.get_next_run(const.DEPLOY_RUNS_PATH, 'run_')
        print("Saving run config in {}...".format(run_dir))

        # save the deploy config
        self.deploy_config.to_json_file(os.path.join(run_dir, const.CENTAUR_CONFIG_JSON))

        # save the deploy command
        with open(os.path.join(run_dir, const.CENTAUR_COMMAND_TXT), 'w') as fp:
            fp.write(self.python_cmd)

        # save the conda packages
        conda_fp = os.path.join(run_dir, 'conda_packages.txt')
        os.system('conda list -e > ' + conda_fp)

        # check that git repos are all checked in and pushed, and save commit hashes
        git_commits = Run.get_git_commits()

        with open(os.path.join(run_dir, const.CENTAUR_COMMIT_FN), 'w') as fp:
            json.dump(git_commits, fp)

        # save the model configs into models/
        # compute the model hashes and save
        run_model_path = os.path.join(run_dir, 'models')
        os.makedirs(run_model_path)
        model_version = self.deploy_config[Config.MODULE_ENGINE, 'model_version']
        centaur_model_path = os.path.join(const.MODEL_PATH, model_version)
        for root, dirs, files in os.walk(centaur_model_path):
            for name in files:
                rel_dir = os.path.relpath(root, centaur_model_path)
                os.makedirs(os.path.join(run_model_path, rel_dir), exist_ok=True)
                dest_path = os.path.join(run_model_path, rel_dir, name)
                source_path = os.path.join(root, name)
                if name.endswith('.json'):
                    subprocess.call('cp {} {}'.format(source_path, dest_path), shell=True)
                elif name.endswith('.hdf5'):
                    sha1 = subprocess.check_output('sha1sum {}'.format(source_path), shell=True)
                    sha1 = sha1.decode('utf-8').strip().split(' ')[0]
                    with open(os.path.join(dest_path + '.txt'), 'w') as fp:
                        fp.write(sha1)

        # save GPU/CPU stats
        cpu_fn = os.path.join(run_dir, 'cpu_info.txt')
        subprocess.call('cat /proc/cpuinfo > {}'.format(cpu_fn), shell=True)
        gpu_fn = os.path.join(run_dir, 'gpu_info.txt')
        subprocess.call(
            'nvidia-smi --query-gpu=name,driver_version,memory.free,memory.total,pstate,temperature.gpu'
            '--format=csv > {}'.format(
                gpu_fn), shell=True)

        print("Finished saving run config!")
        return run_num, run_dir