import json
import pdb
import subprocess
import os


def check_git_commits(states):
    for repo, values in states.items():
        if repo == 'centaur_build':
            continue
        print("Checking commit and branch of repo {}".format(repo))
        commit_id = values['commit']
        branch = values['branch']
        git_dir = os.path.join('/root', repo, '.git')
        current_commit = subprocess.check_output(
            'git --git-dir={} rev-parse --verify HEAD'.format(git_dir), shell=True).decode(
            'utf-8').strip()
        current_branch = subprocess.check_output(
            'git --git-dir={} rev-parse --abbrev-ref HEAD'.format(git_dir), shell=True).decode(
            'utf-8').strip()
        if len(current_commit) != 40:
            raise IOError("Git commit id is not 40 chars: {}".format(current_commit))
        if current_branch not in ['dev', 'master']:
            print("Warning: the current branch '{}' for repo {} is not dev or master".format(current_branch, repo))
        if branch != current_branch:
            raise IOError(
                "Git repo {} current has branch {}; does not match required branch {}".format(repo,
                                                                                              current_branch,
                                                                                              branch))
        if commit_id != current_commit:
            raise IOError(
                "Git repo {} current has commit {}; does not match required commit {}".format(repo,
                                                                                              current_commit,
                                                                                              commit_id))
    print("All Git repos match required commit ids")


def check_hashes(states):
    for path, sha1_sums in states.items():
        print("Checking sha1 hash for file {}".format(path))
        if not isinstance(sha1_sums, list):
            sha1_sums = [sha1_sums]
        current_sha1 = \
        subprocess.check_output('sha1sum {}'.format(path), shell=True).decode('utf-8').strip().split(' ')[0]
        # pdb.set_trace()
        if all([x != current_sha1 for x in sha1_sums]):
            raise IOError(
                "File {} has sha1sum {}; does not match required sha1sums {}".format(path, current_sha1, sha1_sums))
    print("All files match required sha1 sums")


if __name__ == '__main__':
    # open version.json
    with open("/root/version.json", "r") as read_file:
        version_json = json.load(read_file)

    check_git_commits(version_json['repos'])
    check_hashes(version_json['hashes'])

    print("All repos and files successfully passed tests.")
