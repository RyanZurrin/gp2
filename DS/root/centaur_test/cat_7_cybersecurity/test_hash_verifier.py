import os
import shutil
import subprocess, shlex
import json
import random
import tempfile
import string

def run_command_with_returncode(command, verbose=False):
    """ Run command using subprocess and get returncode.

    Args:
        command (str): command to run.
        verbose (bool): verbose. Defaults to False.

    Returns:
        str: return code from execution of command.

    """
    if verbose:
        command += " --verbose"
    p = subprocess.Popen(shlex.split(command))
    _, _ = p.communicate()
    return p.returncode

def get_path_in_error_log(log_filepath):
    """ Get path that have mismatched expected and observed hash values from error log file.

    Args:
        log_filepath (str): filepath for error log.

    Returns:
        str: path recorded in error log. For now only one line is expected to be written.

    """
    paths = []
    with open(log_filepath, "r") as f:
        for line in f:
            paths.append(line.split("\t")[1])
    return paths

def validate_json_format(filepath):
    """ Validate json file format.

    Args:
        filepath (str): json filepath to validate.

    Returns:
        bool: whether json file format is valid or not.

    """
    try:
        with open(filepath) as f:
            _ = json.load(f)
        is_valid = True
    except:
        is_valid = False
    return is_valid

def create_empty_file(filepath):
    """ Create an empty file with a given filepath.

    Args:
        filepath (str): path for empty file to be created.

    """
    with open(filepath, 'w') as _:
        pass

def create_files_dirs_to_hash(base_dir):
    """ Create files and directories to hash.

    Args:
        base_dir (str): base directory that will have new files and directories to be used.

    Returns:
        list: list of paths to files and directories to hash.

    """

    def make_file(filepath, value):
        with open(filepath, 'w') as f:
            f.write("{}\n".format(value))

    os.mkdir(base_dir)

    cnt = 0
    paths_to_hash = []

    # For testing hash generation of files
    filenames = ["file1.txt", "file2.log", "file3.py"]
    for filename in filenames:
        cnt += 1
        filepath = os.path.join(base_dir, filename)
        paths_to_hash.append(filepath)
        random_string = get_random_string(10, seed=cnt)
        make_file(filepath, random_string)

    # For testing hash generation of directories
    dirnames = {
        "dir1": ["dir1_file1.txt", "dir1_file2.log"],
        "dir2": ["dir2_file1.py", "dir2_file2.pkl", "dir2_file3.log"],
    }
    for dirname, filenames in dirnames.items():
        dirpath = os.path.join(base_dir, dirname)
        paths_to_hash.append(dirpath)
        os.mkdir(dirpath)
        for filename in filenames:
            cnt += 1
            filepath = os.path.join(dirpath, filename)
            random_string = get_random_string(10, seed=cnt)
            make_file(filepath, random_string)

    return paths_to_hash

def get_random_string(length, seed=None):
    """ Get random string of a given length.

    Args:
        length (int): length of random string.
        seed (int): seed for random choice. Defaults to None.

    Returns:
        str: a random string
    """

    letters = string.ascii_lowercase
    if seed is not None:
        random.seed(seed)
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def modify_file(paths_to_modify):
    """ Modify files or directories.

    Args:
        paths_to_modify (list): list of files and directories to modify.

    """

    for path_to_modify in paths_to_modify:
        if os.path.isdir(path_to_modify):
            filepath = os.path.join(path_to_modify, os.listdir(path_to_modify)[0])
        else:
            filepath = path_to_modify

        random_string = get_random_string(10, seed=777)
        with open(filepath, "a") as f:
            f.write("{}\n".format(random_string))

def test_T_201():
    """ Test whether hash json is created and verified correctly.

    Tests:
        1. Test whether it returns non-zero exit code if a given hash json is not present.
        2. Test whether it creates a new hash json and verifies it.
            a) create a new hash json.
            c) validate_json_format.
            b) verify the new hash json.
        3. Test whether it makes a backup copy of old hash json before overwriting.
            a) create a new hash json after making a backup copy of old one.
            c) validate_json_format.
            b) verify the new hash json.
        4. Test whether it detects mismatch when files or dirs are modified.
            a) modify files or directories intentionally.
            b.1) verify the modified hash json (without error log file).
            b.2) verify the modified hash json (with error log file).

        (Not in this code, but it can be executed outside a docker container)
        5. Need to run "verifier.py" outside a container to test docker image hash
            docker_image_id = "583526517545.dkr.ecr.us-east-1.amazonaws.com/centaur-rollup:1.2.0"
            To create hash json
                "python verifier.py -j {json_filepath} --create docker_image --docker_image_id {docker_image_id}"
            To verify hash json
                "python verifier.py -j {json_filepath} --docker_image_id {docker_image_id}"

        6. Need to run "verifier.py" outside a container to test sys_env.txt with "conda" command
            To create hash json
                "python verifier.py -j {json_filepath} --create {path_to_sys_env.txt}"
            To verify hash json
                "python verifier.py -j {json_filepath}"

    NOTEs:
        "conda" is not installed inside a container, so any commands containing "conda" cannot be tested inside a container
    """

    # make sure /root/centaur_support exists
    assert os.path.isdir("/root/centaur_support"), "/root/centaur_support should be mounted for this test"
    VERIFIER_PYTHON_PATH = "/root/centaur_support/scripts/hash_verifier/verifier.py"

    # initialize working files and directories
    temp_dir = tempfile.mkdtemp()
    json_filepath = f"{temp_dir}/sha.json"
    error_log_filepath = f"{temp_dir}/error.log"
    conda_versions_filepath = f"{temp_dir}/conda_versions.txt"
    # sys_env_filepath = f"{temp_dir}/sys_env.txt"
    hash_dir = f"{temp_dir}/hash_dir"
    json_backup_dir = f"{temp_dir}/sha_json_backup"

    # create an empty error log file as it assumed to be created by other process
    create_empty_file(error_log_filepath)

    # create a controlled list of files and directories to hash
    paths_to_hash = create_files_dirs_to_hash(hash_dir)
    paths_to_hash.append(conda_versions_filepath)
    # paths_to_hash.append(sys_env_filepath)
    str_paths_to_hash = " ".join(paths_to_hash)

    verbose = True

    try:
        print("--- 1. Test whether it returns non-zero exit code if a given hash json is not present")
        command = f"python {VERIFIER_PYTHON_PATH} -j {json_filepath}"
        returncode = run_command_with_returncode(command, verbose=verbose)
        assert returncode == 1, f"return code 1 should be returned (got return code {returncode})"
        print("\n\n")

        print("--- 2. Test whether it creates a new hash json and verifies it")

        print("a) create a hash json")
        command = f"python {VERIFIER_PYTHON_PATH} -j {json_filepath} --create {str_paths_to_hash}"
        returncode = run_command_with_returncode(command, verbose=verbose)
        assert returncode == 0, f"return code 0 should be returned (got return code {returncode})"
        # Ensure the result file has read-only permissions
        permissions = oct(os.stat(json_filepath).st_mode)[-3:]
        assert permissions == "440", f"Expected permissions 440 (read only) for file {json_filepath}. Got: {permissions}"

        print("b) validate json file format")
        validate_json_format(json_filepath)

        print("c) verify the hash json")
        command = f"python {VERIFIER_PYTHON_PATH} -j {json_filepath} -e {error_log_filepath}"
        returncode = run_command_with_returncode(command, verbose=verbose)
        assert returncode == 0, f"return code 0 should be returned (got return code {returncode})"
        print("\n\n")

        print("--- 3. Test whether it makes a backup copy of old hash json before overwriting")

        print("a) create a hash json after making a backup copy of old one")
        command = f"python {VERIFIER_PYTHON_PATH} -j {json_filepath} --create {str_paths_to_hash} -b {json_backup_dir}"
        returncode = run_command_with_returncode(command, verbose=verbose)
        assert returncode == 0, f"return code 0 should be returned (got return code {returncode})"

        print("b) validate json file format")
        validate_json_format(json_filepath)

        print("c) verify the hash json")
        command = f"python {VERIFIER_PYTHON_PATH} -j {json_filepath}"
        returncode = run_command_with_returncode(command, verbose=verbose)
        assert returncode == 0, f"return code 0 should be returned (got return code {returncode})"
        print("\n\n")

        print("--- 4. Test whether it detects mismatch when a file or dir is modified")

        print("a) modify a file or directory intentionally")
        paths_to_modify = [f"{hash_dir}/file1.txt", f"{hash_dir}/dir2"]
        modify_file(paths_to_modify)

        print("b.1) verify the modified hash json (without error log file)")
        command = f"python {VERIFIER_PYTHON_PATH} -j {json_filepath}"
        returncode = run_command_with_returncode(command, verbose=verbose)
        assert returncode == 1, f"return code 1 should be returned (got return code {returncode})"

        print("b.2) verify the modified hash json (with error log file)")
        command = f"python {VERIFIER_PYTHON_PATH} -j {json_filepath} -e {error_log_filepath}"
        returncode = run_command_with_returncode(command, verbose=verbose)
        assert returncode == 1, f"return code 1 should be returned (got return code {returncode})"
        mismatch_paths = get_path_in_error_log(error_log_filepath)
        assert set(paths_to_modify) == set(mismatch_paths), \
            f"path with a modified hash should be stored in error log (modified={paths_to_modify} in error log={mismatch_paths}"

    finally:
        shutil.rmtree(temp_dir)
