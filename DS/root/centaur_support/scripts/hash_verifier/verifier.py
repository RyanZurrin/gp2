import os, sys
import datetime
import shutil
import subprocess, shlex
import hashlib
import tempfile
import json
import re
from argparse import ArgumentParser

class HashVerifier:
    """This is the class for hash verification and creation

    Args:
        json_file (str): filepath to hash json
        error_log_file (str): filepath to error log for paths with mismatch hash values
        json_backup_dir (str): path to directory to save backup copy of existing hash json file before overwriting
        paths_to_hash (str): list of paths of files and dirs to hash
        docker_image_id (str): id of docker image to hash
        verbose (bool): verbose

    """

    def __init__(self, json_file, error_log_file=None, json_backup_dir=None, paths_to_hash=None, docker_image_id=None, verbose=False):
        self.json_file = json_file
        self.error_log_file = error_log_file
        self.json_backup_dir = json_backup_dir
        self.paths_to_hash = paths_to_hash
        self.docker_image_id = docker_image_id
        self.verbose = verbose

    def run(self):
        """ run "create" if paths to files to hash are given, if not run "verify"
        """

        # verify hash json
        if self.paths_to_hash is None:
            if self.verbose:
                print(f"Verify hash json {self.json_file}")
            self.verify()

        # create hash json
        else:
            if self.verbose:
                print(f"Create a hash json {self.json_file} for a list of files and directories {self.paths_to_hash}")
            self.create()

    def create(self):
        """ create hash json for paths given in self.paths_to_hash
            if self.json_file exists, its backup copy is made before overwriting.
        """

        # generate hash dict
        hash_dict = {}
        for path in self.paths_to_hash:
            sha256 = self.generate_sha256(path, create=True)
            hash_dict[path] = sha256

        # make a backup copy of hash json file if it exists
        if os.path.exists(self.json_file):

            # create hash json backup directory if it doesn't exist
            assert self.json_backup_dir is not None, \
                f"{self.json_file} already exists but json_backup_dir has not been specified"
            if not os.path.exists(self.json_backup_dir):
                os.makedirs(self.json_backup_dir)

            # make filename of the backup json with a timestamp
            json_filename = self.json_file.split("/")[-1].split(".json")[0]
            timestamp = datetime.datetime.utcnow()
            backup_json_file = "{}/{}_{}.json".format(\
                    self.json_backup_dir, json_filename, timestamp.strftime('%Y-%m-%d-%H-%M'))
            shutil.copyfile(self.json_file, backup_json_file)

            if self.verbose:
                print(f"{self.json_file} backup copy to {backup_json_file}")

        # write to json
        with open(self.json_file, 'w') as f:
            json.dump(hash_dict, f)

        # Set read-only permissions
        os.chmod(self.json_file, 0o440)

    def verify(self):
        """ verify hash json given as self.json_file
            if there are paths with mismatch hash values,
                messages are written out to error log file when error log file is provided
                messages are printed out to stdout when error log file is not provided
        """

        # check whether json file exists
        if not os.path.isfile(self.json_file):
            print(f"json file not found ({self.json_file})")
            sys.exit(1)

        # read json to dict
        with open(self.json_file, 'r') as f:
            hash_dict = json.load(f)

        # verify hash
        timestamp_str_format = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')
        hash_mismatch_paths = {}
        for path, expected_hash in hash_dict.items():
            observed_hash = self.generate_sha256(path)
            if expected_hash != observed_hash:
                hash_mismatch_paths[path] = [expected_hash, observed_hash]

        # check whether there are hash mismatches and write out to error log if present
        if len(hash_mismatch_paths) != 0:
            print("Paths that have mismatched hash values are found")

            if self.error_log_file is not None:

                # check whether error log file exists
                if not os.path.isfile(self.error_log_file):
                    print(f"error log file not found ({self.error_log_file})")
                    sys.exit(1)

                # append lines for path with mismatch hash
                with open(self.error_log_file, "a") as f:
                    for path, [expected_hash, observed_hash] in hash_mismatch_paths.items():
                        f.write(f"{timestamp_str_format}\t{path}\t{expected_hash}\t{observed_hash}\n")
            else:
                for path, [expected_hash, observed_hash] in hash_mismatch_paths.items():
                    print(f"Code signature verification mismatch: {path}, "
                          f"expected signature: {expected_hash}, observed_signature: {observed_hash}")

            # non-zero return code
            sys.exit(1)

    def generate_sha256(self, path, create=False):
        """ Generate sha256 of path

        Args:
            path (str): path to a file or a directory to hash
            create (bool): whether create mode or not

        Returns:
            str: SHA256 hash of the path

        """
        if os.path.isdir(path):
            sha256 = self.generate_sha256_dir(path)
        else:
            if "docker_image" in path:
                assert self.docker_image_id is not None, "Docker image id should be assigned for generation of sha256"
                sha256 = self.generate_sha256_docker_image(self.docker_image_id)
            elif "conda_versions.txt" in path:
                path = path if create else None
                sha256 = self.generate_sha256_conda_versions(path)
            elif "sys_env.txt" in path:
                path = path if create else None
                sha256 = self.generate_sha256_sys_env(path)
            else:
                sha256 = self.generate_sha256_file(path)
        return sha256

    @classmethod
    def generate_sha256_docker_image(cls, docker_image_id):
        """
        Generate a SHA256 hash of a docker image
        :param docker_image_id: id of a docker image
        :return: str. SHA256 hash of a docker image
        """

        # get docker inspect information
        command = f"docker inspect --format='{{{{index .Id }}}}' {docker_image_id}"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        if "Got permission denied" in proc.stderr:
            command = f"sudo {command}"
            proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"
        sha256 = proc.stdout.strip().split("sha256:")[1]

        return sha256

    @classmethod
    def generate_sha256_conda_versions(cls, path):
        """ Generate a SHA256 hash of conda_versions.txt

        Args:
            path (str): path to conda_versions.txt
                if None, it will create a temporary conda_versions.txt and generate its SHA256 hash
                otherwises, it will create conda_versions.txt to 'path' and generate its SHA256 hash

        Returns:
            str: SHA256 hash

        """

        # use temp file if path is not assigned
        if path is None:
            temp_dir = tempfile.mkdtemp()
            path = f"{temp_dir}/conda_versions.txt"
        else:
            temp_dir = None

        # create conda_versions.txt
        cls.create_conda_versions_txt(path)

        # generate sha256
        sha256 = cls.generate_sha256_file(path)

        # delete temp file
        if temp_dir is not None:
            shutil.rmtree(temp_dir)

        return sha256

    @classmethod
    def generate_sha256_sys_env(cls, path):
        """ Generate a SHA256 hash of sys_env.txt

        Args:
            path (str): path to sys_env.txt
                if None, it will create a temporary sys_env.txt and generate its SHA256 hash
                otherwises, it will create sys_env.txt to 'path' and generate its SHA256 hash

        Returns:
            str: SHA256 hash
        """

        # use temp file if path is not assigned
        if path is None:
            temp_dir = tempfile.mkdtemp()
            path = f"{temp_dir}/sys_env.txt"
        else:
            temp_dir = None

        # create conda_versions.txt
        cls.create_sys_env_txt(path)

        # generate sha256
        sha256 = cls.generate_sha256_file(path)

        # delete temp file
        if temp_dir is not None:
            shutil.rmtree(temp_dir)

        return sha256

    @classmethod
    def create_conda_versions_txt(cls, path):
        """ Create "conda_versions.txt"
        Note: use 'pip list' instead of 'conda list' because there seems to be a bug with setup tools (sometimes the
        version displayed is different)

        Args:
            path (str): path to "conda_versions.txt"
        """
        command = f"pip list > {path}"
        return_code = os.system(command)
        assert return_code == 0 and os.path.isfile(path), f"Unexpected error when running {command}"
        # Set read only permissions
        os.chmod(path, 0o440)

    @classmethod
    def create_sys_env_txt(cls, path):
        """ Create sys_env.txt

        Args:
            path (str): path to "sys_env.txt"

        """
        with open(path, "w") as f:
            # OS version
            os_ver = cls.get_os_ver()
            f.write(f"OS_VERSION: {os_ver}\n")

            # CUDA version
            cuda_ver = cls.get_cuda_ver()
            f.write(f"CUDA_VERSION: {cuda_ver}\n")

            # NVIDIA DRIVERS version
            nvidia_driver_ver = cls.get_nvidia_driver_ver()
            f.write(f"NVIDIA_DRIVER_VERSION: {nvidia_driver_ver}\n")

            # DOCKER version
            docker_ver = cls.get_docker_ver()
            f.write(f"DOCKER_VERSION: {docker_ver}\n")

            # PYTHON version
            python_ver = cls.get_python_ver()
            f.write(f"PYTHON_VERSION: {python_ver}\n")

            # MINICONDA version
            miniconda_ver = cls.get_miniconda_ver()
            f.write(f"MINICONDA_VERSION: {miniconda_ver}\n")

            # NVIDIA DOCKER version
            nvidia_docker_ver = cls.get_nvidia_docker_ver()
            f.write(f"NVIDIA_DOCKER_VERSION: {nvidia_docker_ver}\n")

        # For security reasons, set read-only permissions
        os.chmod(path, 0o440)

    @classmethod
    def get_cuda_ver(cls):
        """ Get CUDA version

        Returns:
            str: CUDA version
        """
        # get version information
        command = "nvcc --version"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"
        version_info = proc.stdout

        # extract version
        r = re.search("\d{1,3}\.\d{1,3}\.\d{1,3}", version_info, flags=re.MULTILINE)
        assert r is not None, "Nvidia version not found. Full version info: {}".format(version_info)
        version = version_info[r.span()[0]:r.span()[1]]

        return version

    @classmethod
    def get_os_ver(cls):
        """ Get OS version

        Returns:
            str: OS version
        """
        # get version information
        command = "cat /etc/os-release"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"
        version_info = proc.stdout

        # extract version
        r = re.search('PRETTY_NAME="(.*?)"', version_info, flags=re.MULTILINE)
        assert r is not None, "OS version not found. Full version info: {}".format(version_info)
        version = version_info[r.span()[0]:r.span()[1]].split('\"')[1]

        return version

    @classmethod
    def get_nvidia_driver_ver(cls):
        """ Get NVIDIA driver version

        Returns:
            str: NVIDIA driver version
        """
        # get version information
        command = "cat /proc/driver/nvidia/version"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"
        version_info = proc.stdout.split('\n')[0]

        # extract version
        r = re.search(" \d{1,3}\.\d{1,3}(\.\d{1,3}){0,1} ", version_info, flags=re.MULTILINE)
        assert r is not None, "Nvidia Driver version not found. Full version info: {}".format(version_info)
        version = version_info[r.span()[0]:r.span()[1]].strip()

        return version

    @classmethod
    def get_docker_ver(cls):
        """ Get docker version

        Returns:
            str: docker version
        """

        # get version information
        command = "docker version --format '{{.Client.Version}}'"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"

        # extract version
        version = proc.stdout.strip()

        return version

    @classmethod
    def get_python_ver(cls):
        """ Get python version

        Returns:
            str: python version
        """

        # get version information
        command = "python --version"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"
        version_info = proc.stdout

        # extract version
        r = re.search("\d{1,3}\.\d{1,3}\.\d{1,3}", version_info, flags=re.MULTILINE)
        assert r is not None, "Python version not found. Full version info: {}".format(version_info)
        version = version_info[r.span()[0]:r.span()[1]]

        return version

    @classmethod
    def get_miniconda_ver(cls):
        """ Get miniconda version

        Returns:
            str: miniconda version
        """

        # get version information
        command = "conda --version"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"
        version_info = proc.stdout

        # extract version
        r = re.search("\d{1,3}\.\d{1,3}\.\d{1,3}", version_info, flags=re.MULTILINE)
        assert r is not None, "Miniconda version not found. Full version info: {}".format(version_info)
        version = version_info[r.span()[0]:r.span()[1]]

        return version

    @classmethod
    def get_nvidia_docker_ver(cls):
        """ Get nvidia docker version

        Returns:
            str: nvidia docker version
        """

        # get version information
        command = "nvidia-docker version --format '{{.Client.Version}}'"
        proc = subprocess.run(command, capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", f"Unexpected error when running '{command}': {proc.stderr}"
        version_info = proc.stdout

        # extract version
        r = re.search("NVIDIA Docker: (.*)", version_info, flags=re.MULTILINE)
        assert r is not None, "NVIDIA Docker version not found. Full version info: {}".format(version_info)
        version = version_info[r.span()[0]:r.span()[1]].split(":")[1].strip()

        return version

    @classmethod
    def generate_sha256_str(cls, str_to_hash):
        """ Generate a SHA256 hash of a string

        Args:
            str_to_hash (str): a string to hash

        Returns:
            str: SHA256 hash
        """
        return hashlib.sha256(str_to_hash.encode('utf-8')).hexdigest()

    @classmethod
    def generate_sha256_file(cls, filepath_to_hash):
        """ Generate a SHA256 hash of a file

        Args:
            filepath_to_hash (str): path to a file to hash

        Returns:
            str: SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(filepath_to_hash, 'rb') as f:
            sha256.update(f.read())
            return sha256.hexdigest()

    @classmethod
    def generate_sha256_dir(cls, dirpath_to_hash):
        """ Generate a SHA256 hash of a directory

        Args:
            dirpath_to_hash (str): path to a directory to hash

        Steps:
            1. List all files in alphabetical order in the directory.
            2. Initialize an empty string to serve as a master hash variable.
            3. Compute the sha256 hash of each file in the directory using the hashlib library and append to the master hash string.
            4. Compute the sha256 hash of the master hash string.

        Returns:
            str: SHA256 hash
        """
        # Get paths to all files in dirpath_to_hash
        filepaths = []
        for root, dirs, files in os.walk(dirpath_to_hash):
            for file in files:
                filepaths.append(os.path.join(root, file))

        # Sort filepaths alphabetically in ascending order
        filepaths = sorted(filepaths)

        # Concatenate a list of SHA256 hashes from all files in dirpath_to_hash
        full_sha256 = ""
        for filepath in filepaths:
            full_sha256 += cls.generate_sha256_file(filepath)

        # Compute final sha256 hash from the concatenated one
        final_sha256 = cls.generate_sha256_str(full_sha256)

        return final_sha256

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("--json_file", "-j", type=str, help="Path to a json file that has SHA hash", required=True)
    args_parser.add_argument("--error_log_file", "-e", type=str, help="Path to an error log file")
    args_parser.add_argument("--create", nargs="+", type=str, help="list of paths of files and dirs to generate ssh (eg. file1 dir1 file2)")
    args_parser.add_argument("--json_backup_dir", "-b", type=str, help="Path to directory to backup prior json file")
    args_parser.add_argument("--docker_image_id", "-d", type=str, help="docker image id")
    args_parser.add_argument('--verbose', default=False, action='store_true')
    args = args_parser.parse_args()

    # run
    hash_verifier = HashVerifier(args.json_file, args.error_log_file,
                        json_backup_dir=args.json_backup_dir, paths_to_hash=args.create,
                        docker_image_id=args.docker_image_id, verbose=args.verbose)
    hash_verifier.run()