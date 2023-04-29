import tempfile
import json
import os
import shutil
import sys
import subprocess
import time

import pytest

START_DEFAULT_VALUES = {'save_to_ram': True, 'cadt_operating_point_key': 'balanced', 'remove_input_files': False}

def dict_to_json(data, filename):
    """
    given a dictionary, save the dictionary as a json file
    Args:
        data: python dictionary to save
        filename: json file to save the data

    Returns: None

    """
    # Use a customized format to have one line per setting in the same way we have in the regular
    # config.json file. Otherwise the start scripts won't work
    s = json.dumps(data)
    s = ",\n".join(s.split(','))
    s = s[:-1] + "\n}"
    with open(filename, 'w') as f:
        f.write(s)


def data_from_json(filename):
    """
    given a json file extract the data into python dictionary
    Args:
        filename: json file

    Returns: python dictionary that represents the data in the json file

    """

    with open(filename, 'rb') as f:
        data = json.load(f)
    return data


def read_txt_lines(filename):
    """
    read a txt file into lines
    Args:
        filename: txt file name

    Returns: list of str represents the txt of the file

    """
    with open(filename) as f:
        data = f.readlines()
    return data


def line_in_txt(desired, file_path):
    """
    checks if a desire str is in a txt file
    Args:
        desired: txt to look for
        file_path: path of txt in which to look for

    Returns: bool

    """
    lines = read_txt_lines(file_path)
    return any(desired in line for line in lines)


def make_start_up_dir(base_dir, centaur_client_dir, centaur_support_dir, docker_image):
    """
    creates all the folders and files needed to run start.sh

    DH_CLIENT_DIR/
        start.sh
        stop.sh
        config.json
    DH_SUPPORT_DIR/
        scripts
            hash_verification_script


    the two config.json are copies of the same file


    Args:
        base_dir (str): base directory where all the files will be made
        docker_image (str): image id used to create docker container.
        centaur_client_dir (str): CLIENT-DIR folder
        centaur_support_dir (str): directory of the centaur_support to run the test on
            scripts:
                start.sh
                stop.sh
            hash_verifier.py
                verifier.py

    Returns: None

    """

    # for the main config file that the start.sh will read
    config_dict = {
        "DOCKER_IMAGE": docker_image,
        'PACS_RECEIVE_PORT': 1,
        'PACS_RECEIVE_AE': 'test',
        'PACS_SEND_IP': 'test',
        'PACS_SEND_PORT': 1,
        'RIS_SEND_IP': 'test',
        'PACS_SEND_AE': 'test',
        'FACILITY_NAME': 'test'
    }

    # create dh_client_dir
    dh_client_dir = os.path.join(base_dir, 'dh_client_dir')
    config_dict["CLIENT_DIR"] = dh_client_dir
    os.mkdir(dh_client_dir)
    os.system('chmod 770 -R {}'.format(dh_client_dir))

    start_src = os.path.join(centaur_client_dir, "start.sh")
    stop_src = os.path.join(centaur_client_dir, "stop.sh")

    start_des = os.path.join(dh_client_dir, "start.sh")
    stop_des = os.path.join(dh_client_dir, "stop.sh")
    shutil.copyfile(start_src, start_des)
    shutil.copyfile(stop_src, stop_des)
    os.system(f"chmod 770 {start_des}")
    os.system(f"chmod 770 {stop_des}")

    mount_input = os.path.join(dh_client_dir, "input")
    mount_output = os.path.join(dh_client_dir, "output")
    os.mkdir(mount_input)
    os.mkdir(mount_output)
    config_dict["CLIENT_INPUT_DIR"] = mount_input
    config_dict["CLIENT_OUTPUT_DIR"] = mount_output

    client_config_dir = os.path.join(dh_client_dir, 'config.json')

    # create dh_support_dir
    dh_support_dir = os.path.join(base_dir, 'dh_support_dir')
    scripts_dir = os.path.join(dh_support_dir, "scripts")
    config_dict["SUPPORT_DIR"] = dh_support_dir
    os.mkdir(dh_support_dir)
    os.mkdir(scripts_dir)

    sha_ver_src = os.path.join(centaur_support_dir, "scripts/hash_verifier")
    sha_ver_des = os.path.join(dh_support_dir, "scripts/hash_verifier")
    shutil.copytree(sha_ver_src, sha_ver_des)
    os.system('chmod 770 -R {}'.format(dh_support_dir))

    execution_log_path = os.path.join(dh_support_dir, "execution_log.txt")
    sha_json_path = os.path.join(dh_support_dir, 'sha.json')
    sha_errors_path = os.path.join(dh_support_dir, 'sha_errors.txt')
    heartbeat_log_path = os.path.join(dh_support_dir, "heartbeat_log.txt")

    with open(execution_log_path, 'w') as fp:
        pass
    with open(sha_errors_path, 'w') as fp:
        pass
    with open(heartbeat_log_path, 'w') as fp:
        pass


    sys.path.append(scripts_dir)
    from hash_verifier.verifier import HashVerifier

    # create the hash.json needed to operate start.sh, since sha testing is not the main point
    # of the tests we will just check the sha of config.json
    hv = HashVerifier(sha_json_path, paths_to_hash=[start_des])
    hv.create()

    dict_to_json(config_dict, client_config_dir)


def get_container_name(working_dir):
    """
    get the name of the container created by reading the execution log
    Args:
        working_dir: directory where all testing files are located

    Returns: str if a container was launch otherwise return None

    """
    # first read the log and get the container id
    execution_log = os.path.join(working_dir, 'dh_support_dir', 'execution_log.txt')
    data = read_txt_lines(execution_log)
    assert len(data) <= 1, "expect 1 or 0 line in execution_log got {}".format(len(data))

    # no launching of container
    if len(data) == 0:
        return None

    # check the words of the log
    words = data[0].split()
    assert words[0] == 'START:', "expected the first word of the execution_log " \
                                 "to be START but gotten {} instead".format(words[0])
    return "SAIGE-Q-{}".format(words[2])


def clean_up(working_dir):
    """
    after testing is done clean up by stop containers and removed files that were created
    Args:
        working_dir: directory of testing files

    Returns: None

    """
    try:
        container_id = get_container_name(working_dir)

        # if container was launched clean the write protected files in the container
        # and stop the container
        if container_id is not None:
            # command = 'docker exec -d {} rm -r /root/client_dir/output'.format(container_id)
            #
            # val = subprocess.call(command, shell=True)
            # assert val == 0, 'unsuccessful deletion of write protected output folder'

            val = subprocess.call("docker stop {}".format(container_id), shell=True)
            assert val == 0, "stopping of docker container unsucessful"
    finally:
        # remove the temp working folder
        #shutil.rmtree(working_dir)#, onerror=set_rw)
        os.system(f"rm -rf {working_dir}")


def get_centaur_config(working_dir):
    """
    get the centaur_config that shows all centaur parameters
    Args:
        working_dir: directory of testing files

    Returns: python dictionary of centaur_config

    """
    container_name = get_container_name(working_dir)
    output_dir = os.path.join(working_dir, 'dh_client_dir', 'output', container_name)
    centaur_config = os.path.join(output_dir, 'centaur_config.json')
    centaur_config_dict = data_from_json(centaur_config)
    return centaur_config_dict

def helper(working_dir, centaur_client_dir, centaur_support_dir, docker_image, mode=None):
    """
    helper function that will execute the start.sh command and also set up folder etc.
    Args:
        working_dir (str):  directory of testing files
        centaur_client_dir (str): path to CLIENT-DIR folder
        centaur_support_dir (str): directory of centaur_support
        docker_image (str): Docker image on which to launch containers

    Returns: val the exit code of the start.sh command

    """
    assert os.path.exists(centaur_support_dir), "{} doesnt exist".format(centaur_support_dir)

    # set up folder and call the start.sh command
    make_start_up_dir(working_dir, centaur_client_dir, centaur_support_dir, docker_image)
    bash_dir = os.path.join(working_dir, 'dh_client_dir')
    if mode == 'invalid':
        val = subprocess.call(f"{bash_dir}/start.sh --invalid_arg arg_val", shell=True)
    elif mode =='non_default':
        val = subprocess.call(f"{bash_dir}/start.sh --operating_point high_spec --save_to_disk --remove_input_files", shell=True)
    else:
        val = subprocess.call(f"{bash_dir}/start.sh", shell=True)

    # give some time for the files to show up in the working_dir
    time.sleep(10)

    return val

@pytest.mark.external
def test_T_E_214_1(centaur_client_dir, centaur_support_dir, docker_image):
    """
    test the the parameter in client config is propagated into centaur
    Returns: None

    """
    working_dir = tempfile.mkdtemp() + '/'
    try:
        # run start.sh command
        val = helper(working_dir, centaur_client_dir, centaur_support_dir, docker_image)
        assert val == 0, 'start.sh did not execute correctly'
        # get config and check if these values are propagated
        config = get_centaur_config(working_dir)
        assert config['IO']['pacs_receive'], 'pacs_receive should be true'
        assert config['IO']['pacs_send'], 'pacs_receive should be true'
        test_keys = ['pacs_receive_port', 'pacs_receive_ae',
                     'pacs_send_ip', 'pacs_send_port',
                     'pacs_send_ae']

        for key in test_keys:
            if key in ('pacs_receive_port', 'pacs_send_port'):
                assert config['IO'][key] == 1, "{} was not updated based on client config.json"
            else:
                assert config['IO'][key] == 'test', "{} was not updated based on client config.json"
    finally:
        clean_up(working_dir)
        assert not os.path.exists(working_dir), "temp folder {} was not deleted sucessfully".format(working_dir)


@pytest.mark.external
def test_T_E_214_2(centaur_client_dir, centaur_support_dir, docker_image):
    """
    checks that invalid terminal arguments results in error
    Returns:

    """
    working_dir = tempfile.mkdtemp() + '/'

    try:
        val = helper(working_dir, centaur_client_dir, centaur_support_dir, docker_image, mode='invalid')
        # should be error or exit code 1
        assert val == 1, 'start.sh should exit with error if invalid arguments are passed in '

    finally:
        clean_up(working_dir)
        assert not os.path.exists(working_dir), "temp folder {} was not deleted successfully".format(working_dir)


@pytest.mark.external
def test_T_E_214_3(centaur_client_dir, centaur_support_dir, docker_image):
    """
    tests that if no arguments are passed in the default arguments are used
    Returns: None

    """
    working_dir = tempfile.mkdtemp() + '/'
    try:
        val = helper(working_dir, centaur_client_dir, centaur_support_dir, docker_image)
        assert val == 0, 'start.sh did not execute correctly'

        config = get_centaur_config(working_dir)

        assert config['ENGINE']['save_to_ram'] == START_DEFAULT_VALUES['save_to_ram'], \
            "if --save_to_disk is not passed in as argument then save_to_ram " \
            "should be {} in centaur_config".format(START_DEFAULT_VALUES['save_to_ram'])
        assert config['ENGINE']['cadt_operating_point_key'] == START_DEFAULT_VALUES['cadt_operating_point_key'], \
            "if --operating_point is not passed in as argument, then cadt_operating_point_key should be " \
            "{} ".format( START_DEFAULT_VALUES['cadt_operating_point_key'])

        assert config['IO']["remove_input_files"] == START_DEFAULT_VALUES['remove_input_files'], \
            "default value for remove_input_files Should be {} ".format( START_DEFAULT_VALUES['remove_input_files'])

    finally:
        clean_up(working_dir)
        assert not os.path.exists(working_dir), "temp folder {} was not deleted successfully".format(working_dir)


@pytest.mark.external
def test_T_E_214_4(centaur_client_dir, centaur_support_dir, docker_image):
    """
    test the non-default terminal arguments are propagated
    Returns:None

    """
    working_dir = tempfile.mkdtemp() + '/'
    try:
        val = helper(working_dir, centaur_client_dir, centaur_support_dir, docker_image, 'non_default')
        assert val == 0, 'start.sh did not execute correctly'

        config = get_centaur_config(working_dir)

        assert not config['ENGINE']['save_to_ram'], \
            "if --save_to_disk is passed in as argument then save_to_ram should be False in centaur_config"
        assert config['ENGINE']['cadt_operating_point_key'] == 'high_spec', \
            "terminal value for --operating_point is not reflected in centaur_config, expected {} got {}".format(
                'high_spec', config['ENGINE']['cadt_operating_point_key'])
        assert config['IO']['remove_input_files'], \
            "if --remove_input_files is passed in as argument then remove_input_files should be True in centaur_config"

    finally:
        clean_up(working_dir)
        assert not os.path.exists(working_dir), "temp folder {} was not deleted successfully".format(working_dir)
