import os
import sys
import subprocess
import pytest
import shutil
import psutil
import traceback
import tensorflow as tf
import re
from version_parser import Version

from tensorflow.python.client import device_lib

import centaur_deploy.constants as const_deploy
import centaur_test.constants as constants

@pytest.fixture(scope="function", autouse=True)
def override_xml_for_jenkins(record_xml_attribute):
    """
    Override the default 'classname' property in the JUnit XML description for a clearer visualization in Jenkins
    :param record_xml_attribute:
    :return:
    """
    record_xml_attribute("classname", "5_Unit")

def test_T_182():
    """
    Hardware/OS requirements check
    """
    ## HARDWARE
    assert sys.maxsize > 2 ** 32, "64 bits platforms are required"

    # Hard drive free storage in the input folder disk (100 GB)
    disk_usage = shutil.disk_usage(const_deploy.DEFAULT_INPUT_FOLDER)
    avail_gb = disk_usage.free / 2**30
    assert avail_gb >= constants.MIN_HARD_DRIVE_STORAGE_GB, \
        f"At least {constants.MIN_HARD_DRIVE_STORAGE_GB}GB are required. Currently available: {avail_gb}GB"

    # SYSTEM MEMORY
    mem_gb = psutil.virtual_memory().available / 2 ** 30
    assert mem_gb >= constants.MIN_MEMORY_GB, "{} GB of RAM memory are required"

    # GPU MEMORY
    try:
        gpus = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        for gpu in gpus:
            assert gpu.memory_limit / 2 ** 30 >= constants.MIN_MEMORY_GPU_GB, \
                "Found GPU with less than required free memory ({} GB). GPU info:\n{}".format(
                    constants.MIN_MEMORY_GPU_GB, gpu)
    except AssertionError:
        raise
    except:
        traceback.print_exc()
        pytest.fail("GPU/s memory could not be checked")

    ## SOFTWARE
    # Python version
    v = sys.version_info
    assert v.major == 3 and v.minor == 7, "Python 3.7 required"

    # GPU simple test
    assert tf.test.is_gpu_available(), "GPU not detected by tensorflow"

    # CUDA VERSION
    try:
        proc = subprocess.run("nvcc --version", capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", "Unexpected error when running 'nvcc --version': {}".format(proc.stderr)
        version_info = proc.stdout
        r = re.search("V\d{1,2}(.)*$", version_info, flags=re.MULTILINE)
        assert r is not None, "CUDA version not found. Full version info: {}".format(version_info)
        version = version_info[r.span()[0]+1:r.span()[1]]
        print("CUDA version: {}".format(version))
        assert Version(version) >= Version(constants.CUDA_MIN_VERSION), \
            "Expected CUDA>={}. Computed: {}. Full version info:\n{}". \
            format(constants.CUDA_MIN_VERSION, version, version_info)
    except AssertionError:
        raise
    except:
        traceback.print_exc()
        pytest.fail("CUDA version could not be determined")

    # NVIDIA DRIVERS
    try:
        proc = subprocess.run("cat /proc/driver/nvidia/version", capture_output=True, text=True, check=False, shell=True)
        assert proc.stderr == "", "Unexpected error when running 'cat /proc/driver/nvidia/version': {}".format(proc.stderr)
        version_info = proc.stdout.split('\n')[0]
        r = re.search(" \d{1,3}\.\d{1,3}(\.\d{1,3}){0,1} ", version_info, flags=re.MULTILINE)
        assert r is not None, "Nvidia version not found. Full version info: {}".format(version_info)
        v = version_info[r.span()[0]:r.span()[1]].strip()
        if len(v.split('.')) < 3:
            # NVIDIA drivers version usually contains only 2 components
            v += ".0"
        assert Version(v) >= Version(constants.NVIDIA_DRIVERS_MIN_VERSION), \
            "Expected nvidia drivers version>={}. Computed: {}. Full version info:\n{}". \
            format(constants.NVIDIA_DRIVERS_MIN_VERSION, v, version_info)
    except AssertionError:
        raise
    except:
        traceback.print_exc()
        pytest.fail("nvidia drivers version could not be determined")


def test_T_184():
    """
    Verify the system can write into the disk storage (output folder)
    """
    blank_file = os.path.join(const_deploy.DEFAULT_OUTPUT_FOLDER, "test_file")
    try:
        with open(blank_file, 'w') as f:
            f.write("Test")
    finally:
        if os.path.isfile(blank_file):
            os.remove(blank_file)


