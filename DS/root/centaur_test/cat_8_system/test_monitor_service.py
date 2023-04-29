import os
import time
import shutil
import subprocess, shlex
import tempfile

import pytest

from centaur.centaur_deploy.heartbeat import HeartBeat

EXECUTION_ID = "dhmt001"
TIMEOUT = 15
MONITOR_STARTUP_PERIOD = 5
MONITOR_PERIOD = 2
HEARTBEAT_PERIOD = 2

def write_initial_execution_log(filepath):
    with open(filepath, "w") as f:
        f.write("Start DH-MT...\n")
        f.write("Getting a study...\n")
        f.write("Preprocessing...\n")
        f.write("Deploying Centaur on the study...\n")
        f.write("Postprocessing...\n")
        f.write("Results\n")

def read_heartbeat_log(filepath):
    with open(filepath, 'r') as f:
        timestamp = f.read().strip()
    return timestamp

def check_if_CRASH_in_execution_log(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return any("CRASH:" in line for line in lines)

def test_T_189():
    """
    Test whether monitoring service detects unexpected heartbeat crash
    Setup:
        1. Write a fake execution log (will be written by startup script)
        2. Run the heartbeat as a thread (will be executed inside a docker container)
            a) wait for the period of heartbeat signal to make sure heartbeat starts
        3. Start monitoring service as a process
    Test:
        1. Test whether HeartBeat keep updating timestamp
            a) wait for three heartbeat periods and make sure updated timestamps are different from each other
        2. Check whether there is no CRASH in execution log
            a) "CRASH:" should not be present in execution log as heartbeat keeps working
        3. Check whether monitoring service can detect CRASH from heartbeat log
            a) terminate heartbeat
            b) wait for more than TIMEOUT of monitoring service and check whether CRASH is detected by the service
    """

    # Make sure /root/centaur_support exists
    assert os.path.isdir("/root/centaur_support"), "/root/centaur_support should be mounted for this test"

    # output dir
    # output dir that contains execution log (outside docker container)
    output_dir = tempfile.mkdtemp()
    execution_log_path = "{}/execution.log".format(output_dir)
    heartbeat_log_path = "{}/heartbeat.log".format(output_dir)
    try:
        # Write initial execution log (will be written by startup script outside docker container)
        write_initial_execution_log(execution_log_path)

        # Start heartbeat as a thread
        heartbeat_th = HeartBeat(heartbeat_log_path, HEARTBEAT_PERIOD)
        heartbeat_th.start()

        # wait for seconds to let heartbeat update
        time.sleep(HEARTBEAT_PERIOD)

        # start monitoring service in background
        # NOTE: service.py is placed outside the container, but for this test, centaur_support is mounted on this container
        service_python_path = "/root/centaur_support/scripts/monitoring_service/service.py"
        command = "python {} -hf {} -ef {} -id {} -t {} -fr {} -s {}".format(service_python_path, \
                                                                             heartbeat_log_path, execution_log_path, \
                                                                             EXECUTION_ID, TIMEOUT, \
                                                                             MONITOR_PERIOD, MONITOR_STARTUP_PERIOD)
        p = subprocess.Popen(shlex.split(command))

        # 1. Test whether HeartBeat keep updating timestamp
        heartbeat_times = []
        for i in range(3):
            time.sleep(HEARTBEAT_PERIOD)
            heartbeat_times.append(read_heartbeat_log(heartbeat_log_path))
        assert(len(heartbeat_times) == len(set(heartbeat_times)))

        # 2. Check whether there is no CRASH in execution log
        for i in range(3):
            time.sleep(HEARTBEAT_PERIOD)
            assert(not check_if_CRASH_in_execution_log(execution_log_path))

        # 3. Check whether monitoring service can detect CRASH from heartbeat log
        # "stop()" method does not kill heartbeat thread (heartbeat_th.is_alive() = True)
        # But, it stops its running to write heartbeat log so that crash detection can be tested.
        heartbeat_th.stop()
        time.sleep(TIMEOUT+2*MONITOR_PERIOD)
        assert(check_if_CRASH_in_execution_log(execution_log_path))

    finally:
        p.terminate()
        shutil.rmtree(output_dir)
