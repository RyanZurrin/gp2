import os
import subprocess

import pytest
import pydicom

import centaur_deploy.constants as const_deploy
from centaur_deploy.deploys.study_deploy_results import StudyDeployResults
from centaur_io.output.ris_output import RISOutput
from centaur_reports.report import Report
from centaur.centaur_io.output.pacs_output import PACSOutput
from deephealth_utils.data import format_dicom_tag_str
from deephealth_utils.data.dicom_utils import compare_dicoms
from centaur_test.data_manager import DataManager


def send_hl7_message(listener_ip, listener_port, study=DataManager.STUDY_01_HOLOGIC):
    """
    Send HL7 results for a study to a RIS
    Args:
        listener_ip (str): RIS listener IP
        listener_port (str): RIS listener port
        study (str): name of the study
    """
    print(f"Ris params: IP={listener_ip},Port={listener_port}")

    dm = DataManager()
    dm.set_default_baseline_params()

    # Read baseline study
    results_file_path = os.path.join(dm.baseline_dir_centaur_output, study,
                                     const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
    assert os.path.isfile(results_file_path), f"HL7 baseline message not found in {results_file_path}"
    study_results = StudyDeployResults.from_json(results_file_path)
    # Get HL7 results path
    hl7_text_path = study_results.reports_generated[Report.CADT_HL7]['output']

    # Send it to the RIS listener
    ris_output = RISOutput(listener_ip, listener_port)
    results = ris_output.send_results(hl7_text_path)
    assert results == ris_output.RESULT_OK, f"Results not ok: {results}"

def get_dicom_tag_value(ds, tag):
    """ Get DICOM metadata for a given tag

    Args:
        ds (pydicom.dataset.FileDataset): dicom object
        tag (str): dicom tag to read metadata

    Returns:
        str: metadata

    """
    return str(getattr(ds, format_dicom_tag_str(tag), None))

def start_dicom_listener(input_dir, port, PREFFIX="+B", aetitle="DEEPHEALTH"):
    """ start DICOM listener.

    Args:
        input_dir (str): directory to receive DICOM
        aetitle (str): AE title
        port (int): port

    Returns:
        (process): process for storescp execution

    """
    config_file = os.path.join(os.path.dirname(__file__), "storescp.cfg")
    command = f"storescp -su {PREFFIX} -uf -od {input_dir} -xf {config_file} Centaur --aetitle {aetitle} {port}"
    listener = subprocess.Popen(command, shell=True)
    print(f"Running DICOM listener (pid={listener.pid})")
    return listener

def send_dicom(dicom_filepath, ip, port, aetitle):
    print(f"Sending DICOM {dicom_filepath} to a listener with ip={ip}, port={port}, and aetitle={aetitle}")
    pacs_output = PACSOutput(ip, port, aetitle, verbose=True)
    pacs_output.send_results(dicom_filepath)


def test_T_192(run_mode, dicom_listener_ip, dicom_listener_port):
    """ Test whether preview dicom image, and SR can be sent to a dicom listener (inside or outside the container)
    and verify the transfer.

    Prerequisites (only when running with an external container)
        Step 1. Mapping {LISTENER_DIR} to /mnt/listener that is accessible to Saige-Q when running a container
            docker run ... --network host -v {listener_dir}:/mnt/listener
        Step 2. Start DICOM listener
            In python,
                listener = start_dicom_listener(LISTENER_DIR, prefix=PREFIX, aetitle=AETITLE, port=PORT)
            In bash shell,
                storescp -su $PREFIX -uf -od $LISTENER_DIR --aetitle $AETITLE $PORT

    Steps:
        1. get filepath to preview dicom image
        2. send preview dicom image using 'dcmsend'.
        3. check whether it is received at the listener side.
        4. check whether preview images from sender and listener have no differences.

    """

    IP = dicom_listener_ip
    PORT = dicom_listener_port
    DICOM_RECEIVE_FOLDER = "/tmp/dicom_listener"
    PREFFIX = "+B"
    AETITLE = "DEEPHEALTH"
    file_preffix_preview = "SC" # The test is done with Dicom Reports, which are SC images
    file_preffix_sr = 'SRm'

    # Load baseline CADT PREVIEW DICOM image
    dm = DataManager()
    dm.set_baseline_params(run_mode)
    base_dir = dm.baseline_dir_centaur_output

    studies_suspicious = dm.get_suspicious_studies(run_mode=run_mode)
    studies = dm.get_valid_studies()


    # Input folder will need to be mounted if using an external dicom listener
    is_local_dicom_listener = dicom_listener_ip in ("0.0.0.0", "127.0.0.1", "localhost")
    if is_local_dicom_listener:
        # Start an internal DICOM listener
        if not os.path.isdir(DICOM_RECEIVE_FOLDER):
            os.makedirs(DICOM_RECEIVE_FOLDER)
        start_dicom_listener(DICOM_RECEIVE_FOLDER, dicom_listener_port, PREFFIX=PREFFIX)
    else:
        # Make sure that the folder is mounted externally
        assert os.path.isdir(DICOM_RECEIVE_FOLDER), f"Folder {DICOM_RECEIVE_FOLDER} needs to be mounted externally"

    for study in studies:
        print(f"\n--- Testing with {study}...")

        print("Get filepaths")
        baseline_results_path = os.path.join(base_dir, study,
                                             const_deploy.CENTAUR_STUDY_DEPLOY_RESULTS_JSON)
        baseline_result = StudyDeployResults.from_json(baseline_results_path)
        sr_path = baseline_result.reports_generated[Report.SR]["output_file_path"]
        if run_mode == const_deploy.RUN_MODE_CADT:
            if study in studies_suspicious:
                preview_dicom_filepath = baseline_result.reports_generated[Report.CADT_PREVIEW]["dcm_output"]
            else:
                preview_dicom_filepath = None
        elif run_mode == const_deploy.RUN_MODE_CADX:
            preview_dicom_filepath = baseline_result.reports_generated[Report.PDF]["dcm_output"]
        else:
            raise ValueError(f"run_mode not supported: {run_mode}")

        print("Send preview dicom image...")
        if preview_dicom_filepath is not None:
            assert os.path.isfile(preview_dicom_filepath), f"{preview_dicom_filepath} does not exist"
            send_dicom(preview_dicom_filepath, IP, PORT, AETITLE)


        print("Send Structured Report")
        assert os.path.isfile(sr_path), f"{sr_path} does not exist"
        send_dicom(sr_path, IP, PORT, AETITLE)


        print("Check whether preview dicom image is received at the listener side...")
        for file_path, file_preffix in zip([preview_dicom_filepath, sr_path], [file_preffix_preview,file_preffix_sr]):
            if file_path is None:
                continue
            ds_sent = pydicom.dcmread(file_path)
            study_uid = get_dicom_tag_value(ds_sent, "StudyInstanceUID")
            sop_uid = get_dicom_tag_value(ds_sent, "SOPInstanceUID")

            dicom_filepath_received = f"{DICOM_RECEIVE_FOLDER}/{PREFFIX}_{study_uid}/{file_preffix}.{sop_uid}"
            assert os.path.isfile(dicom_filepath_received), f"{dicom_filepath_received} not received at the listener side"

            # 3. check whether preview images from sender and listener have no differences.
            ds_received = pydicom.dcmread(dicom_filepath_received)
            diffs = compare_dicoms(ds_sent, ds_received)
            assert len(diffs) == 0, f"{preview_dicom_filepath} and {dicom_filepath_received} have differences"


def test_T_193(ris_local_listener):
    """
    Send an HL7 output message to an internal RIS listener. It just checks that the message was sent without errors
    """
    # Read baseline HL7
    send_hl7_message('0.0.0.0', ris_local_listener.port)


@pytest.mark.external
def test_T_193_2(ris_external_listener_ip, ris_external_listener_port):
    """
    Send an HL7 output message to an external RIS listener. It just checks that the message was sent without errors
    This test requires than an external RIS server is listening for messages in a port (it could even be a different
    machine). There is a testing RIS listener that can be started running:
    python centaur_support/scripts/ris/listener.py --port [30000]
    (Please note that to start the listener, hl7apy pip package is required)
    Args:
        ris_external_listener_ip (str) (fixture): IP address for the external RIS listener.
                                                  Note: it could be 0.0.0.0 when the docker is run with --network=host
        ris_external_listener_port (int) (fixture): Port for the external RIS listener.
    """
    send_hl7_message(ris_external_listener_ip, ris_external_listener_port)

if __name__ == "__main__":
    #test_T_192("CADx", "0.0.0.0", 19999)
    test_T_193_2("0.0.0.0", 30000)
    print("DONE!!")
