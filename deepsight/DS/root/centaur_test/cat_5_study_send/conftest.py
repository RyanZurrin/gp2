import pytest
from centaur_support.scripts.ris.listener import RISListener


def pytest_addoption(parser):
    """
    Command line parameters
    Args:
        parser:

    Returns:

    """
    parser.addoption("--ris_local_listener_port", type=int, default=30001,
                     help="Port where a local RIS listener will be listening")
    parser.addoption("--ris_external_listener_ip", type=str, default="0.0.0.0",
                     help="IP address where a external RIS listener will be listening")
    parser.addoption("--ris_external_listener_port", type=int, default=20002,
                     help="Port where an external RIS listener will be listening")
    parser.addoption("--dicom_listener_ip", type=str, default="0.0.0.0",
                     help="IP where a DICOM listener will be listening")
    parser.addoption("--dicom_listener_port", type=int, default=19999,
                     help="Port where a DICOM listener will be listening")

@pytest.fixture(scope="session")
def ris_local_listener_port(request):
    """
    Port where a local RIS listener will be started
    Args:
        request:

    Returns:
        int
    """
    return request.config.getoption("--ris_local_listener_port")


@pytest.fixture(scope="session")
def ris_external_listener_port(request):
    """
    Port where an RIS listener (outside the docker container) will be listenign
    Args:
        request:

    Returns:
        int
    """
    return request.config.getoption("--ris_external_listener_port")


@pytest.fixture(scope="session")
def ris_external_listener_ip(request):
    """
    IP where an external RIS listener will be listening
    Args:
        request:

    Returns:
        str
    """
    return request.config.getoption("--ris_external_listener_ip")

@pytest.fixture(scope="session")
def dicom_listener_ip(request):
    """
    IP where an external RIS listener will be listening
    Args:
        request:

    Returns:
        str
    """
    return request.config.getoption("--dicom_listener_ip")

@pytest.fixture(scope="session")
def dicom_listener_port(request):
    """
    IP where an external RIS listener will be listening
    Args:
        request:

    Returns:
        str
    """
    return request.config.getoption("--dicom_listener_port")


@pytest.fixture(scope="function")
def ris_local_listener(request):
    port = request.config.getoption("--ris_local_listener_port")
    print(f"STARTING SERVER IN {port}")
    ris_listener = RISListener(port)
    ris_listener.start_listener()
    return ris_listener