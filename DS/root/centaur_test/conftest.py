import pytest
import centaur_deploy.constants as const_deploy

collect_ignore = ["hello_world.py", "utils.py", "basic_pipeline.py", "cat_3_study_run/manual_integration_tests.py",
                  "cat_3_study_run/I2D.py", "cat_0/unit_tests.py"]


def pytest_addoption(parser):
    parser.addoption("--run_mode", help="Run mode (CADx, CADt, etc.)")

    # Parameters for external tests
    default_client_dir = "/deephealth/saige-q"
    default_support_dir = "/deephealth/saige-q_support"
    default_docker_image = 'centaur-tests'
    parser.addoption("--centaur_client_dir", default=default_client_dir, help="Path to the centaur client_dir repo")
    parser.addoption("--centaur_support_dir", default=default_support_dir, help="Path to the centaur_support repo")
    parser.addoption("--docker_image", default=default_docker_image, help="Docker image id")
    parser.addoption("--intermediate_outputs", default="/root/dh_test_results/intermediate_outputs",
                     help="Folder to save intermediate results (for manual tests, etc.).")

@pytest.fixture(scope="session")
def run_mode(request):
    run_mode = request.config.getoption("--run_mode")
    assert run_mode in (const_deploy.RUN_MODE_CADT, const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_DEMO)


#region External tests

@pytest.fixture(scope="session")
def centaur_client_dir(request):
    """
    CLIENT-DIR directory
    :param request: request object
    :return: str path of the centaur_support_dir
    """
    return request.config.getoption("--centaur_client_dir")

@pytest.fixture(scope="session")
def centaur_support_dir(request):
    """
    centaur_support directory
    :param request: request object
    :return: str path of the centaur_support_dir
    """
    return request.config.getoption("--centaur_support_dir")

@pytest.fixture(scope="session")
def docker_image(request):
    """
    centaur_support directory
    :param request: request object
    :return: str path of the centaur_support_dir
    """
    return request.config.getoption("--docker_image")

@pytest.fixture(scope="session")
def intermediate_outputs_folder(request):
    """
    Intermediate outputs folder (for manual tests, etc).
    Default: "/root/temp_outputs"
    Args:
        request:

    Returns:
        str
    """
    return request.config.getoption("--intermediate_outputs")

# endregion

# def get_actual_model_version(model_version):
#     """
#     Model version being used for all the tests.
#     It will be specified via command line with the "--model" param.
#     If model=='current', the actual model version will be searched
#     :param model_version: str. Model version or 'current'
#     :return: str. Real model version
#     """
#     if model_version == "current":
#         # Get the real model version
#         from centaur_engine.model import ModelSelector
#         model = ModelSelector.select(model_version)
#         model_version = model.get_version()
#     return model_version

# @pytest.fixture(scope="session")
# def centaur_model(request):
#     """
#     Instance of the Centaur model
#     :param request: pytest object
#     :return: str. Dataset where the tests will be run
#     """
#     model_version = request.config.getoption("--model")
#     return _get_model_(model_version)

# def _get_model_(model_version):
#     from centaur_engine.model import ModelSelector
#     return ModelSelector.select(model_version)


def pytest_generate_tests(metafunc):
    """
    Fixture objects that can be parametrized via command line parameters
    :param metafunc: pytest object
    """
    mode = metafunc.config.getoption('run_mode')
    allowed_modes = ('CADt', 'CADx')
    if mode is None:
        modes = allowed_modes
    else:
        assert mode in allowed_modes, "Run mode not allowed ({}). Allowed modes: {}".format(mode, allowed_modes)
        modes = [mode]
    if 'run_mode' in metafunc.fixturenames:
        metafunc.parametrize("run_mode", modes)

#     import centaur_engine.helpers.helper_model as helper_model
#     if 'dataset_name' in metafunc.fixturenames:
#         # datasets = metafunc.config.getoption('datasets').split(',')
#         # metafunc.parametrize("dataset_name", datasets)
#         dataset = metafunc.config.getoption('dataset')
#         metafunc.parametrize("dataset_name", [dataset])
    # if 'centaur_model_version' in metafunc.fixturenames:
    #     # Replace the string by the real model version if model=='current'.
    #     # Only one model will be allowed for performance reasons (only one global 'centaur_model' will be loaded)
    #     model_version = metafunc.config.getoption('model')
    #     real_model_version = helper_model.get_agg_real_model_version(model_version)
    #     metafunc.parametrize("centaur_model_version", [real_model_version], ids=[real_model_version])
    # if 'centaur_model' in metafunc.fixturenames:
    #     # Replace the string by the real model version if model=='current'.
    #     # Only one model will be allowed for performance reasons (only one global 'centaur_model' will be loaded)
    #     model_version = metafunc.config.getoption('model')
    #     real_model_version = helper_model.get_agg_real_model_version(model_version)
    #     metafunc.parametrize("centaur_model", [_get_model_(real_model_version)], ids=[real_model_version])


