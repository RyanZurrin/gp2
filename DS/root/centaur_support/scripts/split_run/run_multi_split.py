import os
import argparse
import centaur_support.scripts.split_run.constants as const
import centaur_support.scripts.split_run.split_utils as split_utils


def parse_args():
    """
    Parses command line arguments for running data on multiple GPUs.
    Returns:
        dict: A dictionary containing the values for the specified arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', required=True, help='The run mode to use when running on the data')
    parser.add_argument('--checker_mode', required=True, help='The checker mode to use when running on the data')
    parser.add_argument('--checks_to_ignore', help='Checks to ignore (if any) when running on the data')
    parser.add_argument('--checks_to_include', help='Checks to include (if any) when running on the data')
    parser.add_argument('--save_synthetics', action='store_true', default=False,
                        help='Save generated synthetics when running on the data')
    parser.add_argument('--mount', action='append', help='Mount an external directory in the container')
    parser.add_argument('--gpus', required=True,
                        help='Which GPUs to use for the splits. GPU names should be separated by underscores')
    parser.add_argument('--container_base_name', required=True,
                        help='The base container name. The script will create containers named {base_name}_split_#')
    parser.add_argument('--image_id', required=True,
                        help='The ID of the Docker image that will be used to run on the data')
    # One of input_file, input_dir must be specified
    parser.add_argument('--input_file', help='The absolute path to the input file. This file must contain the mounted '
                                             'paths of files to run on. If the file is a .csv or .pkl DF, it must have '
                                             'a "file_path" column')
    parser.add_argument('--input_dir', help='The absolute path to the input directory')
    parser.add_argument('--save_dir', required=True, help='The absolute path to the directory in which the inputs and '
                                                          'outputs will be saved')

    return vars(parser.parse_args())


def run_splits(input_splits, run_mode, checker_mode, checks_to_ignore, checks_to_include, save_synthetics,
               mount, gpus, container_base_name, image_id, **kwargs):
    """
    Calls run_split() for each individual split.
    Args:
        input_splits (dict): A dictionary containing each split's file DF, the path to the split's input DF, and the
        path to the split's output directory.
        run_mode (str): The run mode that will be used when running the data.
        checker_mode (str): The checker mode that will be used when running the data.
        checks_to_ignore (str): Checks to ignore (separated by spaces), independent of run_mode and checker_mode.
        checks_to_include (str): Checks to include (separated by spaces), independent of run_mode and checker_mode.
        save_synthetics (bool): Whether to save generated synthetic images (this does not affect MSP report generation).
        mount (list): Directories to mount in the container. This is a list containing strings of the form
        "/original/path/:/container/path/".
        gpus (str): The numbers of GPUs to use when running the data (separated by underscores). The number of specified
        GPUs will be used to determine the number of splits (e.g. "--gpus 0_1_2" will use 3 splits).
        container_base_name (str): The base name used for the container. The script will created a container named
        "{base_name}_split_#" for each data split.
        image_id (str): The ImageID of the Docker image that will be used to run on the data.
        **kwargs: Additional unused arguments specified in the parsed arg dict.

    Returns:
        None

    """
    gpus = gpus.split('_')

    for split_idx, (input_split, gpu) in enumerate(list(zip(input_splits, gpus))):
        container_name = f'{container_base_name}_split_{split_idx + 1}'
        run_split(input_split, run_mode, checker_mode, checks_to_ignore, checks_to_include, save_synthetics,
                  mount, gpu, container_name, image_id)


def run_split(input_split, run_mode, checker_mode, checks_to_ignore, checks_to_include, save_synthetics,
              mount, gpu_num, container_name, image_id):
    """
    Creates a Docker container and runs centaur/centaur_deploy/deploy.py for the data split.
    Args:
        input_split (dict): A dictionary containing the split file DF, the path to the split's input DF, and the path
        to the split's output directory.
        run_mode (str): The run mode that will be used when running the data.
        checker_mode (str): The checker mode that will be used when running the data.
        checks_to_ignore (str): Checks to ignore (separated by spaces), independent of run_mode and checker_mode.
        checks_to_include (str): Checks to include (separated by spaces), independent of run_mode and checker_mode.
        save_synthetics (bool): Whether to save generated synthetic images (this does not affect MSP report generation).
        mount (list): Directories to mount in the container. This is a list containing strings of the form
        "/original/path/:/container/path/".
        gpu_num (str): The number of the GPU to use when running the data.
        container_name (str): The name of the container used for the data split.
        image_id (str): The ImageID of the Docker image that will be used to run on the data.

    Returns:
        None

    """
    mounted_split_dir = os.path.join('/root', const.CONTAINER_MOUNTED_SPLIT_DIR_NAME)
    input_dir = os.path.join(mounted_split_dir, input_split["input"])
    output_dir = os.path.join(mounted_split_dir, input_split["output"])
    mounts = mount if mount is not None else []
    mounts = ' '.join([f'-v {mount}' for mount in mounts] + [f'-v {input_split["dir"]}:{mounted_split_dir}'])

    deploy_command = f'python centaur/centaur_deploy/deploy.py --input_dir {input_dir} --output_dir {output_dir} ' \
                     f'--run_mode {run_mode} --checker_mode {checker_mode}'
    if checks_to_ignore is not None:
        deploy_command += f' --checks_to_ignore {checks_to_ignore}'
    if checks_to_include is not None:
        deploy_command += f' --checks_to_include {checks_to_include}'
    if save_synthetics is not None:
        deploy_command += ' --save_synthetics'
    docker_init_base_command = f'docker run -idt -e CUDA_VISIBLE_DEVICES={gpu_num} --runtime nvidia --privileged ' \
                               f'{mounts} --name {container_name} {image_id} {deploy_command}'
    os.system(docker_init_base_command)


def run_setup(input_splits, save_dir):
    """
    Creates the directory structure and input DFs required to run the data over multiple splits.
    Args:
        input_splits (list): A list of file DFs (each DF contains the files for a single split).
        save_dir (str): The path to the root directory in which split directories and files will be created.

    Returns:
        list: A dictionary containing each split's file DF, the path to the split's input DF, and the
        path to the split's output directory.

    """
    assert not os.path.isdir(save_dir)
    os.mkdir(save_dir)
    base_save_dir = os.path.join(save_dir, const.BASE_SAVE_DIR_NAME)
    assert not os.path.isdir(base_save_dir)
    os.mkdir(base_save_dir)

    all_split_dirs = []
    for split_idx, split_df in enumerate(input_splits):
        split_name = split_idx + 1
        split_dir = os.path.join(base_save_dir, const.BASE_SPLIT_DIR_NAME.format(split_name))
        assert not os.path.isdir(split_dir)
        os.mkdir(split_dir)
        split_input_dir = os.path.join(split_dir, const.SPLIT_INPUT_DIR_NAME)
        split_output_dir = os.path.join(split_dir, const.SPLIT_OUTPUT_DIR_NAME)
        os.mkdir(split_input_dir)
        os.mkdir(split_output_dir)
        split_df.to_csv(os.path.join(split_input_dir, const.SPLIT_CSV_NAME.format(split_name)), index=False)
        all_split_dirs.append({'dir': split_dir,
                               'input': os.path.join(const.SPLIT_INPUT_DIR_NAME, const.SPLIT_CSV_NAME.format(split_name)),
                               'output': const.SPLIT_OUTPUT_DIR_NAME})

    return all_split_dirs


if __name__ == '__main__':
    args = parse_args()
    n_splits = len(args['gpus'].split('_'))
    input_df = split_utils.get_input_df(args['input_file'], args['input_dir'])
    input_splits = split_utils.split_inputs(input_df, n_splits)
    split_dirs = run_setup(input_splits, args['save_dir'])
    run_splits(split_dirs, **args)
