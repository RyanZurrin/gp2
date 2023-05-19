import os
import numpy as np
import pandas as pd
import centaur_support.scripts.split_run.constants as const


def get_input_df(input_file, input_dir):
    """
    Creates an input DF from the specified input file or input directory. Splitting happens by study directory, and it
    is therefore assumed that all files in a study are contained in the same directory.
    If an input file is specified:
        1. If the file is a .csv or .pkl, it must be a DF with the column "file_path". This column must contain the
        absolute path to each file inside the docker container. The data directory specified by the arguments is mounted
        to /mnt/ inside the docker container by default.
        2. If the file is a .txt file, it must contain the absolute paths to the files inside the docker container,
        separated by the "\n" character.
    If an input directory is specified, the code will search for files under the specified directory.
    Args:
        input_file (str): The absolute path to the input file.
        input_dir (str): The absolute path to the input directory.

    Returns:
        DataFrame: The input DF used for running the splits.

    """
    # Either an input file or an input directory can be specified, not both
    assert (input_file is not None) ^ (input_dir is not None)
    if input_file:
        assert os.path.isfile(input_file)
        if input_file.endswith('.csv'):
            input_df = pd.read_csv(input_file)
        elif input_file.endswith('.pkl'):
            input_df = pd.read_pickle(input_file)
        elif input_file.endswith('.txt'):
            with open(input_file, 'r') as in_f:
                text_inputs = [i for i in in_f.read().split('\n') if len(i) > 0]
            input_df = pd.DataFrame({const.SPLIT_DF_FPATH_COL_NAME: text_inputs})
        else:
            raise NotImplementedError(f'File {input_file} has an unsupported format (expected one of .csv, .pkl, .txt')

        # Add a study directory column to the input DF
        input_df[const.SPLIT_DF_STUDY_DIR_COL_NAME] = input_df[const.SPLIT_DF_FPATH_COL_NAME].apply(lambda p: p.split('/')[-2])
    else:
        assert os.path.isdir(input_dir)
        all_files = []
        all_studies = []
        for root, dirs, files in os.walk(input_dir):
            files = [f for f in files if not f.startswith('.')]
            if len(files) > 0:
                file_list = [os.path.join(root, x).replace(input_dir, const.MOUNT_DIR) for x in files]
                study_name = os.path.basename(root)
                all_files += file_list
                all_studies += [study_name] * len(file_list)

        input_df = pd.DataFrame({const.SPLIT_DF_FPATH_COL_NAME: all_files, const.SPLIT_DF_STUDY_DIR_COL_NAME: all_studies})

    return input_df


def split_inputs(input_df, n_splits):
    """
    Splits the input DF created by get_input_df() into multiple DFs. Splitting happens by study directory.
    Args:
        input_df (DataFrame): The input DF created by get_input_df().
        n_splits (int): The number of sub-DFs that the input DF should be split into.

    Returns:
        list: A list containing split DFs.

    """
    study_splits = np.array_split(input_df[const.SPLIT_DF_STUDY_DIR_COL_NAME].unique(), n_splits)
    split_dfs = [input_df[input_df[const.SPLIT_DF_STUDY_DIR_COL_NAME].isin(study_split)].copy() for study_split in study_splits]

    # Check that the concatenated split DFs match up to the original input DF.
    assert pd.concat(split_dfs, ignore_index=True).sort_values(by=const.SPLIT_DF_FPATH_COL_NAME).equals(
        input_df.sort_values(by=const.SPLIT_DF_FPATH_COL_NAME)), 'Split DFs do not equal original DF'

    return split_dfs
