import os
import tqdm
import pandas as pd
import sys
import itertools
import numpy as np
import argparse

sys.path.append('../')
from deephealth_utils.misc.utils import run_pool_unordered
from deephealth_utils.data import dh_dcmread


def create_file_df(base_dir, out_file):
    _, study_dirs, files = next(os.walk(base_dir))
    assert len(files) == 0, 'Directory contains files not in study directories'

    all_files = []
    n_empty_directories = 0
    print('Getting list of all files...')
    for this_dir in tqdm.tqdm(study_dirs, total=len(study_dirs), ascii=True):
        full_dir, subdirs, files = next(os.walk(base_dir + this_dir))
        assert len(subdirs) == 0, this_dir + ' has subdirectories'
        if len(files) == 0:
            n_empty_directories += 1
        all_files += [os.path.join(full_dir, f) for f in files]

    dicom_tags = ['SOPClassUID', 'SOPInstanceUID', 'StudyInstanceUID', 'PatientID', 'StudyDate',
                  'StudyDescription', 'Manufacturer', 'ManufacturerModelName',
                  'ViewPosition', 'ImageLaterality', 'BreastImplantPresent']

    data = {v: [] for v in dicom_tags + ['study_dir', 'file_name']}

    inputs = zip(all_files, itertools.repeat(dicom_tags))
    print('Extracting info from all files...')
    results = run_pool_unordered(tag_extract_one, inputs, n=len(all_files))

    for tag_dict in results:
        for t in dicom_tags:
            data[t].append(tag_dict[t])

        data['study_dir'].append(tag_dict['study_dir'])
        data['file_name'].append(tag_dict['file_name'])

    df = pd.DataFrame(data)
    df.sort_values('StudyInstanceUID', inplace=True)
    df.reset_index(drop=True, inplace=True) # reset index to 0

    str_tags = ['SOPInstanceUID', 'StudyInstanceUID', 'PatientID', 'StudyDate']
    convert_to_int = ['StudyDate']
    for s in str_tags:
        for i in df.index:
            if df.loc[i, s] == df.loc[i, s]:  # not nan
                val = str(df.loc[i, s])
                if s in convert_to_int:
                    if val.endswith('.0'):
                        val = val[:-2]
                df.loc[i, s] = val

    print('Num. empty directories:', n_empty_directories, '/', len(study_dirs))

    df.to_csv(out_file)


def tag_extract_one(inputs):
    file_name, dicom_tags = inputs
    ds = dh_dcmread(file_name, stop_before_pixels=True)
    tag_data = {}

    dir_path, f = os.path.split(file_name)
    study_dir = dir_path[dir_path.rfind('/'):].replace('/', '')
    tag_data['study_dir'] = study_dir
    tag_data['file_name'] = f

    for t in dicom_tags:
        try:
            val = ds.dh_getattribute(t)
        except Exception as e:
            if 'KeyError' in str(type(e)):
                val = None
            else:
                val = None
                # pass
                # print(file_name)
                # raise e

        if val is None:
            val = np.nan

        if 'pydicom' in str(type(val)):
            val = str(val)

        if len(str(val)) > 400:
            val = str(val)[:400]

        tag_data[t] = val

    return tag_data


def sanity_check_file_df(df):
    # Check that each file in a study has same tags
    tags_to_check = ['PatientID', 'StudyDate', 'StudyInstanceUID']
    errors = []
    for t in tags_to_check:
        df2 = df[['study_dir', t]]
        df2 = df2[pd.notna(df2[t])]

        if not len(df2):
            print('No non-na entries for', t, '\n')
            continue

        df2 = df2.groupby('study_dir')[t].nunique()
        idx = df2 > 1
        n_dups = np.sum(idx)
        if n_dups:
            errors.append((t, n_dups))
            print('Multiple values for', t, n_dups, 'times', '\n')

    # Check for duplicated SOPInstanceUIDs
    duplicate_SOPInstanceUIDs = df[df['SOPInstanceUID'].duplicated(keep=False)]
    if len(duplicate_SOPInstanceUIDs):
        print('The following rows with duplicate SOP Instance UIDs were found: \n')
        pd.set_option('display.max_rows', len(duplicate_SOPInstanceUIDs))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', -1)
        print(duplicate_SOPInstanceUIDs)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_colwidth')

    # Get number of NaNs per column
    print('\nCounts of NaNs by column:')
    for c in df.columns:
        count = pd.isnull(df[c]).sum()
        print(c + ': ' + str(count) + '/' + str(len(df)))
    print('\n')

    # Look at number of files per folder
    print('Counts of number of files per folder:')
    counts_df = df.groupby('study_dir').count()
    print(counts_df['file_name'].value_counts())
    print('\n')


# def validate_consistent_patient_dir(df):
#     '''
#     Check that each file in directory has consistent information
#     '''
#     u_folders = np.unique(df['study_dir'])
#     print('Validating PHI consistency within study directories...')
#
#     cols_to_check = ['PatientID', 'StudyInstanceUID', 'StudyDate']
#
#     bad_directories = []
#     for fi, f in tqdm.tqdm(enumerate(u_folders),total=len(u_folders)):
#         idx = df['study_dir'] == f
#         this_df = df[idx]
#         if len(this_df) > 1:
#             ref_row = this_df.iloc[0]
#             for i in range(1, len(this_df)):
#                 row = this_df.iloc[i]
#                 for c in cols_to_check:
#                     if ref_row[c] != row[c]:
#                         if f not in bad_directories:
#                             bad_directories.append(f)
#
#     print('# of inconsistent directories:', len(bad_directories))
#     print('Bad directories:', bad_directories)


def print_counts_by_col(df, cols=['SOPClassUID', 'Manufacturer', 'ManufacturerModelName', 'StudyDescription', 'BreastImplantPresent']):
    for c in cols:
        print('Counts by', c)
        print(df[c].value_counts())
        print()


def print_folders_with_sop_class(df):
    tag = 'SOPClassUID'
    vals = df[tag].unique()

    print('Total number of folders:', df['study_dir'].nunique())
    print('Number of folders with each file type:')
    for v in vals:
        if(v==v): #check that not NaN
            this_df = df[df[tag] == v]
            print(v + ': ' + str(this_df['study_dir'].nunique()))
    print('\n')


def summarize_sop_class_by_dir(df):
    u_dirs = np.unique(df['study_dir'])
    counts = {}
    for d in u_dirs:
        this_df = df[df['study_dir'] == d]
        this_df = this_df.groupby('SOPClassUID').count()
        s = ''
        for i, row in this_df.iterrows():
            s += i + '-' + str(row['study_dir']) + ' '
        if s not in counts:
            counts[s] = 0
        counts[s] += 1
    print('# of Occurences of Directory Structure: (file types and numbers, count of directories)')
    for s in counts:
        print(s, counts[s])
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('df_path')
    parser.add_argument('-dicom_dir', dest='dicom_dir', default=None)
    for step in ['create_df', 'sanity_check', 'summarize']:
        parser.add_argument('-' + step, dest='run_' + step, action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])

    if args.run_create_df:
        assert args.dicom_dir is not None, 'Need to specify dicom_dir'
        create_file_df(args.dicom_dir, args.df_path)

    if args.run_sanity_check:
        df = pd.read_csv(args.df_path, index_col=0)
        sanity_check_file_df(df)

    if args.run_summarize:
        df = pd.read_csv(args.df_path, index_col=0)
        print_counts_by_col(df)
        print_folders_with_sop_class(df)
        summarize_sop_class_by_dir(df)
