"""
Simple parser of a results_json file to simplify future changes in json file format
"""
import copy
import datetime
import glob
from tqdm import tqdm
import pandas as pd
import pydicom
import json
import os
import numpy as np
import os.path as osp

from deephealth_utils.data.dicom_type_helpers import DicomTypeMap


def sr_to_df(sr_path, output_csv_path=None, csv_save_name='sr_bbx_df.csv'):
    """
    Extracts bounding box information from an SR and puts it in a DF. The DF will also be saved as a CSV if output_csv_path
    is specified.
    Args:
        sr_path (str): The path to the SR file.
        output_csv_path (str): The path to the directory to which the generated CSV will be saved. (Optional)
        csv_save_name (str): The name that the generated CSV will be saved as. (Optional)

    References:
        centaur_engine.helpers.helper_category

    Returns:
        pd.DataFrame: A DF containing the extracted bounding box information.
    """
    from centaur.centaur_engine.helpers.helper_category import CategoryHelper

    dcm = pydicom.dcmread(sr_path)

    # First need to parse the findings container.
    # It doesn't contain the SOPInstanceUID, so we will just need to use the library index as key.
    box_dict = dict()
    timestamp = datetime.datetime.now().isoformat()
    study_instance_uid = dcm.StudyInstanceUID

    for item in dcm.ContentSequence[2].ContentSequence:
        if item.ConceptNameCodeSequence[0].CodeMeaning == 'Individual Impression/Recommendation':
            for box in item.ContentSequence[1:]:
                slice_num = None
                for box_item in box.ContentSequence:
                    name = box_item.ConceptNameCodeSequence[0].CodeMeaning
                    if name == 'Finding Assessment':
                        category = box_item.TextValue
                    elif name == 'Outline':
                        coords = box_item.GraphicData
                        box_coords = [coords[0], coords[1], coords[2], coords[5]]
                        library_idx = box_item.ContentSequence[0].ReferencedContentItemIdentifier[2]
                    elif name == 'Linked DBT Frame':
                        slice_num = box_item.TextValue

                box_dict[library_idx] = {'category': category,
                                         'coords': box_coords,
                                         'slice': slice_num}

    # Next, we parse the library container to match with SOPInstanceUID
    sop_uid_dict = dict()

    for idx, item in enumerate(dcm.ContentSequence[1].ContentSequence):
        sop_uid = item.ReferencedSOPSequence[0].ReferencedSOPInstanceUID
        ref_sop_class_uid = item.ReferencedSOPSequence[0].ReferencedSOPClassUID  # ADDED
        try:
            slice_num = box_dict[idx + 1]['slice']
        except KeyError:
            continue
        if item.ReferencedSOPSequence[0].ReferencedSOPClassUID == '1.2.840.10008.5.1.4.1.1.13.1.3':
            slice_num = item.ReferencedSOPSequence[0].ReferencedFrameNumber

        coords = box_dict[idx + 1]['coords']

        bbox_info = {'SOPInstanceUID': sop_uid,
                     'ref_SOPClassUID': ref_sop_class_uid,
                     'bbx_transf': 'none',
                     'bbx_ix': 0,
                     'score': None,
                     'category': [k for k, v in CategoryHelper.get_category_abbreviations('CADx').items() if
                                  v == box_dict[idx + 1]['category']][0],
                     'coords': coords,
                     'slice': int(slice_num) if slice_num is not None else None,
                     'origin': np.nan,
                     'StudyInstanceUID': study_instance_uid,
                     'timestamp': timestamp}

        if sop_uid in sop_uid_dict:
            bbox_info['bbx_ix'] += len(sop_uid_dict[sop_uid])  # CHANGED (was bugged before)
            sop_uid_dict[sop_uid].append(bbox_info)
        else:
            sop_uid_dict[sop_uid] = [bbox_info]

    df_rows = []
    for sop_uid in sop_uid_dict:
        for box in sop_uid_dict[sop_uid]:
            df_rows.append(box)

    df = pd.DataFrame(df_rows)

    if output_csv_path is not None:
        df.to_csv(os.path.join(output_csv_path, csv_save_name))

    return df


def compare_sr_boxes_to_results(sr_df, results_df, study_uid):
    """
    For a given study, hecks that the boxes extracted from the SR and the boxes in the results summary report are the
    same.
    Args:
        sr_df (pd.DataFrame): A DF containing bounding box information extracted from an SR.
        results_df (pd.DataFrame): The bounding box predictions DF.
        study_uid (str): The StudyInstanceUID of the study.

    Returns:
        None
    """
    # We are not comparing origin or timestamp since the information is not contained in SR
    columns_to_compare = ['SOPInstanceUID', 'bbx_transf', 'bbx_ix', 'category', 'coords', 'StudyInstanceUID']
    # Sort in ascending order to compare element-wise
    try:
        sr_df_cp = sr_df.copy().sort_values(by=['SOPInstanceUID', 'bbx_ix'], ascending=True).reset_index(drop=True)
    except KeyError:
        assert results_df[results_df['StudyInstanceUID'] == study_uid].shape[0] == 0
        raise KeyError
    sr_sop_instance_uids = sr_df_cp['SOPInstanceUID'].unique()
    sr_study_uids = sr_df_cp['StudyInstanceUID'].unique()
    # Sort in ascending order to compare element-wise
    results_df_sub = results_df[results_df['SOPInstanceUID'].isin(sr_sop_instance_uids)].copy()
    results_df_sub = results_df_sub.sort_values(by=['SOPInstanceUID', 'bbx_ix'], ascending=True).reset_index(drop=True)
    results_sop_instance_uids = results_df_sub['SOPInstanceUID'].unique()
    results_study_uids = results_df_sub['StudyInstanceUID'].unique()
    # Check that the SOPInstanceUIDs in the two DFs are the same
    # This effectively checks that all the SOPInstanceUIDs in the SR DF are also found in the results DF
    assert set(results_sop_instance_uids) == set(sr_sop_instance_uids), \
        'Got different SOPInstanceUIDs. Results: {}, SR: {}'.format(results_sop_instance_uids, sr_sop_instance_uids)
    # Check that the StudyInstanceUIDs in the two DFs are the same
    assert set(results_study_uids) == set(sr_study_uids), 'Got different StudyInstanceUIDs. Results: {}, SR: {}'.format(
        results_study_uids, sr_study_uids)
    # Check that the two DFs have the same number of rows (effectively the same number of boxes)
    assert results_df_sub.shape[0] == sr_df_cp.shape[0], 'Unequal number of boxes. Results: {}, SR: {}'.format(
        results_df_sub.shape[0], sr_df_cp.shape[0])

    for column in columns_to_compare:
        assert (sr_df_cp[column] == results_df_sub[column]).all(), 'DFs are not equal in column {}'.format(column)

    # Check that slice values are the same in both DFs
    assert ((sr_df_cp['slice'].isnull()) | (sr_df_cp['slice'] >= 0)).all(), 'The SR DF contains negative slice values'
    assert ((results_df_sub['slice'].isnull()) | (results_df_sub['slice'] >= 0)).all(), \
        'The results DF contains negative slice values'

    dxm_mask = sr_df_cp['ref_SOPClassUID'] == DicomTypeMap.get_dxm_class_id()
    bt_mask = sr_df_cp['ref_SOPClassUID'] == DicomTypeMap.get_dbt_class_id()
    assert sr_df_cp[dxm_mask]['slice'].isnull().all()
    assert sr_df_cp[bt_mask]['slice'].notnull().all()
    assert sr_df_cp.loc[bt_mask, 'slice'].astype(int).equals(results_df_sub.loc[bt_mask, 'slice'].astype(int)), \
        'Got unequal slice numbers'


def load_results_json(results_file_path):
    """
    Loads a json results file.
    Args:
        results_file_path (str): The path to the json results file.

    Returns:
        dict: The json results file loaded as a dictionary.
    """
    assert os.path.isfile(results_file_path), "Results file not found ({})".format(results_file_path)
    with open(results_file_path, 'r') as fp:
        results_dict = json.load(fp)
    return results_dict


def get_metadata(results_dict):
    raise NotImplementedError("This function has been now deprecated. Please use the StudyDeployResults class")


def get_model_results(results_dict):
    """
    Get a dictionary with the 'model_results' items.
    This function has been modified to adapt to the new StudyDeployResults class and it's kept just for
    backwards compatibility reasons
    Args:
        results_dict (dict): StudyDeployResults.results object expected

    Returns:
        It just return the same object (identity function). Kept for backwards compatibility reasons
    """
    return(results_dict)


def get_study_results(results_dict):
    return get_model_results(results_dict)['study_results']


def get_dicom_results(results_dict):
    return get_model_results(results_dict)['dicom_results']


def get_proc_info(results_dict):
    return get_model_results(results_dict)['proc_info']


def get_studyUID(results_dict):
    raise NotImplementedError("This function has been now deprecated. Please use the StudyDeployResults class instead")


def are_equal(obj1, obj2, tolerance=1e-6):
    """
    Compare two objects recursively. If the item to compare is a float number, use a tolerance threshold
    to allow small discrepancies
    Args:
        obj1: Any object.
        obj2: Any object.
        tolerance (float): The tolerance threshold. Objects that differ by a value smaller than the tolerance threshold
        will be considered equal.

    Returns:
        bool: Whether the two objects are equal.
    """
    # Convert tuples to lists to facilitate the process
    if isinstance(obj1, tuple):
        o1 = list(obj1)
    else:
        o1 = copy.copy(obj1)
    if isinstance(obj2, tuple):
        o2 = list(obj2)
    else:
        o2 = copy.copy(obj2)

    if isinstance(o1, float) and pd.isnull(o1):
        o1 = None
    if isinstance(o2, float) and pd.isnull(o2):
        o2 = None

    if isinstance(o1, list):
        if len(o1) != len(o2):
            return False
        try:
            o1.sort()
            o2.sort()
        except:
            pass

        for i in range(len(o1)):
            return are_equal(o1[i], o2[i])
    elif isinstance(o1, dict):
        if not isinstance(o2, dict):
            return False
        if o1.keys() != o2.keys():
            return False
        for k, v in o1.items():
            if not are_equal(v, o2[k]):
                return False
    elif isinstance(o1, float):
        if not isinstance(o2, float):
            return False
        # Check for tolerance
        t = abs(o1 - o2) <= tolerance
        if t:
            ret = True
        else:
            ret = False
        return ret

    elif str(o1) != str(o2):
        return False
    return True


def parse_json_results_files(input_results_dir, output_dir=None, include_study_df=False, simplify=False):
    """
    Parse all the results.json files in a folder and aggregate the image scores results in a dataframe.
    Args:
        input_results_dir (str): The path to the centaur results folder.
        output_dir (str): The directory in which the result file will be saved.
        include_study_df (bool): Whether to generate a DF with aggregated study scores.
        simplify (bool): Whether to simplify the output results (for instance, when sending results to an external
        source).

    Returns:
        pd.DataFrame or dictionary of pd.DataFrame objects: DF containing parsed results.
    """
    # TODO Implement deprecated function (if needed)
    raise NotImplementedError("This function needs to be updated for the new StudyResults format")
    results_files = glob.glob(input_results_dir + "/**/results.json", recursive=True)
    timestamp = datetime.datetime.now().isoformat()
    dicom_bbx_dfs = []
    study_dfs = [] if include_study_df else None

    for results_file in tqdm(results_files):
        with open(results_file, 'r') as f:
            results_dict = json.load(f)
        dfs = parse_results(results_dict, include_study_df=include_study_df)
        if include_study_df:
            study_dfs.append(dfs['study_df'])
            dicom_bbx_dfs.append(dfs['dicom_bbx_df'])
        else:
            dicom_bbx_dfs.append(dfs)

    # Aggregate all the results
    dicom_bbx_df = pd.concat(dicom_bbx_dfs)
    dicom_bbx_df['timestamp'] = timestamp
    study_df = pd.concat(study_dfs) if study_dfs is not None else None

    if simplify:
        dicom_bbx_df, study_df = _simplify_aggregated_df_results(dicom_bbx_df, study_df)

    if output_dir is not None:
        try:
            dicom_bbx_df.to_csv(osp.join(output_dir, "results_dicom_bbx_df.csv"))
            if include_study_df:
                study_df.to_csv(osp.join(output_dir, "results_study_df.csv"))
            print("Results file saved to {}".format(output_dir))
        except Exception as ex:
            print("Results could not be saved: {}".format(ex))

    if include_study_df:
        return {
            'dicom_bbx_df': dicom_bbx_df,
            'study_df': study_df
        }
    return dicom_bbx_df

def parse_results(study_deploy_results, include_study_df=False):
    """
    Read a StudyDeployResults object and  generate aggregated dataframes with the bounding boxes info
    Args:
        study_deploy_results (StudyDeployResults): StudyDeployResults object
        include_study_df (bool): Include study info

    Returns:
        dataframe or dictionary of dataframes if include_study_df==True
    """
    timestamp = datetime.datetime.now().isoformat()
    study_df = None

    if include_study_df:
        study_df = pd.DataFrame(columns=['score_L', 'category_L',
                                         'score_R', 'category_R',
                                         'score_total', 'category_total',
                                         'study_dir', 'timestamp'])
        study_df.index.name = "StudyInstanceUID"

    dicom_bbx_df = pd.DataFrame(columns=['SOPInstanceUID', 'bbx_transf', 'bbx_ix', 'score',
                                         'category', 'coords', 'slice', 'origin', 'StudyInstanceUID', 'timestamp'])


    study_uid = study_deploy_results.get_studyUID()
    study_results = study_deploy_results.results
    if include_study_df:
        study_dir = study_uid
        # Aggregated study results
        score_L = category_L = None
        score_R = category_R = None
        if 'L' in study_results:
            score_L = study_results['L']['score']
            category_L = study_results['L']['category']
        else:
            print("WARNING: 'L' laterality not available in study {}".format(study_uid))

        if 'R' in study_results:
            score_R = study_results['R']['score']
            category_R = study_results['R']['category']
        else:
            print("WARNING: 'R' laterality not available in study {}".format(study_uid))

        study_df.loc[study_uid] = [score_L, category_L,
                                   score_R, category_R,
                                   study_results['total']['score'], study_results['total']['category'],
                                   study_dir, timestamp]

    dicom_results = get_dicom_results(study_results)
    for instance_uid, transf_dict in dicom_results.items():
        for transf_key, bbxs_list in transf_dict.items():
            for i in range(len(bbxs_list)):
                ix = len(dicom_bbx_df)
                bbx = bbxs_list[i]
                slice_ = bbx['slice'] if 'slice' in bbx else None
                origin = bbx['origin'] if 'origin' in bbx else None
                dicom_bbx_df.loc[ix] = [instance_uid, transf_key, i, bbx['score'],
                                        bbx['category'] if 'category' in bbx else None,
                                        bbx['coords'], slice_, origin, study_uid, timestamp]

    if include_study_df:
        return {
            'dicom_bbx_df': dicom_bbx_df,
            'study_df': study_df
        }
    return dicom_bbx_df

def _simplify_aggregated_df_results(dicom_bbx_df, study_df):
    """
    Simplify a dicom and (optionally) a study aggregated dataframe for an easier results visualization
    :param dicom_bbx_df: Dataframe
    :param study_df: Dataframe (or None)
    :return: tuple with:
        - Image bounding boxes dataframe (StudyInstanceUID, SOPInstanceUID, x1, x2, y1, y2, score, slice)
        - Study dataframe (StudyInstanceUID, score). None if study_df==None
    """
    study_df_simplified = None
    def simplify_dicom(row):
        return pd.Series(data={
            'StudyInstanceUID': row['StudyInstanceUID'],
            'SOPInstanceUID': row['SOPInstanceUID'],
            'x1': int(row['coords'][0]),
            'y1': int(row['coords'][1]),
            'x2': int(row['coords'][2]),
            'y2': int(row['coords'][3]),
            'score': row['score'],
            'slice_num': int(row['slice']) if not pd.isnull(row['slice']) else -1
        })

    dicom_bbx_df_simplified = dicom_bbx_df.loc[dicom_bbx_df['bbx_transf']=='none'].reset_index().apply(simplify_dicom, axis=1)

    if study_df is not None:
        study_df_simplified = study_df[['score_total_postprocessed']].reset_index().\
            rename(columns={'score_total_postprocessed': 'score'})
    return dicom_bbx_df_simplified, study_df_simplified

def get_most_malignant_lesion(study_deploy_results):
    """
    Get the SOPInstanceUID and the bounding box info for the most malignant lesion found
    Args:
        study_deploy_results (StudyDeployResults): StudyDeployResults object
    Returns:
        instance_uid (str), bounding box info (dict)
    """

    dicom_results = get_dicom_results(study_deploy_results.results)
    metadata = study_deploy_results.get_metadata(passed_checker_only=True)
    is_dbt_study = DicomTypeMap.get_study_type(metadata) == DicomTypeMap.DBT
    most_malignant_image_id = None
    most_malignant_bbx = None
    max_score = 0

    for _, row in metadata.iterrows():
        sop_instance_uid = row['SOPInstanceUID']
        results_bbxs = dicom_results[sop_instance_uid]['none']
        for bbx in results_bbxs:
            if bbx['score'] > max_score:
                # if the study is dbt, and we combined 2D and 3D boxes. if the
                # boxes are combined it will have the origin key.
                if is_dbt_study and 'origin' in bbx:
                    # If the study is dbt, we ensure we select the original image and not a projected bbx
                    if bbx['origin'] == DicomTypeMap.get_type_row(row):
                        most_malignant_image_id = sop_instance_uid
                        most_malignant_bbx = copy.copy(bbx)
                        max_score = bbx['score']

                # 2D study, or 3D study without boxes combined
                else:
                    most_malignant_image_id = sop_instance_uid
                    most_malignant_bbx = copy.copy(bbx)
                    max_score = bbx['score']

                    # case where 2D and we have an origin key
                    if 'origin' in bbx:
                        assert most_malignant_bbx['origin'] == DicomTypeMap.get_type_row(row), \
                            "projected box in a 2D study"
                    else:
                        most_malignant_bbx['origin'] = DicomTypeMap.get_type_row(row)

    assert max_score > 0, "Scores were not found!"
    return most_malignant_image_id, most_malignant_bbx


