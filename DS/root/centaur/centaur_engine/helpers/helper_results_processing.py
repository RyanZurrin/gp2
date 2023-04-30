import copy
import logging
import numpy as np
import os

import scipy.stats as stats
import centaur_deploy.constants as const_deploy
from centaur_deploy.deploys.config import Config
from centaur_engine.helpers import helper_preprocessor
from centaur_engine.helpers.helper_model import get_fourway_category, get_binary_category
from centaur_engine.helpers.helper_category import CategoryHelper
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from deephealth_utils.data.parse_logs import log_line
from deephealth_utils.ml.nms_helpers import nms


def sort_bounding_boxes(model_results):
    """
    Sort all the bounding boxes from higher score to lower score
    :param model_results: dict. Model results
    :return: dictionary (copy of model_results with applied changes)
    """
    mr = copy.deepcopy(model_results)
    dicom_results = mr['dicom_results']
    for instance_uid, transforms in dicom_results.items():
        for transform, bbxs in transforms.items():
            bbxs.sort(key=lambda bb: bb['score'], reverse=True)
    return mr


def orient_coordinates(model_results, metadata):
    """
    Orient the coordinates in images that were orientated before the model prediction.
    Args:
        model_results: Original model results
        metadata: Dataframe. Metadata dataframe

    Returns:dictionary (copy of model_results with applied changes)

    """
    model_results_copy = copy.deepcopy(model_results)

    for _, row in metadata.iterrows():
        orientation_change = helper_preprocessor.get_orientation_change(row['ImageLaterality'],
                                                                        row['PatientOrientation'])

        if orientation_change is not None and \
                orientation_change != "unknown" and \
                orientation_change != "identity":

            instance_uid = row['SOPInstanceUID']
            results_row = model_results_copy['dicom_results'][instance_uid]

            for transform, bbxs in results_row.items():
                for i in range(len(bbxs)):
                    bbx = bbxs[i]
                    # "Rotate" or flip the coordinates if the original image was flipped or  rotated

                    model_results_copy['dicom_results'][instance_uid][transform][i]['coords'] = \
                        orient_bounding_box_coords(bbx['coords'],
                                                   model_results_copy['proc_info'][instance_uid]['original_shape'],
                                                   orientation_change)
    return model_results_copy


def orient_bounding_box_coords(coords, original_shape, orientation_change):
    """
    Calculate the modified coords for a bounding box based on orientation_change.
    The coordinates and the original shape should have the same format is used in model results (see engine.py)
    
    Note: All the transforms in this function assume that calling the transform twice is equal to the identity transform. 
    If a transform is added such that this is not the case, logic will need to be updated in a few places in centaur.

    Args:
        coords: list of 4-int. Coordinates
        original_shape: Original image shape (in proc_info format)
        orientation_change: str stating the type of orientation transformation

    Returns:list 4-int. modified coords

    """

    # image upside down need to rotate 180
    if orientation_change == "rotate_180":
        return rotate_bounding_box_coords(coords, original_shape)
    elif orientation_change == "left_right_flip":
        return fliplr_bounding_box_coords(coords, original_shape)
    elif orientation_change == "up_down_flip":
        return flipud_bounding_box_coords(coords, original_shape)

    elif orientation_change == "identity" or \
            orientation_change is None or \
            orientation_change == "unknown":
        return coords
    else:
        raise ValueError ("dont not recognize orientation_change of {} ".format(orientation_change))


def fliplr_bounding_box_coords(coords, original_shape):
    """
    Calculate the left right flipped coords for a bounding box.
    The coordinates and the original shape should have the same format is used in model results (see engine.py)
    Args:
        coords: list of 4-int. Coordinates
        original_shape: Original image shape (in proc_info format)

    Returns:  list 4-int. flipped coords

    """

    _, width = original_shape

    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    x1_new = width - x1 - 1
    x2_new = width - x2 - 1
    flipped_box = [x2_new, y1, x1_new, y2]
    return flipped_box


def flipud_bounding_box_coords(coords, original_shape):
    """
    Calculate the up and down flipped coords for a bounding box.
    The coordinates and the original shape should have the same format is used in model results (see engine.py)
    Args:
        coords:  list of 4-int. Coordinates
        original_shape: Original image shape (in proc_info format)

    Returns: list 4-int. flipped coords

    """
    height, _ = original_shape

    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    y1_new = height - y1 - 1
    y2_new = height - y2 - 1
    flipped_box = [x1, y2_new, x2, y1_new]
    return flipped_box


def rotate_bounding_box_coords(coords, original_shape):
    """
    Calculate the rotated coords for a bounding box.
    The coordinates and the original shape should have the same format is used in model results (see engine.py)
    :param coords: list of 4-int. Coordinates
    :param original_shape: Original image shape (in proc_info format)
    :return: list 4-int. Rotated coords
    """
    original_shape = [original_shape[1], original_shape[0]]
    rotated_bbx = [
        original_shape[0] - coords[2] - 1,
        original_shape[1] - coords[3] - 1,
        original_shape[0] - coords[0] - 1,
        original_shape[1] - coords[1] - 1,
    ]
    return rotated_bbx


def assign_cadx_categories(model_results, metadata_df, thresholds):
    """
    Assign a category
    Args:
        model_results (dictionary):  dict. Original model results
        metadata_df (DataFrame): study metadata
        thresholds (dictionary): contain categorical thresholds for each study modality

    Returns:
        model_results with applied changes (dictionary)
    """
    model_results_copy = copy.deepcopy(model_results)
    dicom_results = model_results_copy['dicom_results']

    thresholds = thresholds[DicomTypeMap.get_study_type(metadata_df)]

    # Bbx categories
    for _, row in metadata_df.iterrows():
        instance_uid = row['SOPInstanceUID']

        for transform in dicom_results[instance_uid]:
            bbxs = dicom_results[instance_uid][transform]
            for i in range(len(bbxs)):
                bbxs[i]['category'] = get_fourway_category(bbxs[i]['score'], thresholds)

    # Lateralities / Global score
    for lat in model_results_copy['study_results']:
        # Assign a category number and category name to each study
        study_category = get_fourway_category(model_results_copy['study_results'][lat]['score'], thresholds)
        model_results_copy['study_results'][lat]['category'] = study_category
        model_results_copy['study_results'][lat]['category_name'] = CategoryHelper.get_category_text(
            study_category, run_mode=const_deploy.RUN_MODE_CADX)

    return model_results_copy


def assign_cadt_categories(model_results, study_modality, thresholds):
    """
    Give a classification of 'SUSPICIOUS' or 'NOT SUSPICIOUS' based on the score and operating point.
    Args:
        model_results (dict): Original model results.
        study_modality (str): study modality (Dxm/DBT)
        thresholds (dictionary): contains operating thresholds for dxm and dbt modalities

    Returns:
        dict. Model results with applied changes.
    """
    model_results_copy = copy.deepcopy(model_results)
    modality = study_modality.lower()
    assert isinstance(thresholds, dict) and modality in thresholds, \
        f"Modality {modality} not found in thresholds: ({thresholds})"
    suspicion_cutoff = thresholds[modality]

    # Assign a category number and category name to each study
    for lat in model_results_copy['study_results']:
        study_suspicion = get_binary_category(model_results_copy['study_results'][lat]['score'], suspicion_cutoff)
        model_results_copy['study_results'][lat]['category'] = study_suspicion
        model_results_copy['study_results'][lat]['category_name'] = CategoryHelper.get_category_text(
            study_suspicion, run_mode=const_deploy.RUN_MODE_CADT)
    return model_results_copy


def cap_bounding_box_category(model_results, metadata, logger=None):
    """
    The category for each laterality and for each bounding box cannot be higher than the category for the study
    :param model_results: dict. Original model results
    :param metadata: dataframe. Metadata dataframe
    :param logger: Logger object (optional)
    :return: dictionary (copy of model_results with applied changes)
    """
    model_results_copy = copy.deepcopy(model_results)
    if logger is None:
        logger = logging.getLogger()

    study_category = model_results_copy['study_results']['total']['category']

    for laterality in [l for l in model_results_copy['study_results'].keys() if l != "total"]:
        laterality_results = model_results_copy['study_results'][laterality]
        if 'category' in laterality_results:
            # The category for each laterality cannot be higher than the category for the study
            laterality_results['category'] = min(study_category, laterality_results['category'])
        else:
            # "Weird" scenario. Allow this at the moment but probably it should change
            logger.warning(log_line(-1, "Category not found for '{}' laterality.".format(laterality)))
            continue

        # The category for each bounding box in the laterality cannot be higher than the category for that laterality
        instance_uids = metadata[metadata['ImageLaterality'] == laterality]['SOPInstanceUID']
        for instance_uid in instance_uids:
            for transform, bbxs in model_results_copy['dicom_results'][instance_uid].items():
                for i in range(len(bbxs)):
                    bbx = bbxs[i]
                    cat = bbx['category']
                    model_results_copy['dicom_results'][instance_uid][transform][i]['category'] = min(cat,
                                                                                                      laterality_results[
                                                                                                          'category'])
    return model_results_copy


def cap_bbx_number_per_image(model_results, max_bbxs_displayed_total, max_bbxs_displayed_intermediate):
    """
    Cap the number of bounding boxes displayed per image, based on the score.
    :param model_results: dict. Original model results
    :param max_bbxs_displayed_total: int. Max number of bbxs displayed in the image
    :param max_bbxs_displayed_intermediate: int. Max number of Intermediate bbxs displayed in the image
    :return: dictionary (copy of model_results with applied changes)
    """
    model_results_copy = copy.deepcopy(model_results)
    dicom_results = model_results_copy['dicom_results']

    for instance_uid in dicom_results:
        # Only look at original images (no other transforms)
        bbxs = dicom_results[instance_uid]['none']
        if len(bbxs) == 0:
            continue
        # Enforce the bounding boxes are sorted by score (most malignant first)
        bbxs.sort(reverse=True, key=lambda bbx: bbx['score'])

        num_high_bbx = num_int_bbx = 0
        filtered_bbx = []
        prev_score = bbxs[0]['score']
        for i in range(min(max_bbxs_displayed_total, len(bbxs))):
            bbx = bbxs[i]
            cat = bbx['category']
            score = bbx['score']
            assert score <= prev_score, "The bounding boxes should be sorted by score"
            if cat == CategoryHelper.HIGH:
                # Always include
                num_high_bbx += 1
                filtered_bbx.append(bbx)
            elif cat == CategoryHelper.INTERMEDIATE:
                if num_high_bbx + num_int_bbx < max_bbxs_displayed_intermediate:
                    num_int_bbx += 1
                    filtered_bbx.append(bbx)
                else:
                    # We reached the maximum number of allowed intermediate bbxs
                    break
            elif cat == CategoryHelper.LOW:
                # Only add if no other bbxs have been added so far
                if num_high_bbx == 0 and num_int_bbx == 0:
                    filtered_bbx.append(bbx)
                # Only one LOW box is allowed
                break
            else:
                # No MINIMAL bbxs are displayed
                break
        dicom_results[instance_uid] = {'none': filtered_bbx}
    return model_results_copy


def fix_bbxs_size(model_results, min_relative_bbx_size_allowed, max_relative_bbx_size_allowed):
    """
    Adjust the maximum and minimum size of the bounding box
    :param model_results: dict. Results
    :param min_relative_bbx_size_allowed: float [0,1]. Min relative size for any bounding box
    :param max_relative_bbx_size_allowed: float [0,1]. Max relative size for any bounding box
    :return: dictionary (copy of model_results with applied changes)
    """
    model_results_copy = copy.deepcopy(model_results)
    dicom_results = model_results_copy['dicom_results']

    for instance_uid, transforms in dicom_results.items():
        for transform in transforms:
            bbxs = dicom_results[instance_uid][transform]
            for i in range(len(bbxs)):
                dicom_results[instance_uid][transform][i]['coords'] = _fix_size(
                    dicom_results[instance_uid][transform][i]['coords'],
                    model_results_copy['proc_info'][instance_uid]['original_shape'],
                    min_relative_bbx_size_allowed, max_relative_bbx_size_allowed)
    return model_results_copy


def _fix_size(bbx_coords, shape, min_rel_size, max_rel_size):
    """
    Adjust a bounding box to a max-min size
    :param bbx_coords: list of 4 int. [Y0,X0,Y1,X1]
    :param shape: 2-int tuple. Full image shape (Y=rows,X=columns)
    :param min_rel_size: float 0-1. Min rate factor for bbx size
    :param max_rel_size: float 0-1. Max rate factor for bbx size
    :return: list of 4-int. Coordinates adjusted
    """
    x0, y0, x1, y1 = bbx_coords
    size_y, size_x = shape

    # Fix possible rounding error of coordinates very close to 1
    if x1 == size_x:
        x1 = size_x - 1
    if y1 == size_y:
        y1 = size_y - 1

    # Sanity check
    if x1 < x0 or y1 < y0 \
            or x0 < 0 or x1 >= size_x \
            or y0 < 0 or y1 >= size_y:
        raise Exception("Wrong coordinates. Coords: {}; Size_x: {}; Size_y: {}".format(bbx_coords, size_x, size_y))

    min_size = int(min(size_x, size_y) * min_rel_size)
    max_size = int(max(size_x, size_y) * max_rel_size)

    # Fix min width (increase size in small bbxs)
    width = x1 - x0
    if width < min_size:
        delta = min_size - width
        x0 -= delta // 2
        x1 += delta // 2
        if x1 - x0 < min_size:
            # Correct rounding error (increase one more pixel)
            x0 -= 1
        if x0 < 0:
            # Out of bounds. Move the bounding box to the right
            despl = abs(x0)
            x0 += despl
            x1 += despl
        elif x1 >= size_x:
            # Idem. Moving to the left
            d = x1 - (size_x + 1)
            x1 -= d
            x0 -= d

    # Fix min height (increase size in small bbxs)
    height = y1 - y0
    if height < min_size:
        delta = min_size - height
        y0 -= delta // 2
        y1 += delta // 2
        if y1 - y0 < min_size:
            # Correct rounding error (increase one more pixel)
            y0 -= 1
        if y0 < 0:
            # Out of bounds. Move the bounding box down
            despl = abs(y0)
            y0 += despl
            y1 += despl
        elif y1 >= size_y:
            # Idem. Move the bounding box up
            d = y1 - (size_y + 1)
            y1 -= d
            y0 -= d

    # Fix max width (decrease size in big bbxs, no need to control out of bounds)
    if width > max_size:
        delta = width - max_size
        x0 += (delta // 2)
        x1 -= (delta // 2)
        # Correct rounding error (decrease one more pixel)
        if x1 - x0 > max_size:
            x0 += 1

    # Fix max height (decrease size in big bbxs, no need to control out of bounds)
    if height > max_size:
        delta = height - max_size
        y0 += delta // 2
        y1 -= delta // 2
        # Correct rounding error (decrease one more pixel)
        if y1 - y0 > max_size:
            y0 += 1

    return [x0, y0, x1, y1]


def model_results_sanity_checks(model_results, raw_model_results, metadata_df,
                                max_bbxs_displayed_total,
                                min_relative_bbx_size_allowed, max_relative_bbx_size_allowed,
                                run_mode='CADx'):
    """
    Basic sanity checks for the model results after post processing:
        - There are not more than X bounding boxes in each image
        - All the bounding boxes have a right size
        - The scores are in the [0-1] range
        - The slice number for the bounding boxes is inbounds (3D images only)
    :param model_results: dict. Model results ('model_results' section in the output file)
    :param max_bbxs_displayed_total: int. Max number of bbxs displayed in the image
    :param min_relative_bbx_size_allowed: float [0,1]. Min relative size for any bounding box
    :param max_relative_bbx_size_allowed: float [0,1]. Max relative size for any bounding box
    :param run_mode: str. Centaur run mode, used to potentially disable certain incompatible checks
    """
    # Compare study-level scores between raw results and post-processed results (they shouldn't change)
    assert model_results['study_results'].keys() == raw_model_results['study_results'].keys()
    for k1 in model_results['study_results'].keys():
        assert model_results['study_results'][k1]['score'] == raw_model_results['study_results'][k1]['score']

    dicom_results = model_results['dicom_results']
    proc_info = model_results['proc_info']

    for instance_uid, transforms in dicom_results.items():
        max_size_expected = int(max(proc_info[instance_uid]['original_shape']) * max_relative_bbx_size_allowed)
        min_size_expected = int(min(proc_info[instance_uid]['original_shape']) * min_relative_bbx_size_allowed)
        row = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid].iloc[0]
        for transform in transforms:
            bbxs = dicom_results[instance_uid][transform]
            if run_mode not in (const_deploy.RUN_MODE_CADT, const_deploy.RUN_MODE_DEMO):
                assert len(bbxs) <= max_bbxs_displayed_total, \
                    "The image {}/{} contains {} bounding boxes (max allowed: {})".format(
                        instance_uid, transform, len(bbxs), max_bbxs_displayed_total)
            for i in range(len(bbxs)):
                coords = bbxs[i]['coords']
                width = coords[2] - coords[0]
                height = coords[3] - coords[1]
                assert width > 0 and height > 0, "Wrong coordinates for bbx {}/{}/{}: {}".format(
                    instance_uid, transform, i, coords)
                assert min_size_expected <= width <= max_size_expected, "Wrong width for bbx {}/{}/{}: {}".format(
                    instance_uid, transform, i, coords)
                assert min_size_expected <= height <= max_size_expected, "Wrong height for bbx {}/{}/{}: {}".format(
                    instance_uid, transform, i, coords)
                assert 0 < bbxs[i]['score'] < 1, "Wrong score for bbx {}/{}/{}: {}".format(
                    instance_uid, transform, i, bbxs[i]['score'])

                if DicomTypeMap.get_type_row(row) == DicomTypeMap.DBT:
                    # 3D image. Check that the slice number is in the allowed range
                    assert 'num_slices' in proc_info[instance_uid], "Error in bbx {}/{}/{}. Number of slices not found" \
                        .format(instance_uid, transform, i)
                    num_slices = proc_info[instance_uid]['num_slices']
                    assert 0 <= bbxs[i]['slice'] < num_slices, \
                        "Wrong slice number for bbx {}/{}/{}: {}. Total slices: {}" \
                            .format(instance_uid, transform, i, bbxs[i]['slice'], num_slices)


def get_percentile(score, percentile_base):
    """
    calculate linear interpolated percentile score
    Args:
        score: score for which to calculate percentile score
        percentile_base: list of scores used to calculate percentiles

    Returns: float that represent percentile

    """
    if score <= min(percentile_base):
        return 0.0
    if score >= max(percentile_base):
        return 100.

    approx_percentile = stats.percentileofscore(percentile_base, score)
    lower_score = np.percentile(percentile_base, approx_percentile, interpolation='lower')
    higher_score = np.percentile(percentile_base, approx_percentile, interpolation='higher')

    lower_percentage = stats.percentileofscore(percentile_base, lower_score)
    higher_percentage = stats.percentileofscore(percentile_base, higher_score)

    return lower_percentage + (score - lower_score) / (higher_score - lower_score) * (
                higher_percentage - lower_percentage)


def add_percentile_study_scores(model_results, metadata, model_config):
    """
    Adds a percentile score key 'postprocessed_percentile_score' to the passed-in study results dictionary
    :param model_results: dict. A dictionary containing the keys 'dicom_results' and 'study_results'.
    :param metadata: DataFrame. A DataFrame containing the metadata of the files on which the model ran. Used to determine whether the study is 2D or 3D.
    :param model_config: dict. The config of the aggregate model. Used to obtain the 'base' scores that the percentiles are calculated from.
    :return: dictionary (copy of model_results with applied changes)
    """
    model_results_copy = copy.deepcopy(model_results)

    assert 'percentile_scores' in model_config, 'The key "percentile_scores" is missing in the model config'

    study_type = DicomTypeMap.get_study_type(metadata)
    assert study_type in model_config['percentile_scores'], \
        'The key "{}" was not found under the "percentile_scores" key in the model config.'.format(study_type)

    percentile_base = model_config['percentile_scores'][study_type]
    study_results = model_results_copy['study_results']
    study_percentile_score = get_percentile( study_results['total']['score'], percentile_base)
    study_results['total']['postprocessed_percentile_score'] = study_percentile_score
    return model_results_copy


def group_dicoms(dicom_results, metadata, model_results, pixel_data):

    """
    loop through dicom_results and re-organize by foruid and shape
    Args:
        dicom_results: dictionary representing the results of dicoms
        metadata: pandas df, contains meta of the study
        model_results: results of the model
        pixel_data: dictionary of numpy arrays or None

    Returns:
        to_combine: list of foruid that needs to be combined
        groups: dictionary that maps foruid to list of dicoms with that foruid
        dicom_type_map: maps dicom to modality (DBT, DXM)
        foruid_slice_map: dictionary that maps foruid and its corresponding slice_map
        sop_to_orientation: dictionary that maps dicoms to the orientation change needed before combining
        sop_uid_resize_factor: dictioanry that maps dicoms to the size refactor ratio needed before combining

    """
    # map foruid to a list of boxes
    groups = {}
    # map SOPInstanceUID to the dbt or dxm
    dicom_type_map = {}
    # map foruid to the slice_map
    foruid_slice_map = {}
    # map SOPInstanceUID to the orientation changes needed
    sop_to_orientation = {}
    # map SOPInstanceUID to an flaot that specifies the resizing that is needed to transform
    # the shape of the DICOM to match with the rest of the images in its FORUID group
    sop_uid_resize_factor = {}
    # The reference shape for each
    foruid_to_shape = {}

    # loop throught results and update the dictionaries above
    for sop_uid, result in dicom_results.items():
        row = metadata[metadata['SOPInstanceUID'] == sop_uid]
        shape = np.array(row['pixel_array_shape'].values[0][-2:])
        dicom_type = DicomTypeMap.get_type_row(row)
        orientation_change = helper_preprocessor.get_orientation_change(row['ImageLaterality'].values[0],
                                                                        row['PatientOrientation'].values[0])
        sop_to_orientation[sop_uid] = orientation_change
        dicom_type_map[sop_uid] = dicom_type
        foruid = row['FrameOfReferenceUID'].values[0]
        if not isinstance(foruid, str) or foruid == '':
            continue

        # DBT Dicoms we need to keep track of the slice map
        if dicom_type == DicomTypeMap.DBT:
            if len(pixel_data) > 0:  # save to ram
                slice_map = pixel_data[row.index.values[0]]['slice_map']
            else:
                assert row['np_paths'].values[0] == row['np_paths'].values[
                    0], "row['np_paths'] is nan; needs to be valid"
                slice_map = np.load(
                    os.path.join(os.path.dirname(row['np_paths'].values[0]), 'synth/synth_slice_map.npy'))

            assert foruid not in foruid_slice_map, "can't have two slice maps for one FORUID"

            foruid_slice_map[foruid] = [slice_map, 1]

        boxes = copy.deepcopy(result['none'])

        if orientation_change is not None:
            # orient coordinates to move them to the "preprocessed image coordinates space"
            for box in boxes:
                temp_box = copy.deepcopy(box)
                temp_box_coords = orient_bounding_box_coords(temp_box['coords'],
                                                             model_results['proc_info'][sop_uid]['original_shape'],
                                                             orientation_change)
                box['coords'] = temp_box_coords
        assert len(boxes) > 0, "No bboxes found for image with SOPInstanceUID {}".format(sop_uid)

        # make sure coordinates are with respect to the same image shape
        if foruid not in foruid_to_shape:
            foruid_to_shape[foruid] = shape

        # reshape the boxes to match the reference shape
        if not np.array_equal(shape, foruid_to_shape[foruid]):
            factors = foruid_to_shape[foruid] / shape
            assert round(factors[0], 3) == round(factors[1],
                                                 3), "dicom images of foruid  {} do not have the same aspect ratio".format(
                foruid)
            resize_factor = factors[0]
            sop_uid_resize_factor[sop_uid] = resize_factor

            for box in boxes:
                temp_box = copy.deepcopy(box)
                temp_box_coords = [x * resize_factor for x in temp_box['coords']]
                box['coords'] = temp_box_coords
            if dicom_type == DicomTypeMap.DBT:
                foruid_slice_map[foruid][1] = resize_factor

        dicom_box = {
            "sop_uid": sop_uid,
            "boxes": boxes,
            "shape": row['pixel_array_shape'].values[0]
        }

        # put boxes to their respective groups
        if foruid in groups:
            groups[foruid].append(dicom_box)
        else:
            groups[foruid] = [dicom_box]

    # get a list of the FORUIDs that need to be combined

    to_combine = []
    for key, value in groups.items():
        if len(value) > 1:
            to_combine.append(key)

    return to_combine, groups, dicom_type_map, foruid_slice_map, sop_to_orientation, sop_uid_resize_factor


def helper_gather_info_FORUID(boxes, slice_map, slice_map_resize_factor, nms_threshold, dicom_type_map):
    """
    Gather all the information needed to reconstruct boxes for that FORUID
    Args:
        boxes: dictionary of box information for one FORUID
        slice_map: dictionary mapping of slices or None
        slice_map_resize_factor: a float that specify by what factor to resize the slice_map to match the target image shape
        nms_threshold: float
        dicom_type_map: dictionary that maps sop to modality

    Returns: a list of boxes and orientation map

    """
    # these information are needed to reconstruct the original structure of the boxes.
    sop_uids = []
    coords = []
    scores = []
    slices = []
    origins = []
    categories = []

    all_sops_combined = []

    for dicom_box_dictionary in boxes:
        sop = dicom_box_dictionary['sop_uid']
        all_sops_combined.append(sop)
        for box in dicom_box_dictionary["boxes"]:
            dicom_coords = box['coords']
            dicom_score = box['score']

            if dicom_type_map[sop] == DicomTypeMap.DXM:
                if slice_map is None:
                    slice_num = -1
                else:
                    # also make sure the slice_map is in the correct shape
                    x1, y1, x2, y2 = dicom_coords[0] / slice_map_resize_factor, dicom_coords[1] / slice_map_resize_factor, \
                                     dicom_coords[2] / slice_map_resize_factor, dicom_coords[3] / slice_map_resize_factor
                    slice_num = stats.mode(slice_map[int(y1):int(y2), int(x1):int(x2)], axis=None)[0][0]
            else:
                slice_num = box['slice']


            coords.append(list(dicom_coords))
            scores.append(float(dicom_score))
            slices.append(int(slice_num))
            categories.append(box['category'])
            origins.append(dicom_type_map[sop])
            sop_uids.append(sop)

    result_boxes = []
    coords = np.array(coords)
    scores = np.array(scores)
    slices = np.array(slices)
    categories = np.array(categories)
    origins = np.array(origins)
    nms_idx = nms(coords, scores, iou_threshold=nms_threshold, top_k=100)

    # reformatting of the boxes to the original structure
    for idx in range(len(coords[nms_idx])):
        result_boxes.append({'coords': coords[nms_idx][idx].tolist(),
                             'score': scores[nms_idx][idx],
                             # 'category': get_fourway_category(scores[nms_idx][idx], case_thresholds),
                             'category': int(categories[nms_idx][idx]),
                             'slice': int(slices[nms_idx][idx]),
                             'origin': origins[nms_idx][idx],
                             # "sop_uid_origin": sop_uids[nms_idx][idx]

                             })

    return result_boxes, all_sops_combined


def combine_foruid_boxes(model_results, metadata, model_config, pixel_data=[], logger=None):
    if logger is None:
        logger = logging.getLogger()
    model_results_copy = copy.deepcopy(model_results)
    nms_threshold = model_config['combined_nms_threshold']
    dicom_results = copy.deepcopy(model_results['dicom_results'])

    # group all by FORUID
    to_combine, groups, dicom_type_map, foruid_slice_map, sop_to_orientation, sop_uid_resize_factor = \
        group_dicoms(dicom_results, metadata, model_results_copy, pixel_data)

    for foruid in to_combine:

        # resizing of the slice map
        if foruid in foruid_slice_map:
            slice_map, resize_factor = foruid_slice_map[foruid][0], foruid_slice_map[foruid][1]
        else:
            slice_map, resize_factor = None, None

        boxes = groups[foruid]
        result_boxes, all_sops_combined = helper_gather_info_FORUID(boxes, slice_map, resize_factor, nms_threshold,
                                                                  dicom_type_map)
        # combine
        for sop_uid in all_sops_combined:
            temp_boxes = copy.deepcopy(result_boxes)
            orientation_change = sop_to_orientation[sop_uid]

            # orientation and resize box coordinates for each dicom
            for box in temp_boxes:
                temp_box = copy.deepcopy(box)
                temp_box_coords = copy.deepcopy(temp_box['coords'])
                if orientation_change is not None:
                    temp_box_coords = orient_bounding_box_coords(temp_box_coords,
                                                                 model_results['proc_info'][sop_uid][
                                                                     'original_shape'], orientation_change)
                if sop_uid in sop_uid_resize_factor:
                    resize_factor = sop_uid_resize_factor[sop_uid]
                    temp_box_coords = [x / resize_factor for x in temp_box_coords]
                box['coords'] = temp_box_coords

            model_results_copy['dicom_results'][sop_uid]['none'] = temp_boxes
    return model_results_copy
