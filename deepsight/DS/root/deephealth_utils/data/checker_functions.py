import datetime
import os
import numpy as np
import pydicom
import re
import json
from fractions import Fraction

from .dicom_type_helpers import DicomTypeMap
from . import format_helpers as format_helpers
from . import validation as dh_val
from .format_helpers import map_manufacturer, map_manufacturer_model_for_specs

REGEXS = {'uid': '(?=.{1,64}$)^[0-9]+(.[0-9]+)*$',
          'PatientAge': '^[0-9]{3}[DMY]$',
          'date': '^\d{4}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])$'}

checker_config_path = os.path.abspath(os.path.dirname(__file__)) + '/dicom_specs/checker_config.json'
with open(checker_config_path) as f:
    CHECKER_CONFIG = json.load(f)

def patient_age_not_less_than(df, max_age=int(CHECKER_CONFIG['max_age'])):

    d = {}
    for field in ['PatientAge', 'PatientBirthDate', 'StudyDate']:
        if field in df:
            d[field] = getattr(df, field).iloc[0]
        else:
            d[field] = ''
    years = None
    years_patientage = None
    years_birthdate = None
    if re.compile(REGEXS['PatientAge']).match(d['PatientAge']):
        unit_dict = {'D': 365, 'M': 12, 'Y': 1}
        years = float(d['PatientAge'][0:3]) / unit_dict[d['PatientAge'][3]]
        years_patientage = years
    if all([re.compile(REGEXS['date']).match(date) for date in [d['PatientBirthDate'], d['StudyDate']]]):
        study = datetime.datetime.strptime(d['StudyDate'], '%Y%m%d')
        born = datetime.datetime.strptime(d['PatientBirthDate'], '%Y%m%d')
        years = study.year - born.year - ((study.month, study.day) < (born.month, born.day))
        years_birthdate = years
    if years_patientage is not None and years_birthdate is not None:
        if years_patientage == years_birthdate:
            return years_patientage >= max_age, years_patientage
        else:
            return False, 'PatientAge: {}, AgeFromBirthDate: {}'.format(years_patientage, years_birthdate)
    elif years_patientage is not None:
        return years_patientage >= max_age, years_patientage
    elif years_birthdate is not None:
        return years_birthdate >= max_age, years_birthdate
    else:
        return True, years


def check_laterality(ds):
    lat = ds.dh_getattribute('ImageLaterality')
    if lat is not None:
        if lat in ['L', 'R']:
            return True, lat
    return False, lat

def no_additional_views(df):
    other_views = []
    for view in df['ViewPosition'].values:
        if view not in ['CC', 'MLO']:
            other_views.append(view)
    if len(other_views) > 0:
        return False, other_views
    else:
        return True, None


def check_patient_orientation(ds):
    """
    Checks if patient orientation make sense.
    The first should be A,P and second value should be [R,L,FL,FR,HL,HR]
    Args:
        ds: pydicom dataset

    Returns: (bool, string): bool saying if its one of the accepted orientations,
    and string representing the orientation

    """
    po = ds.dh_getattribute('PatientOrientation')

    if po is not None:
        po = list(po)
        if len(po) < 2:
            return False, po
        if po[0] in ["A","P"] and po[1] in ["R","L","FL","FR","HL","HR","F"]:
            return True, po
    return False, po


def pixel_intensity_values_in_range(ds, high_threshold_fraction=CHECKER_CONFIG['high_threshold_fraction']):

    px_array = getattr(ds, 'pixel_array', '')
    high_bit = getattr(ds, 'HighBit', '')

    if px_array == '':
        return False, 'pixel_array not found'
    if high_bit == '':
        return False, 'Attribute HighBit not found'

    highest_value = (2 ** (high_bit + 1) - 1)

    if px_array.min() < 0:
        return False, 'Min lower than 0 | Min: {}, Max: {}, High Bit: {}'\
            .format(px_array.min(), px_array.max(), high_bit)
    if px_array.max() < highest_value * float(Fraction(high_threshold_fraction)):
        return False, 'Max lower than threshold| Min: {}, Max: {}, High Bit: {}'\
            .format(px_array.min(), px_array.max(), high_bit)

    return True, None


def check_fields(ds):
    _, common_reasons = dh_val.validate_dicom_using_specs(ds, specs_name='Common', raise_exception_if_false=False)
    _, manu_reasons = dh_val.validate_dicom_using_specs(ds, raise_exception_if_false=False)

    reasons = common_reasons + manu_reasons

    checks_failed = {}
    for r in reasons:
        field = r[3]
        acceptance_criteria = r[4]
        if type(field) == pydicom.Sequence:
            field = re.sub(' +', ' ', str(field))
        checks_failed[acceptance_criteria] = field

    # Run the GE windowing check separately because we need explanation to access correct index
    invalid_fields, acceptance_criterias = window_center_width_in_range_with_explanation(ds)
    if len(acceptance_criterias) > 0:
        for i, acceptance_criteria in enumerate(acceptance_criterias):
            checks_failed[acceptance_criteria] = invalid_fields[i]

    return checks_failed


def viewlat_exists(df, viewlat):
    for idx, row in df.iterrows():
        if row['ImageLaterality'] == viewlat[0] and row['ViewPosition'] == viewlat[1:]:
            return True, None
    return False, None


def field_consistent(df, field):
    df = df.replace('', np.nan, inplace=False)
    return df[field].nunique(dropna=False) == 1, field + ': ' + str(tuple(df[field]))

def study_size_check(file_list, max_size=CHECKER_CONFIG['max_mb_study']):
    file_size = sum(os.path.getsize(f) for f in file_list) * 0.000001
    return file_size <= max_size, file_size


def file_size_check(filename, max_size=CHECKER_CONFIG['max_mb_file']):
    file_size = os.path.getsize(filename) * 0.000001
    return file_size <= max_size, file_size

def transfer_syntax_valid(ds, accepted_uids=CHECKER_CONFIG['accepted_transfer_syntax_uids']):

    accepted_uids = [getattr(pydicom.uid, uid) for uid in accepted_uids]
    file_meta = getattr(ds, 'file_meta', '')
    if file_meta != '':
        transfer_syntax_uid = getattr(file_meta, 'TransferSyntaxUID', '')
        if transfer_syntax_uid != '':
            return transfer_syntax_uid in accepted_uids, transfer_syntax_uid
    return False, None


def manufacturer_supported(ds, supported_manufacturers=CHECKER_CONFIG['supported_manufacturers']):
    manufacturer = format_helpers.map_manufacturer(getattr(ds, 'Manufacturer', ''))
    return manufacturer in supported_manufacturers, manufacturer


def studyinstanceuid_valid(ds):
    studyinstanceuid = getattr(ds, 'StudyInstanceUID', '')
    return re.compile(REGEXS['uid']).match(studyinstanceuid), studyinstanceuid


def bt_has_corresponding_2d(df):
    '''
    bt_latviews.difference(non_bt_latviews) returns latviews in BT that are not found in non_bt latviews
    '''
    bts = df[df.apply(DicomTypeMap.get_type_row, axis=1) == DicomTypeMap.DBT]
    if bts.empty:
        return True, None
    non_bts = df[df.apply(DicomTypeMap.get_type_row, axis=1) == DicomTypeMap.DXM]
    bt_latviews = set(bts['ImageLaterality'] + bts['ViewPosition'])
    non_bt_latviews = set(non_bts['ImageLaterality'] + non_bts['ViewPosition'])
    return bt_latviews.issubset(non_bt_latviews), str(tuple(bt_latviews.difference(non_bt_latviews)))

def dxm_has_corresponding_dbt(metadata_df):
    """
    If the study is DBT, each Dxm should have its corresponding DBT
    :param df: Dataframe. Metadata
    :return: True/False, list-str with wrong files
    """
    errors = []
    is_dbt = len(metadata_df[metadata_df.apply(DicomTypeMap.get_type_row, axis=1) == DicomTypeMap.DBT]) > 0
    if is_dbt:
        dxm_images =  metadata_df[metadata_df.apply(DicomTypeMap.get_type_row, axis=1) == DicomTypeMap.DXM]
        for ix, row in dxm_images.iterrows():
            # Search for matching Dxm
            if len(metadata_df.loc[(metadata_df.apply(DicomTypeMap.get_type_row, axis=1) == DicomTypeMap.DBT)
                                   & (metadata_df['ImageLaterality'] == row['ImageLaterality'])
                                   & (metadata_df['ViewPosition'] == row['ViewPosition'])]) == 0:
                errors.append("DBT-{}-{}".format(row['ImageLaterality'], row['ViewPosition']))

        if len(errors) == 0:
            return True, None
        return False, errors
    # Dxm only
    return True, None


def consistent_2D_3D(metadata_df):
    """
    If DBT images are present, they have to be consistent for each laterality, meaning
    we either have both views or none
    :param metadata_df: metadata dataframe
    :return: True/False, list-str with wrong lateralities
    """
    errors = []
    is_dbt = len(metadata_df.loc[metadata_df['SOPClassUID'] == '1.2.840.10008.5.1.4.1.1.13.1.3']) > 0
    if is_dbt:
        for laterality in ('L', 'R'):
            # If there is DBT for a view in one laterality, there must be for the other view too
            dbt_cc = len(metadata_df.loc[(metadata_df['SOPClassUID'] == '1.2.840.10008.5.1.4.1.1.13.1.3')
                                         & (metadata_df['ImageLaterality'] == laterality)
                                         & (metadata_df['ViewPosition'] == 'CC')]) > 0
            dbt_mlo = len(metadata_df.loc[(metadata_df['SOPClassUID'] == '1.2.840.10008.5.1.4.1.1.13.1.3')
                                          & (metadata_df['ImageLaterality'] == laterality)
                                          & (metadata_df['ViewPosition'] == 'MLO')]) > 0
            if dbt_cc != dbt_mlo:
                errors.append("DBT-{}".format(laterality))
        if len(errors) == 0:
            return True, None
        return False, errors
    # Dxm only
    return True, None


def window_center_width_in_range_with_explanation(ds, center_range=CHECKER_CONFIG['center_range'],
                                                  width_range=CHECKER_CONFIG['width_range'],
                                                  prefix=CHECKER_CONFIG['check_prefix']):

    '''
    Implements XX-160, XX-170, XX-80 checks if
    1. WindowCenter, WindowWidth and WindowCenterWindowWidthExplanation values are valid(XX_80)
    2. WindowCenter(XX-160) and WindowWidth(XX-180) are within range

    Implementation logic
    1. Checks if the Manufacture is as expected if not it should fail FAC-140 and we bypass this check
    3. Skips  secondary capture files
    2. If Manufacture is as expected but we are missing either manufacturer_model_name or sop_class_uid we raise XX-80
    3. If WindowCenterWindowWidthExplanation exist then :
        3.1  Check all three are multivalue arrays of the same length
        3.2  Check that 'NORMAL' exists in the Explanation multivalue array
        3.3 Use the corresponding index to find WindowCenter, WindowWidth.
        3.4 Check that WindowCenter, WindowWidth each fall within range.

    4. If WindowCenterWindowWidthExplanation doesn't exist
        4.1 checks if the WindowCenter WindowWidth exists and not an array and within the corresponding range


    Args:
        ds (pydicom.dataset.Dataset): Dataset represent the dicom
        center_range: dictionary that include the valid WindowCenter ranges for each manufacturer
        width_range: dictionary that include the valid WindowWidth ranges for each manufacturer
        prefix: dictionary that maps each manufacture to a check prefix

    Returns: tuple of two lists the first of which is invalid values and the later is a list of failed checks

    '''
    assert center_range.keys() == width_range.keys(), 'center_range and width_range should have the same set of keys'

    invalid_values = []
    failed_checks = []

    #### Seeing if check applies to this file and if we have all the information need to perform the check
    manufacturer = map_manufacturer(getattr(ds, 'Manufacturer', ''))
    manufacturer_model_name = map_manufacturer_model_for_specs(getattr(ds, 'ManufacturerModelName', ''))
    sop_class_uid = getattr(ds, 'SOPClassUID', '')

    # Skips  secondary capture files
    if '1.2.840.10008.5.1.4.1.1.7' in sop_class_uid:
        return invalid_values, failed_checks

    # If we dont know the manufacture we skip these function entirely as the file would fail the FAC-140
    if manufacturer not in center_range.keys():
        return invalid_values, failed_checks
    else:
        check_prefix = prefix[manufacturer]

    # if missing either manufacturer_model_name or sop_class_uid we raise XX-160 and XX-170
    if None in [manufacturer_model_name, sop_class_uid] or '' in [manufacturer_model_name, sop_class_uid]:
        invalid_values.append('ManufacturerModelName, and SOPClassUID required for WindowCenter WindowWidth check: {}'.format(
            [manufacturer_model_name, sop_class_uid]))
        failed_checks.append(check_prefix + '80')
        return invalid_values, failed_checks

    # If GE file contains unsupported ManufacturerModelName, fail the check.
    if manufacturer == 'ge':
        if manufacturer_model_name not in ['SenographeEssential', 'VolumePreview']:
            invalid_values.append(
                'File is GE but does not have supported ManufacturerModelName: {}'.format(
                    manufacturer_model_name))
            failed_checks.extend('GE-80')
            return invalid_values, failed_checks


    ### Comparing values to the specified range
    window_center = ds.dh_getattribute('WindowCenter')
    window_width = ds.dh_getattribute('WindowWidth')
    explanation = ds.dh_getattribute('WindowCenterWidthExplanation')

    center_range = center_range[manufacturer]
    width_range = width_range[manufacturer]


    # checking for WindowCenterWidthExplanation
    if explanation in [None, '']:
        is_array = False

    elif isinstance(explanation, pydicom.multival.MultiValue):
        is_array = True
        explanation = [str(x).lower() for x in explanation]
        if 'normal' not in explanation:
            invalid_values.append('\'NORMAL\' needs to be in '
                                  'WindowCenterWidthExplanation to check range')
            failed_checks.append(check_prefix + '80')

            return invalid_values, failed_checks

    else:
        invalid_values.append('invalid value for WindowCenterWidthExplanation : {}'.format(explanation))
        failed_checks.append(check_prefix + '80')
        return invalid_values, failed_checks

    if is_array:

        # not all [window_center, window_width, explanation] array like
        if not (isinstance(window_center, pydicom.multival.MultiValue) or isinstance(window_center, list)) or \
                not (isinstance(window_width, pydicom.multival.MultiValue) or isinstance(window_width, list)):

            invalid_values.append('WindowCenter, or WindowWidth is not  multival array when '
                                  'WindowCenterWidthExplation is an array: {}'
                                  .format(str([window_center, window_width, explanation])))
            failed_checks.append( check_prefix + '80')
            return invalid_values, failed_checks

        #all [window_center, window_width, explanation] array like
        else:
            lengths = [len(arr) for arr in [window_center, window_width, explanation]]
            # all [window_center, window_width, explanation] array like but not same length
            if len(set(lengths)) != 1:
                invalid_values.append('Explanation, center, or width not multival arrays of equal length: {}'
                                      .format(str([window_center, window_width, explanation])))
                failed_checks.append(check_prefix + '80')
                return invalid_values, failed_checks

            # all [window_center, window_width, explanation] array like but not same length
            normal_idx = list(explanation).index('normal')
            window_center = window_center[normal_idx]
            window_width = window_width[normal_idx]


    # at this point window_center window_center should no longer be arrays

    try:
        window_center = int(window_center)
    except:
        invalid_values.append("Invalid value for WindowCenter {}".format(window_center))
        failed_checks.append(check_prefix + '80')


    try:
        window_width = int(window_width)
    except:
        invalid_values.append("Invalid value for WindowWidth {}".format(window_width))
        failed_checks.append(check_prefix + '80')

    if isinstance(window_center, int):
        if window_center < center_range[0] or window_center > center_range[1]:
            invalid_values.append(window_center), failed_checks.append(check_prefix + '160')

    if isinstance(window_width, int):
        if window_width < width_range[0] or window_width > width_range[1]:
            invalid_values.append(window_width), failed_checks.append(check_prefix + '170')

    return invalid_values, failed_checks


def view_code_sequence_length_1(ds):
    view_code_sequence = getattr(ds, 'ViewCodeSequence', '')
    if view_code_sequence == '':
        return False, None
    else:
        return len(view_code_sequence) == 1, str(view_code_sequence)

def breast_orientation_expected(ds, columns_to_check=CHECKER_CONFIG['columns_to_check']):

    view = ds.dh_getattribute('ViewPosition')
    patient_orientation = ds.dh_getattribute('PatientOrientation')
    laterality = ds.dh_getattribute('ImageLaterality')
    SOPClassUID = ds.dh_getattribute('SOPClassUID')
    pixel_data = ds.dh_getattribute('pixel_array')

    if pixel_data is None or laterality is None or SOPClassUID is None or patient_orientation is None or view is None:
        # have to compare individually since comparing pixel_data within a list causes ambiguous error.
        return False, None

    if view != "MLO":
        # We only care about MLO, the model should be able to handle CC images even if they are flipped
        return True, None
    
    if len(pixel_data.shape) == 3:
        # Make sure that 0th dimension is the min dimension
        if np.argmin(pixel_data.shape) != 0:
            return False, 'pixel_data shape is not channel first: {}'.format(pixel_data.shape)
        # 3D image. Take the middle slice
        # First, ensure the first dimension is the slice
        assert min(pixel_data.shape) == pixel_data.shape[0], \
            "Unexpected shape {}. The first dimension should be the number of slices".format(pixel_data.shape)
        pixel_data = pixel_data[pixel_data.shape[0] // 2]

    # left_mean = pixel_data[:, 0:columns_to_check].mean()
    # right_mean = pixel_data[:, -1 * columns_to_check:].mean()
    # 
    # if SOPClassUID == '1.2.840.10008.5.1.4.1.1.13.1.3':
    #     if left_mean > right_mean:
    #         return False, None
    # else:
    #     if laterality == 'L':
    #         if left_mean < right_mean:
    #             return False, None
    #     elif laterality == 'R':
    #         if left_mean > right_mean:
    #             return False, None
    #     else:
    #         return False, 'Laterality not in [\'L\', \'R\']: {}'.format(laterality)
    # 
    # return True, None
    
    # quads: 0 1
    #        2 3
    quads = [pixel_data[:pixel_data.shape[0] // 2, :pixel_data.shape[1] // 2],
             pixel_data[:pixel_data.shape[0] // 2, pixel_data.shape[1] // 2:],
             pixel_data[pixel_data.shape[0] // 2:, :pixel_data.shape[1] // 2],
             pixel_data[pixel_data.shape[0] // 2:, pixel_data.shape[1] // 2:]]
    if laterality == 'L' and "A" in patient_orientation:
        # High: 0;
        quad_high = quads[0]
    elif laterality == 'L' and "P" in patient_orientation:
        # High: 3
        quad_high = quads[3]
    elif laterality == 'R' and "P" in patient_orientation:
        # High: 1
        quad_high = quads[1]
    elif laterality == 'R' and "A" in patient_orientation:
        # High: 2
        quad_high = quads[2]
    else:
        return False, "Wrong laterality/orientation: {}/{}".format(laterality, patient_orientation)

    # One quadrant should have a higher intensity than the mean
    ok = quad_high.mean() > pixel_data.mean()
    return ok, None

def pixel_data_exists(ds):
    return hasattr(ds, 'PixelData'), None


def check_image_classification(ds, exclude=CHECKER_CONFIG['image_classification_exclude']):
    '''
    checks if to exclude this file because it one of the ImageClassification listed in exclude
    Args:
        ds: Dataset represent a dicom file
        exclude: list of string that specifies which ImageClassification to exclude

    Returns: bool, invalid values

    '''

    classification = DicomTypeMap.get_image_classification(ds)
    if classification.value in exclude:
        return False, classification
    return True, None

class CheckFnSelector(object):
    """
    CheckFnSelector maps a function string to function
    """
    fn_dict = {
        'patient_age_not_less_than': patient_age_not_less_than,
        'check_laterality': check_laterality,
        'pixel_intensity_values_in_range': pixel_intensity_values_in_range,
        'check_fields': check_fields,
        'no_additional_views': no_additional_views,
        'viewlat_exists': viewlat_exists,
        'field_consistent': field_consistent,
        'study_size_check': study_size_check,
        'file_size_check': file_size_check,
        'transfer_syntax_valid': transfer_syntax_valid,
        'manufacturer_supported': manufacturer_supported,
        'studyinstanceuid_valid': studyinstanceuid_valid,
        'bt_has_corresponding_2d': bt_has_corresponding_2d,
        'dxm_has_corresponding_dbt': dxm_has_corresponding_dbt,
        'consistent_2D_3D': consistent_2D_3D,
        'view_code_sequence_length_1': view_code_sequence_length_1,
        'breast_orientation_expected': breast_orientation_expected,
        'check_patient_orientation': check_patient_orientation,
        'check_image_classification': check_image_classification
    }

    @staticmethod
    def get_fn(fn_str):
        if fn_str in CheckFnSelector.fn_dict:
            return CheckFnSelector.fn_dict[fn_str]
        else:
            raise NotImplementedError("Check function name \'{}\' is not implemented".format(fn_str))
