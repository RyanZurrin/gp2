import pandas as pd
import pdb
import sys
import traceback
import os
import pydicom
import ast

# from dicom_constants import ALT_TAGS
if sys.version[0] == '2':  # Python 2.X
    # from utils import get_dcm_attribute, nested_index
    from format_helpers import map_manufacturer, map_manufacturer_model_for_specs
elif sys.version[0] == '3':  # Python 3.X
    # from .utils import get_dcm_attribute, nested_index
    from deephealth_utils.data.format_helpers import map_manufacturer, map_manufacturer_model_for_specs
    from deephealth_utils.data.dicom_constants import ALT_TAGS

VALIDATED_MANUFACTURERS = ['hologic', 'ge']

VALIDATION_RULES = ['EQUAL', 'NOT_DIFF', 'IN', 'NOT_EQUAL', 'RANGE', 'REQUIRED',
                    'N/A', 'CONTAINS', 'LIST_CONTAINS', 'NOT_CONTAINS', 'EMPTY_SEQ', 'NOT_EXIST']


def validate_dicom_using_specs(ds, specs_name=None, specs_dict=None, raise_exception_if_false=True):
    manufacturer = getattr(ds, 'Manufacturer', 'MANUFACTURER_MISSING')
    if specs_name is None:
        specs_name = map_manufacturer(manufacturer)
    if specs_name != 'Common' and specs_name not in VALIDATED_MANUFACTURERS:
        reason = ('Manufacturer', 'Not Validated', 'N/A', manufacturer, 'FAC-140')
        if raise_exception_if_false:
            raise Exception(reason)
        else:
            return False, [reason]

    if specs_dict is None or specs_name not in specs_dict:
        specs_df = read_dicom_specs(specs_name)
    else:
        specs_df = specs_dict[specs_name]

    if specs_name != 'Common':
        sop = getattr(ds, 'SOPClassUID', '')
        manufacturer_model_name = getattr(ds, 'ManufacturerModelName', '')
        model_name = map_manufacturer_model_for_specs(manufacturer_model_name)
        key = model_name + '_' + sop
        reason = []

        # accepted_manufacturer_model_names = list(set([v.split('_')[0] for v in specs_df.columns if '_' in v]))
        # if model_name not in accepted_manufacturer_model_names:
        #     reason.append(('ManufacturerModelName+SOP', 'Not Validated', 'N/A', manufacturer_model_name, 'FAC-141'))
        # if key not in specs_df.columns:
        #     reason.append(('ManufacturerModelName+SOP', 'Not Validated', 'N/A',
        #                    '{}_{}'.format(manufacturer_model_name, sop), 'FAC-142'))

        if key not in specs_df.columns:
            new_key = None
            for c in specs_df.columns:
                if "_" not in c:
                    continue
                column_sop = c.split("_")[1]
                if column_sop == sop:
                    new_key = c
                    break
            if new_key is None:
                reason.append(('SOPClassUID', 'Not Validated', 'N/A', sop, 'FAC-20'))
            else:
                key = new_key

        if len(reason) > 0:
            if raise_exception_if_false:
                raise Exception(reason)
            else:
                return False, reason
    else:
        key = 'Common'

    unique_rules = specs_df[key + '-Rule'].unique()
    for r in unique_rules:
        if r not in VALIDATION_RULES:
            raise ValueError('Unrecognized Rule: ' + str(r))

    reasons = []

    for i, row in specs_df.iterrows():

        rule_name = row[key + '-Rule']
        rule_target = row[key]
        if rule_name in ['N/A'] or rule_target in ['-', 'N/A']:
            continue

        val = ds.dh_getattribute(ast.literal_eval(row['Tag']))

        if val is None:
            if key in ALT_TAGS:
                if row['TagName'] in ALT_TAGS[key]:
                    val = ds.dh_getattribute(row['TagName'])

        meets_rule = True

        if val is None:
            if rule_name in ['NOT_DIFF', 'NOT_EQUAL', 'NOT_CONTAINS', 'EMPTY_SEQ', 'NOT_EXIST']:
                continue
            else:
                meets_rule = False

        else:  # val is not None
            if 'multival' in str(type(val)) and 'LIST' not in rule_name and 'RANGE' != rule_name:
                val = str(val)

            if rule_name == 'REQUIRED':
                if val == '' or val != val:  # if empty string or nan value
                    meets_rule = False
            elif rule_name == 'EMPTY_SEQ':
                if val == pydicom.sequence.Sequence():
                    continue
                else:
                    meets_rule = False
            elif rule_name == 'CONTAINS':
                if rule_target not in str(val):
                    meets_rule = False
            elif rule_name == 'LIST_CONTAINS':
                if rule_target not in val:
                    meets_rule = False
            elif rule_name == 'NOT_CONTAINS':
                if '|' in str(rule_target):
                    meets_rule = not any([t.lower() in str(val).lower() for t in rule_target.split('|')])
                elif rule_target.lower() in str(val).lower():
                    meets_rule = False
            elif rule_name == 'RANGE':
                if val == '' or val != val:  # if empty string or nan value
                    meets_rule = False
                elif 'multival' in str(type(val)):
                    meets_rule = False
                else:
                    low_bound = float(rule_target[rule_target.find('[') + 1:rule_target.find(',')])
                    up_bound = float(rule_target[rule_target.find(',') + 1:rule_target.find(']')])

                    val = float(val)
                    if val < low_bound or val > up_bound:
                        meets_rule = False
            else:
                # need to process_target
                if rule_name == 'IN' or (rule_name == 'NOT_DIFF' and '|' in rule_target):
                    options = rule_target.split('|')
                    target = [process_target(v) for v in options]
                else:
                    target = process_target(rule_target)
                if rule_name == 'IN' or (rule_name == 'NOT_DIFF' and '|' in rule_target):
                    if val not in target:
                        meets_rule = False
                elif rule_name == 'NOT_EQUAL':
                    if type(target) == str:
                        target = target.lower()
                    if type(val) == str:
                        val = val.lower()
                    if val == target:
                        meets_rule = False

                elif val != target:  # EQUAL or NOT_DIFF
                    meets_rule = False

        if not meets_rule:
            r_tup = (row['TagName'], rule_name, rule_target, val, row['AcceptanceCriteria'])
            reasons.append(r_tup)

    is_good = len(reasons) == 0

    if raise_exception_if_false and not is_good:
        raise(Exception(reasons[0]))

    return is_good, reasons


def validate_preprocessing_specs(ds, raise_exception_if_false=True):
    '''
    Validate the parameters used in processing the pixel data (e.g. WindowCenter and WindowWidth) are reasonable.
    Logic is implemented below by manufacturer, manufacturer model, and SOP Class
    '''
    reasons = []

    manufacturer = map_manufacturer(ds.Manufacturer)
    if manufacturer not in VALIDATED_MANUFACTURERS:
        is_validated = False
        reasons.append(('Manufacturer', 'Not Validated', 'N/A', ds.Manufacturer))
    else:
        sop = ds.SOPClassUID
        model_name = map_manufacturer_model_for_specs(ds.ManufacturerModelName)
        key = model_name + '_' + sop
        if manufacturer == 'ge':
            if key == 'SenographeEssential_1.2.840.10008.5.1.4.1.1.1.2':
                explanations = list(
                    ds.dh_getattribute('WindowCenterWidthExplanation'))  # looks like ['NORMAL', 'HARDER', 'SOFTER']

                if 'NORMAL' not in explanations:
                    is_validated = False
                    reasons.append(('WindowCenterWidthExplanation', 'CONTAINS', 'NORMAL', str(explanations)))
                else:
                    normal_index = explanations.index('NORMAL')
                    center = float(ds.dh_getattribute('WindowCenter')[normal_index])

                    high_bit = ds.dh_getattribute('HighBit')
                    if high_bit == 11:
                        if center > 3300:  # two std above median in AST dataset
                            is_validated = False
                            reasons.append(('WindowCenter', 'Too High', 3300, center))
                        elif center < 2241:  # two std below median in AST dataset
                            is_validated = False
                            reasons.append(('WindowCenter', 'Too Low', 2241, center))
                        else:
                            is_validated = True

                    else:
                        is_validated = False
                        reasons.append(('HighBit', 'Rule not set up for HighBit for ' + key, 'N/A', high_bit))
            else:
                is_validated = False
                reasons.append(('ManufacturerModelName_SOPClassUID', 'Not Validated', 'N/A', key))

        elif manufacturer == 'hologic':
            if key == 'SeleniaDimensions_1.2.840.10008.5.1.4.1.1.1.2':
                center = float(ds.dh_getattribute('WindowCenter'))
                if center < 530:
                    is_validated = False
                    reasons.append(('WindowCenter', 'Too Low', 530, center))
                elif center > 2100:
                    is_validated = False
                    reasons.append(('WindowCenter', 'Too High', 2100, center))
                else:
                    is_validated = True  # currently no restrictions on WindowCenter or WindowWidth
            elif key == 'SeleniaDimensions_1.2.840.10008.5.1.4.1.1.7':
                center = float(ds.dh_getattribute('WindowCenter'))
                if center < 512:
                    is_validated = False
                    reasons.append(('WindowCenter', 'Too Low', 512, center))
                else:
                    is_validated = True
            elif key == 'SeleniaDimensions_1.2.840.10008.5.1.4.1.1.13.1.3':
                center = float(ds.dh_getattribute('WindowCenter'))
                if center < 512:
                    is_validated = False
                    reasons.append(('WindowCenter', 'Too Low', 512, center))
                else:
                    is_validated = True
            else:
                is_validated = False
                reasons.append(('ManufacturerModelName_SOPClassUID', 'Not Validated', 'N/A', key))

    if raise_exception_if_false and not is_validated:
        raise(Exception(reasons[0]))

    return is_validated, reasons


def validate_dcm_array(ds, X, raise_exception=True):
    '''
    Validate that array X has dimensions matching those based off of dicom ds
    '''
    is_good = True
    reason = None
    if ds.Rows != X.shape[0]:
        is_good = False
        reason = "Shape[0] of array doesn't match Rows attribute"
    elif ds.Columns != X.shape[1]:
        is_good = False
        reason = "Shape[1] of array doesn't match Rows attribute"
    elif X.ndim == 3 and X.shape[2] != ds.NumberofFrames:
        is_good = False
        reason = "Shape[2] of array doesn't match NumberofFrames attribute"

    if raise_exception and not is_good:
        raise(Exception(reason))

    return is_good, reason


def read_dicom_specs(manufacturer, raise_exception=True):
    f_name = os.path.abspath(os.path.dirname(__file__)) + '/dicom_specs/' + manufacturer.lower() + '.csv'
    # if not os.path.isfile(f_name):
    #     if raise_exception:
    #         raise ValueError("Don't have dicom_specs for " + manufacturer)
    #     else:
    #         return None
    # else:
    df = pd.read_csv(f_name, dtype={'TagGroup': str, 'TagElement': str}, index_col=False, keep_default_na=False)
    return df


def process_target(target):
    if target.isdigit():
        target = int(target)
    elif target.replace('"', '').isdigit():  # keep as string
        target = target.replace('"', '')
    else:
        try:
            target = float(target)
        except:
            pass
    return target


def validate_dicom_specs_df(df):
    for idx, row in df.iterrows():
        for c in ['TagGroup', 'TagElement']:
            assert len(row[c]) == 4, 'Length of ' + c + ' is ' + str(len(row[c])) + ': ' + row[c]


if __name__ == '__main__':
    try:
        from utils import dh_dcmread

        study_path = sys.argv[1]
        filename = os.path.join(study_path, os.listdir(study_path)[1])
        ds = dh_dcmread(filename)
        is_good, reasons = validate_dicom_using_specs(ds)
        pdb.set_trace()
    except:
        ty, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
