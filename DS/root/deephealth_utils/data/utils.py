import glob
import logging
import os
import sys
import numpy as np
from functools import partial
import psutil
import pydicom
from pydicom.tag import Tag
import warnings
import ast
import re
import pandas as pd

from deephealth_utils.data.dicom_constants import ALT_TAGS
from deephealth_utils.data.format_helpers import map_manufacturer_model_for_specs, format_dicom_tag_str, \
    map_manufacturer, verbose_position_to_code
from deephealth_utils.data.dicom_utils import determine_if_hologic_sco
from deephealth_utils.misc.gen_helpers import is_str_type

if sys.version[0] == '3':  # Python 3.X
    unicode = type(None)
try:
    from decode_hologic_sco.convert import extract_hologic_sc_pixels
except ImportError as e:
    warnings.warn("{}".format(e))


def window_im(x, center, width, ymin=0., ymax=65535., return_type='uint16'):
    """Windows a numpy array.
    Note: Based off of https://www.dabsoft.ch/dicom/3/C.11.2.1.2/
    Args:
        x (np.ndarray): Numpy array containing the pixel_array from a DICOM file.
        center (int):
        width (int):
        ymin (float):
        ymax (float):
    """

    assert type(x) is np.ndarray, 'Input array is not a numpy array'
    assert type(center) in [int, float], '"center" is not numeric type'
    assert type(width) in [int, float], '"width" is not numeric type'

    y = ((x - (center - 0.5)) / (width - 1) + 0.5) * (ymax - ymin) + ymin
    np.clip(y, ymin, ymax, out=y)
    if 'int' in return_type:
        y = np.round(y)
    if return_type != str(y.dtype):
        return y.astype(return_type)
    return y


def window_im_sigmoid(x, center, width, ymin=0., ymax=65535., return_type='uint16'):
    """Windows a numpy array using sigmoid function.
    Note: Based off of http://dicom.nema.org/medical/dicom/2017a/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.3.1
    Args: See function window_im
    Returns: See function window_im
    Raises: See function window_im
    """

    assert type(x) is np.ndarray, 'Input array is not a numpy array'
    assert type(center) in [int, float], '"center" is not numeric type'
    assert type(width) in [int, float], '"width" is not numeric type'
    y = ymax / (1. + np.exp(-4. * (x.astype(float) - center) / width))
    if 'int' in return_type:
        y = np.round(y)

    return y.astype(return_type)


def nested_index(data, tag, raise_attribute_error=True):
    """Accesses values from a indexable-object given nested indicies.
    Note: Example includes accessing val[0][1][0].
        Assumes tags are without error.
    Args:
        data: An indexable-object to be accessed according to indices in t
        tag (int or list): An index or a list of indices.
        raise_attribute_error: If data does not have tag, and is set to False, it will raise an error;
            else if it is False, it will return None
    Returns:
        Pydicom DataElememt

    Some test examples:
    nested_index(ds, ['ViewCodeSequence', 0, 'CodeValue'])
    nested_index(ds, 'SOPClassUID')
    nested_index(ds, ["0008", "0016"])
    nested_index(ds, [["0054", "0220"],0,["0008","0100"]])
    """

    def _isdcmcode(tag):

        if isinstance(tag, list) or isinstance(tag, tuple):
            if len(tag) == 2:
                if len(tag[0]) == 4 and len(tag[1]) == 4:
                    return True
        return False

    def _getattr(data, tag):  # Need to already make sure tag is either list of codes or string
        if _isdcmcode(tag) or isinstance(tag, int):  # Account for either receiving a tuple of str ints or just one int
            if not raise_attribute_error:
                if data is None:
                    return None
                try:
                    return data[tag]
                except (KeyError, IndexError):
                    return None
            else:
                return data[tag]
        else:
            if raise_attribute_error:
                return getattr(data, tag)
            else:
                return getattr(data, tag, None)

    if not isinstance(tag, list) or _isdcmcode(tag):
        value = _getattr(data, tag)

    else:
        value = data
        for t in tag:
            value = _getattr(value, t)

    if raise_attribute_error is False and value == 'does_not_have_attribute':
        return None
    else:
        if isinstance(value, pydicom.dataelem.DataElement):
            value = value.value
        return value


def dcm_value_to_string(value, field):
    if value is None:
        return ''

    if isinstance(value, dict):
        if field == 'PatientName':
            return ' '.join(value['components'])
        return ' '.join(map(str, value.values()))
    elif isinstance(value, pydicom.multival.MultiValue):
        if field == 'PatientOrientation':
            return '|'.join(map(str, value))
    if isinstance(value, list) or isinstance(value, tuple):
        return ' '.join(map(str, value))

    if field in ('ContentDate', 'StudyDate', 'PatientBirthDate'):
        # Date (DA) fields. YYYYMMDD or YYYY.MM.DD --> YYYYMMDD
        return str(value).strip().replace(".", "")
    if field in ('ContentTime', 'StudyTime'):
        # Time (TM) fields. HHMMSS.FFFFFF or HH:MM:SS.frac --> HHMMSS.FFFFFF
        val = str(value).strip().replace(":", "")
        # It's not obligatory to get all the HHHMMSS components. Force
        l = len(val.split('.')[0])
        val = "0" * (6-l) + val
        return val
    return str(value)


def dcm_to_array(ds, orig_file_path=None):
    """Extracts numpy array from pydicom dataset
    """

    if is_str_type(ds):
        assert orig_file_path is None or ds == orig_file_path, \
            "File path specified by ds (" + ds + ") doesn't match orig_file_path (" + str(orig_file_path) + ")"
        orig_file_path = ds
        ds = pydicom.dcmread(ds)

    if determine_if_hologic_sco(ds):
        assert orig_file_path is not None, "Need to specify file path when extracting Hologic SC pixels"
        x = extract_hologic_sc_pixels(orig_file_path)
    else:
        x = ds.pixel_array
    return x


def dcm_to_preprocessed_array(ds=None, x=None, attribute_dict=None, slice_num=None, standardize_window_params=True,
                              allow_other_manufacturers=False):
    '''
    The data can all be in a DICOM (ds) or the pixel data and attributes can be passed in (x, attribute_dict).
    ds can also be a string, in which case it is assumed it is a file_path and will be read in
    '''
    assert ds is not None or (x is not None and attribute_dict is not None), \
        'Must provide DICOM OR both the pixel data AND the attribute dictionary'

    if is_str_type(ds):
        orig_file = ds
        ds = dh_dcmread(ds)
    else:
        orig_file = None  # needed for dcm_to_array

    if x is None:
        x = dcm_to_array(ds, orig_file)

    def get_value(tag):
        if ds is not None:
            val = ds.dh_getattribute(tag)
        else:
            val = attribute_dict[tag]
        return val

    # Get manufacturer name
    manu = get_value('Manufacturer')
    # manu = ''.join(m.lower() for m in manu if not m.isspace())
    manu = map_manufacturer(manu)

    max_bit = get_value('BitsAllocated')
    ymax = float(2 ** max_bit - 1)

    if get_value('WindowCenterWidthExplanation') in ['', None]:
        width = int(get_value('WindowWidth'))
        center = int(get_value('WindowCenter'))
    else:
        explanations = list(get_value('WindowCenterWidthExplanation'))  # looks like ['NORMAL', 'HARDER', 'SOFTER']
        explanations = [str(x).lower() for x in explanations]
        normal_index = explanations.index('normal')
        width = float(get_value('WindowWidth')[normal_index])
        center = float(get_value('WindowCenter')[normal_index])

    unknown_manufacturer = False

    if manu.lower() == 'hologic':
        window_func = window_im
    elif manu.lower() == 'ge':
        window_func = window_im_sigmoid
        if standardize_window_params:
            if get_value('HighBit') == 11:
                center = min(center, 3060)  # one standard deviation above median in AST data
                center = max(center, 2500)
    elif allow_other_manufacturers:
        unknown_manufacturer = True
    else:
        raise ValueError('Manufacturer not supported: ' + manu)

    if slice_num is not None:
        assert x.ndim == 3, 'Trying to get slice_num from an array of dimension ' + str(x.ndim)
        assert slice_num >= 0, 'slice_num must be >=0'
        assert slice_num < x.shape[0], 'array has ' + str(x.shape[0]) + ' slices and desired slice_num is ' + str(
            slice_num)
        x = x[slice_num]

    assert x.ndim in [2, 3], 'x must have 2 or 3 dimensions'

    # Use custom windowing if the manufacturer is GE/Hologic, otherwise use built-in Pydicom windowing.
    if not unknown_manufacturer:
        if x.ndim == 3:  # iterate by slice to save memory
            for i in range(x.shape[0]):
                x[i] = window_func(x[i], center, width, ymax=ymax)
            return x
        else:
            return window_func(x, center, width, ymax=ymax)
    else:
        # Use pydicom builtin function
        return pydicom.pixel_data_handlers.util.apply_voi_lut(x, ds)


def alt_tag_format_function_map(function_name):
    """
    return the function given the function name
    Args:
        function_name: str, that represents function or 'None'

    Returns: function

    """
    format_function_map = {
        "verbose_position_to_code": verbose_position_to_code,
        "None": lambda x: x
    }
    if function_name not in format_function_map:
        raise ValueError("unidentified function name {}".format(function_name))
    return format_function_map[function_name]


def evaluate_alt_tag(row, ds):
    """
    given a row from ALT_TAGS evaluate value of that tag
    Args:
        row: a row from ALT_TAGS
        ds: pydicom.Dataset

    Returns: value from the tag

    """

    alt_tag = row.tag
    format_function = alt_tag_format_function_map(row.function)
    value = None

    if isinstance(alt_tag[0], list) and len(alt_tag[0][0]) == 2 and Tag(alt_tag[0][0]).is_private:
        attr_tag, private_tag, cond_val = alt_tag
        tmp = nested_index(ds, private_tag, raise_attribute_error=False)

        if isinstance(tmp, str) and isinstance(cond_val, str):
            tmp = tmp.lower()
            cond_val = cond_val.lower()
        if tmp is not None and tmp == cond_val:
            # some times its 'HOLOGIC, Inc.' instead of 'Hologic, Inc.'
            value = nested_index(ds, attr_tag, raise_attribute_error=False)

    else:
        value = nested_index(ds, alt_tag, raise_attribute_error=False)

    return format_function(value)


def get_dcm_attribute(ds, spec_id, alt_tags, tag, convert_to_string=False, raise_none_error=False):
    """
    get DICOM attribute
    Args:
        ds: Pydicom.Dataset
        spec_id: str represent manufacture and SOPClassUID
        alt_tags: pandas DataFrame of alternative tags
        tag: tag to get value from
        convert_to_string: bool whether or not to convert the value to string
        raise_none_error: bool whether or not to raise error if the value is None

    Returns: value of the tag

    """
    is_str_tag_type = type(tag) in [str, unicode] if sys.version[0] == '2' else isinstance(tag, str)
    if is_str_tag_type:
        tag = format_dicom_tag_str(str(tag))

    value = nested_index(ds, tag, raise_attribute_error=False)
    if value is None:
        if is_str_tag_type:
            sub_df = alt_tags[(alt_tags['spec_id'] == spec_id) & (alt_tags['field'] == tag)]

            for i, row in sub_df.iterrows():
                value = evaluate_alt_tag(row, ds)
                if value is not None:
                    break

    if value is None and raise_none_error:
        raise AttributeError("Attribute '{}': No Alt Tag Specified".format(tag))
    else:
        if convert_to_string:
            value = dcm_value_to_string(value)
        return value

def verify_tag(tag):
    """
    Verifies that a regular or nested dicom tag matches the expected format.
    :param tag: str. A string specifying a regular or nested dicom tag.
    :return: bool. True if the tag is formatted as expected, otherwise False.
    """
    dicom_tag_regex = "^\[(\"|\')[a-zA-Z0-9]{4}(\"|\'),\s?(\"|\')[a-zA-Z0-9]{4}(\"|\')\]|\d$"  # Tags or digits
    bracket_tag_regex = '\[(.*?)\]'  # Elements in square brackets
    tag_valid = True

    # If brackets exist around the tag
    if not bool(re.match(bracket_tag_regex, tag)):
        tag_valid = False
    else:
        # If this is a nested tag, verify each individual element
        if tag.startswith('[['):
            # Use ast.literal_eval to parse elements. If not parsable, then it is invalid
            try:
                tag_elements = ast.literal_eval(tag)
            except SyntaxError:
                tag_valid = False
            else:
                for tag_element in tag_elements:
                    if not bool(re.match(dicom_tag_regex, str(tag_element))):
                        tag_valid = False
        else:
            if not bool(re.match(dicom_tag_regex, tag)):
                tag_valid = False

    return tag_valid


def verify_alt_tags(ALT_TAGS):
    """
    Verifies that ALT_TAGS is structured as expected. Extracts dicom tags from ALT_TAGS and uses the verify_tag()
    function to verify their validity.

    Args:
        ALT_TAGS: pandas dataframe of tags

    Returns:True if ALT_TAGS is structured as expected and its dicom tags are verified successfully, otherwise False.

    """
    for alt_tag in ALT_TAGS.tag.values:
        tag_to_compare = alt_tag

        if isinstance(alt_tag[0], list) and Tag(alt_tag[0][0]).is_private:
            if len(alt_tag) != 3 or len(alt_tag[0][0]) != 2:
                return False, 'Tag specified needs to contain [attr_tag, tag_to_compare, expected_val]'
            attr_tag, tag_to_compare, expected_val = alt_tag
        if not verify_tag(str(tag_to_compare)):
            return False, 'Found invalid tag in ALT_TAGS: {}'.format(alt_tag)

    for function in ALT_TAGS.function.values:
        try:
            alt_tag_format_function_map(function)
        except ValueError:
            return False, "invalid format function {}".format(function)

    return True, None

# def add_alt_tag(manu_model_name, sopclassuid, field, attr_tag,
#                 tag_to_compare=None, expected_val=None, format_function='None'):
#     """
#     Adds a new alt_tag to the dictionary defined in deephealth_utils.data.dicom_constants.ALT_TAGS. Can handle new
#     alt_tag addition whether or not the specified manufacturer model name or field already exist in the dictionary.
#
#     Args:
#         manu_model_name: str that represents model name
#         sopclassuid: SOPClassUID
#         field: str  DICOM filed
#         attr_tag: tag to be added
#         tag_to_compare: tag to compare for private tag
#         expected_val: compare the expected value for the tag_to_compare
#         format_function: function/function name use to format the value of the tag
#
#     Returns: ALT_TAGS with the newly added alt_tag.
#
#     """
#
#
#
#     new_ALT_TAGS = ALT_TAGS.copy()
#     spec_id = '_'.join([manu_model_name, sopclassuid])
#
#     # Create entry
#     if tag_to_compare is not None:
#         assert expected_val is not None, 'If tag_to_compare is not None then expected_val should not be None'
#         entry = list()
#         entry.append(attr_tag)
#         entry.append(tag_to_compare)
#         entry.append(expected_val)
#     else:
#         entry = attr_tag
#
#     if not isinstance(format_function, str):
#         if format_function.callable():
#             format_function = format_function.__name__
#         else:
#             format_function = str(format_function)
#
#     if attr_tag not in df.groupby(['spec_id', 'field'])['tag'].get_group((spec_id, field)).values:
#         row = dict(zip(['spec_id', 'field', 'tag', 'function'], [spec_id, field, entry, format_function]))
#         return new_ALT_TAGS.append(row, ignore_index=True)
#     return new_ALT_TAGS

alt_tags_valid, error_message = verify_alt_tags(ALT_TAGS)
if not alt_tags_valid:
    raise ValueError(error_message)

# hack class for demo mode
class DHDataset(pydicom.Dataset):
    @property
    def pixel_array(self):
        """
        Use a manual property to skip the default behavior of a pydicom dataset
        Returns:
            numpy array or None
        """
        return getattr(self, 'PixelData', None)

def dh_dcmread(file_or_folder_name, stop_before_pixels=False, external_metadata=None):
    """
    In regular scenario (external_metada==None), read a pydicom Dataset and create a "DeepHealth pydicom dataset",
    extending the behavior of the regular pydicom dataset.
    In "Demo" scenario, the method receives numpy arrays only and it needs to create the datasets using external
    metadata information
    Args:
        file_or_folder_name (str): path to a dataset (regular mode) or a folder that contains numpy array/s for an image
        stop_before_pixels (bool): stop before reading the pydicom dataset pixel_array
        external_metadata (Dataframe): external metadata information when the method receives just numpy files

    Returns:
        pydicom dataset with extended behaviour
    """
    if external_metadata is None:
        # Regular scenario
        ds = pydicom.dcmread(file_or_folder_name, stop_before_pixels=stop_before_pixels)
    else:
        # Demo mode.
        ds = DHDataset()
        assert isinstance(external_metadata, pd.Series), f"pandas Series expected. Got: {external_metadata}"
        for col in external_metadata.index:
            setattr(ds, col, external_metadata[col])
        # The data should go into filename
        frames = glob.glob(file_or_folder_name + "/*.npy")
        array = None
        for j, frame_filename in enumerate(frames):
            x = np.load(frame_filename, allow_pickle=True)
            pixel_array_shape = x.shape
            if array is None:
                shape = (len(frames),) + pixel_array_shape
                array = np.zeros(shape)
            frame = x
            array[j] = frame
        setattr(ds, 'PixelData', np.squeeze(array))

    if hasattr(ds, 'ManufacturerModelName'):
        model_name = map_manufacturer_model_for_specs(ds.ManufacturerModelName)
        spec_id = model_name + '_' + ds.SOPClassUID
    else:
        spec_id = None

    dh_getattribute = partial(get_dcm_attribute, ds, spec_id, ALT_TAGS)
    ds.dh_getattribute = dh_getattribute

    return ds


def get_memory_used():
    """
    Get the total number of memory bytes used by the current python process
    :return: int. Number of bytes used
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def get_study_metadata(dicom_fields=(
                                "PatientName", "PatientID", "PatientFirstName", "PatientLastName", "PatientBirthDate",
                                "PatientSex", "AccessionNumber", "StudyInstanceUID", "StudyDate"),
                        file_list=None, metadata_df=None, fail_if_missing_field=True):
    """
    Read different dicom fields from a study dicom header files.
    If metadata_df is not None, the field will be searched in the dataframe first.
    If the field is not in the dataframe, then it will be searched in file_list.
    In the second scenario, the method will loop over all the input fields until all the requested fields are read.
    The format of the values will be always a string to be consistent with metadata_df
    Args:
        dicom_fields (tuple/list of str): dicom fields
        file_list (tuple/list of str): paths to the study dicom files
        metadata_df (Dataframe): pandas dataframe with the information read by the Checker (when applicable)

    Returns:
        dictionary of str: str (dicom field: value_str_representation)
    """
    results = {}
    df_available = metadata_df is not None and len(metadata_df) > 0
    file_list_available = file_list is not None and len(file_list) > 0

    if file_list_available:
        cached_files = [None] * len(file_list)
    special_fields = ("PatientFirstName", "PatientLastName")
    for field in dicom_fields:
        found = False
        if df_available and field in metadata_df.columns and not field in special_fields:
            df = metadata_df[(metadata_df[field].notnull()) & (metadata_df[field]!='')]
            if df.empty:
                # Search for empty strings and accept them as valid value to distinguish from non-dicom files
                df = metadata_df[metadata_df[field] == '']
            # select the first NonNull value, '' if the field was not found or None if there are not any dicom fields
            results[field] = None if df.empty else df.iloc[0][field]
            found = True
        elif file_list_available:
            # Loop over the file list
            for i in range(len(file_list)):
                if cached_files[i] is None:
                    try:
                        cached_files[i] = dh_dcmread(file_list[i], stop_before_pixels=True)
                    except:
                        logging.getLogger().exception("File ({}) could not be read".format(file_list[i]))
                        continue
                ds = cached_files[i]
                if field == "PatientFirstName" and "PatientName" in ds:
                    name = ds.PatientName
                    if isinstance(ds.PatientName, str) and ds.PatientName == '':
                        results[field] = ''
                    else:
                        results[field] = name.given_name
                    found = True
                    break
                elif field == "PatientLastName" and "PatientName" in ds:
                    name = ds.PatientName
                    if isinstance(ds.PatientName, str) and ds.PatientName == '':
                        results[field] = ''
                    else:
                        results[field] = name.family_name
                    found = True
                    break
                elif field in ds:
                    # Read in the same way that it's done in the preprocessing
                    field_val = ds.dh_getattribute(field)
                    if field_val is None:
                        field_val = ''
                    else:
                        field_val = dcm_value_to_string(field_val, field)
                    results[field] = field_val
                    found = True
                    break
        if not found:
            if fail_if_missing_field:
                raise ValueError(f"Field {field} could not be read")
            else:
                warnings.warn(f"Field {field} could not be read")
    return results