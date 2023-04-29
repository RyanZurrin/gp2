import numpy as np
import pydicom

def get_bt_framecount(dcm):
    '''
    Args:
         dcm (dicom): A BT dicom. Pixel data is not needed.
    Returns:
         count (int): The number of frames in the dicom's pixel data.

    This function counts the nested sequences within the tag 'Per-frame Functional Groups Sequence'
    to determine the number of frames in the dicom's pixel data.
    '''
    count = dcm[('5200', '9230')].VM  # 'VM' returns sequence length

    return int(count)


def classify_dicom(dcm, strict=True):
    '''
    Args:
         dcm (dicom): A dicom. Pixel data is preferred, but not required.
         strict (bool): When set to True, any SCs that don't contain pixel data
            will return np.nan for classification['dimensionality'] regardless of
            the other content of the SC. When set to False, the script will ASSUME
            that all non-Hologic SCs are 2D, and will return a classification['dimensionality']
            value of np.nan for Hologic dicoms and '2D' for all other manufacturers.

    Returns:
         classification (dict): 'dicomtype' : Specifies whether the dicom is DXm, BT, or SC
                                'dimensionality' : 2D or 3D
                                'imageType' : Whether the dicom is a Slab, Plane, or Synthetic if it's a BT
                                              Whether the dicom is FFDM or Synthetic if it's a DXm
                                              Whether the dicom is Normal or a VPreview if it's an SC

    This function attempts to classify a passed-in dicom by finding its dicom type (BT, DXm, SC),
    its dimensionality (2D vs 3D), and its image type (synthetic, VPreview, plane, slab, etc...)
    '''

    classification = {
        'dicomType' : None,
        'dimensionality' : None,
        'imageType' : None,
    }


    try:
        dcm_SOPClassUID = str(dcm.SOPClassUID.name)

    except AttributeError:
        dcm_SOPClassUID = 'None'


    classification['dicomType'] = 'SC' if 'secondary capture' in dcm_SOPClassUID.lower() \
        else 'DXm' if 'digital mammography' in dcm_SOPClassUID.lower() \
        else 'BT' if 'breast tomosynthesis' in dcm_SOPClassUID.lower() else 'None'


    pixel_data_present = 'PixelData' in dcm

    if pixel_data_present:
        classification['dimensionality'] = '2D' if len(dcm.pixel_array.shape) == 2 or dcm.pixel_array.shape[0] == 1 else '3D'

    elif classification['dicomType'] == 'SC':
        try:
            num_frames = int(dcm.NumberOfFrames)
            classification['dimensionality'] = '2D' if num_frames == 1 else '3D'

        except AttributeError:
            if strict:
                classification['dimensionality'] = np.nan

            else:
                try:
                    manufacturer = str(dcm.Manufacturer)
                    if 'hologic' in manufacturer.lower():
                        classification['dimensionality'] = np.nan
                    else:
                        classification['dimensionality'] = '2D'

                except AttributeError:
                    classification['dimensionality'] = np.nan

    elif classification['dicomType'] == 'BT':
        try:
            num_frames = dcm[('5200','9230')].VM
            classification['dimensionality'] = '2D' if num_frames == 1 else '3D'

        except AttributeError:
            classification['dimensionality'] = np.nan

    elif classification['dicomType'] == 'DXm':
        classification['dimensionality'] = '2D'

    else:
        classification['dimensionality'] = np.nan


    mmn = str(dcm.ManufacturerModelName).lower()
    im_type = str(dcm.ImageType).lower()

    if classification['dicomType'] == 'SC':
        classification['imageType'] = 'Normal' if 'volumepreview' not in mmn\
            else 'VPreview' if 'volumepreview' in mmn\
            else np.nan

    elif classification['dicomType'] == 'DXm':
        classification['imageType'] = 'FFDM' if 'volumepreview' not in mmn\
                else 'Synthetic' if 'volumepreview' in mmn\
                else np.nan

    elif classification['dicomType'] == 'BT':
        classification['imageType'] = 'Slab' if 'derived' in im_type and 'none' in im_type\
                else 'Plane' if 'original' in im_type and 'none' in im_type\
                else 'Synthetic' if 'derived' in im_type and 'generated_2d' in im_type\
                else np.nan

    else:
        classification['imageType'] = np.nan


    return classification


def determine_if_hologic_sco(ds):
    match_tags = [('Manufacturer', 'HOLOGIC, Inc.'),
                  ('ManufacturerModelName', 'Selenia Dimensions'),
                  ('SOPClassUID', '1.2.840.10008.5.1.4.1.1.7')]

    for tup in match_tags:
        if getattr(ds, tup[0], '') != tup[1]:
            return False
    else:
        return True

def get_all_elements(dcm):
    """
    This function recursively goes through datasets and sequences to produce a list of tuples:
    (full_tag_trace, tag, name, VR, value) where the full_tag_trace is unique for each data element in the file
    For reference: There are basically three data elements in pydicom:
    DataSet is derived from python dict and has keys, value pairs
    DataElement is one set of tags, VR, VM and Value
    Sequence is a list of DataElements or DataSets

    Args:
        dcm (pydicom.dataset.FileDataset): dicom object

    Returns:
        list of lists: list of tags and values for all elements

    """

    elements_list = []
    if type(dcm) is pydicom.dataset.Dataset or type(dcm) is pydicom.dataset.FileDataset:
        for tag in dcm.keys():
            elements_list.extend(get_all_elements(dcm[tag]))
        return elements_list

    elif type(dcm) is pydicom.sequence.Sequence :
        for index in range(len(dcm)):
            for temp_element in get_all_elements(dcm[index]):
                elements_list.append(temp_element)
                elements_list[-1][0] = "-"+str(index+1)+"-" + elements_list[-1][0]
        return elements_list

    elif type(dcm) is pydicom.dataset.DataElement or type(dcm) is pydicom.dataset.RawDataElement:
        if type(dcm.value) is pydicom.sequence.Sequence:
            for temp_element in get_all_elements(dcm.value):
                elements_list.append(temp_element)
                elements_list[-1][0] = str(dcm.tag) + elements_list[-1][0]
            return elements_list
        else:
            return [[str(dcm.tag), str(dcm.tag), dcm.name, dcm.VR, dcm.value]]


def compare_dicoms(ds1, ds2):
    """ Compare two DICOMs.


    Args:
        ds1 (pydicom.dataset.FileDataset): dicom1 object
        ds2 (pydicom.dataset.FileDataset): dicom2 object

    Returns:
        list of tuples: list of tuples (tag, value1, value2) for differences
    """

    elements1 = get_all_elements(ds1)
    elements2 = get_all_elements(ds2)

    e1_dict = {row[0]: row[-1] for row in elements1}
    e2_dict = {row[0]: row[-1] for row in elements2}

    in1_not2 = [k for k in e1_dict.keys() if k not in e2_dict.keys()]
    in2_not1 = [k for k in e2_dict.keys() if k not in e1_dict.keys()]

    # if verbose:
    #     if len(in1_not2):
    #         print('Elements in 1 not 2: {}'.format(in1_not2))
    #     if len(in2_not1):
    #         print('Elements in 2 not 1: {}'.format(in2_not1))

    diffs = []
    for k in in1_not2:
        diffs.append((k, e1_dict[k], None))
    for k in in2_not1:
        diffs.append((k, e2_dict[k], None))

    shared_keys = [k for k in e1_dict.keys() if k in e2_dict.keys()]
    for k in shared_keys:
        if e1_dict[k] != e2_dict[k]:
            diffs.append((k, e1_dict[k], e2_dict[k]))

    return diffs
