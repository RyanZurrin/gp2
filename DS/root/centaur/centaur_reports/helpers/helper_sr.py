import numpy as np
from centaur_reports import constants as CONST
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence


class CodedEntry:

    def __init__(self, values):

        self.seq = self.process(values)
        self.val = self.seq[0]

    def process(self, values):
        """
        make values into a Sequence
        Args:
            values: list like object represent DICOM codes

        Returns: Pydicom.Sequence

        """

        if not isinstance(values, list):
            values = [values]
        seq = Sequence()
        for val in values:
            assert len(val) in [3,4], "value can either be length 3 or 4"
            co = Dataset()
            co.CodeValue = val[0]
            co.CodingSchemeDesignator = val[1]
            co.CodeMeaning = val[2]
            if len(val) == 4:
                co.CodingSchemeVersion = val[3]
            seq.append(co)
        return seq


class DataEntity:

    def __init__(self, relationship_type, value_type=None, continuity=None):

        self.co = Dataset()
        self.co.RelationshipType = relationship_type
        if value_type is not None:
            self.co.ValueType = value_type
        if continuity is not None:
            self.co.ContinuityOfContent = continuity

        self.val = self.co
        self.seq = Sequence([self.co])

    def set_code(self, values):
        """
        set the concept code sequence
        Args:
            values: coded values to be in the sequence

        Returns: None

        """
        if not isinstance(values, list):
            values = [values]

        seq = Sequence()
        for val in values:
            seq.append(CodedEntry(val).val)

        self.co.ConceptCodeSequence = seq

    def set_name_code(self, values):
        """
        sets Concept Name Code sequence
        Args:
            values: The value of the Concept Name Code Sequence

        Returns: None

        """

        if not isinstance(values, list):
            values = [values]

        seq = Sequence()
        for val in values:
            seq.append(CodedEntry(val).val)

        self.co.ConceptNameCodeSequence = seq

    def set_content_seq(self, values):
        """
        Set the Content Sequence
        Args:
            values: Value of the Content Sequence

        Returns: None

        """
        if not isinstance(values, list):
            values = [values]
        seq = Sequence()
        for val in values:
            seq.append(val)
        self.co.ContentSequence = seq


class MeasuredValue:

    def __init__(self, values):

        self.seq = self.process(values)
        self.val = self.seq[0]

    @staticmethod
    def process(values):
        """
        process values into a Sequence that represent the measured value
        Args:
            values: list that represents a measured value

        Returns:

        """
        if not isinstance(values, list):
            values = [values]
        seq = Sequence()
        for val in values:
            seq2 = Sequence()
            co = Dataset()
            co.CodeValue = val[1]
            co.CodingSchemeDesignator = val[2]
            co.CodingSchemeVersion = val[3]
            co.CodeMeaning = val[4]
            seq2.append(co)
            co2 = Dataset()
            co2.MeasurementUnitsCodeSequence = Sequence([co])
            co2.NumericValue = val[0]
            seq.append(co2)
        return seq


def get_image_attrs(row, report_info):
    """
    add a list of attribute for the image specified by the row
    Args:
        row: a row of metatdata
        report_info: dictionary that has ContentTime and ContentDate

    Returns:

    """

    for el in ['ContentTime', 'ContentDate']:
        row[el] = report_info[el]

    d = row.copy()

    image_attrs = []


    for attr in ['ImageLaterality',
                 'ViewPosition',
                 'StudyDate',
                 'StudyTime',
                 'ContentDate',
                 'ContentTime',
                 ]:

        attr_d = CONST.IMAGE_LIBRARY_DICT[attr]
        attr_co = DataEntity('HAS ACQ CONTEXT', attr_d['ValueType'])
        attr_co.set_name_code(attr_d['ConceptName'])

        if attr_co.co.ValueType == 'TEXT':
            attr_co.co.TextValue = d[attr]
        elif attr_co.co.ValueType == 'DATE':
            attr_co.co.Date = d[attr]
        elif attr_co.co.ValueType == 'TIME':
            attr_co.co.Time = d[attr]
        elif attr_co.co.ValueType == 'CODE':
            attr_co.set_code(attr_d['Value'][d[attr]])
        elif attr_co.co.ValueType == 'NUM':
            attr_co.co.MeasuredValueSequence = MeasuredValue((d[attr], 'um', 'UCUM', '1.4', 'micrometer')).seq

        image_attrs.append(attr_co.val)

    return image_attrs


def create_bbox(bbox):

    x1, y1, x2, y2 = bbox

    bbox = [x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]

    bbox = list(map(float, bbox))

    return bbox

def get_center(bbox):

    x1, y1, x2, y2 = bbox

    x = int((x1+x2)/2)
    y = int((y1+y2)/2)

    return [x, y]


def convert_score(s):
    num = float(s) * 100

    return '%s' % float('%.3g' % num)


def flip_coords(coords, dims):
    coords = list(map(int, coords))

    frame = np.zeros(dims)

    frame[coords[1]:coords[3], coords[0]:coords[2]] = 1
    frame2 = np.rot90(frame, 2)
    frame2 = np.rot90(frame2, 2)
    xs = np.where(frame2)[1]
    ys = np.where(frame2)[0]
    new = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

    return list(map(float, new))


def value_container(value_type, name_code, val):
    co = DataEntity('HAS PROPERTIES', value_type)
    co.set_name_code(name_code)
    co.co.TextValue = val
    return co


def get_bbox_container(report_info, box_info, category_mappings):
    """
    get a container for a bounding box
    Args:
        report_info: dictionary with information about this report
        box_info: dictionary that represents a box
        category_mappings: dictionary that maps numeric to text category

    Returns: None

    """
    ann_co_list = []

    # main annotation container
    ann_co = DataEntity('HAS CONCEPT MOD', 'CODE')
    ann_co.set_name_code(('111056', 'DCM', 'Rendering Intent'))
    ann_co.set_code(('111150', 'DCM', 'Presentation Required: Rendering device is expected to present'))
    ann_co_list.append(ann_co.val)

    # add algorithm name and algorithm version
    ann_co_list.append(
        value_container('TEXT', ('111001', 'DCM', 'Algorithm Name'), report_info['AlgorithmName']).val)

    ann_co_list.append(
        value_container('TEXT', ('111003', 'DCM', 'Algorithm Version'), report_info['AlgorithmVersion']).val)

    # adding category
    if report_info['intended_workstation'] in ['ThreePalm', "eRad"]:
        ann_co6 = DataEntity('HAS PROPERTIES', 'TEXT')
        ann_co6.set_name_code(('DH0411', 'DHCODE', 'Finding Assessment'))
        ann_co6.val.TextValue = category_mappings[box_info['category']]

    else:
        raise ValueError('Config \'intended_workstation\': {} not supported.'
                         .format(report_info['intended_workstation']))

    ann_co_list.append(ann_co6.val)

    # add the center
    ann_co5 = DataEntity('HAS PROPERTIES', 'SCOORD')
    ann_co5.set_name_code(('111010', 'DCM', 'Center'))
    ref = DataEntity('SELECTED FROM')
    ref.co.ReferencedContentItemIdentifier = [1, 2, box_info['library_idx']]
    ann_co5.set_content_seq(ref.val)
    ann_co5.co.GraphicData = get_center(box_info['coords'])
    ann_co5.co.GraphicType = 'POINT'
    ann_co_list.append(ann_co5.val)

    # add the coordinates
    ann_co4 = DataEntity('HAS PROPERTIES', 'SCOORD')
    ann_co4.set_name_code(('111041', 'DCM', 'Outline'))
    ref = DataEntity('SELECTED FROM')
    ref.co.ReferencedContentItemIdentifier = [1, 2, box_info['library_idx']]
    ann_co4.set_content_seq(ref.val)
    ann_co4.co.GraphicData = create_bbox(box_info['coords'])
    ann_co4.co.GraphicType = 'POLYLINE'
    ann_co_list.append(ann_co4.val)

    return ann_co_list
