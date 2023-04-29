import os
from centaur_deploy.constants import RUN_MODE_CADT, RUN_MODE_CADX

CENTAUR_REPORTS_PATH = os.path.dirname(os.path.abspath(__file__))

STRUCTURED_REPORT_CONFIG_PATH = os.path.join(CENTAUR_REPORTS_PATH, 'configs/structured_reports.json')
MSP_SC_REPORT_CONFIG_PATH = os.path.join(CENTAUR_REPORTS_PATH, 'configs', 'msp_sc_reports.json')

EXPLICIT_VR_LITTLE_ENDIAN = '1.2.840.10008.1.2.1'

SUMMARY_REPORT_STUDY_CSV = 'DH_Summary_studies.csv'
SUMMARY_REPORT_DICOM_CSV = 'DH_Summary_dicoms.csv'
SUMMARY_REPORT_DICOM_AGGREGATED_PKL = 'DH_Summary_dicoms_aggregated.pkl'

IMAGE_LIBRARY_DICT = {

    'ImageLaterality': {'ConceptName': ('111027', 'DCM', 'Image Laterality'),
                        'Value':
                            {'L': ('T-04030', 'SRT', 'Left breast'),
                             'R': ('T-04020', 'SRT', 'Right breast')
                             },
                        'ValueType': 'CODE'
                        },
    'ViewPosition': {'ConceptName': ('111031', 'DCM', 'Image View'),
                  'Value':
                      {'MLO': ('R-10226', 'SRT', 'medio-lateral oblique'),
                       'CC': ('R-10242', 'SRT', 'cranio-caudal')
                       },
                  'ValueType': 'CODE'
                  },
    'PatientOrientationRow': {'ConceptName': ('111044', 'DCM', 'Patient Orientation Row'),
                              'ValueType': 'TEXT'
                              },
    'PatientOrientationColumn': {'ConceptName': ('111043', 'DCM', 'Patient Orientation Column'),
                                 'ValueType': 'TEXT'
                                 },
    'StudyDate': {'ConceptName': ('111060', 'DCM', 'Study Date'),
                  'ValueType': 'DATE'},
    'StudyTime': {'ConceptName': ('111061', 'DCM', 'Study Time'),
                  'ValueType': 'TIME'},
    'ContentDate': {'ConceptName': ('111018', 'DCM', 'Content Date'),
                    'ValueType': 'DATE'},
    'ContentTime': {'ConceptName': ('111019', 'DCM', 'Content Time'),
                    'ValueType': 'TIME'},
    'HorizontalPixelSpacing': {'ConceptName': ('111026', 'DCM', 'Horizontal Pixel Spacing'),
                               'ValueType': 'NUM'},
    'VerticalPixelSpacing': {'ConceptName': ('111066', 'DCM', 'Vertical Pixel Spacing'),
                             'ValueType': 'NUM'}
}

# PRODUCT NAMES
SAIGE_Q = "Saige-Q"
SAIGE_DX = "Saige-Dx"


# RUN  MODE PRODUCT MAP
RUN_MODE_PRODUCT_MAP = {
    RUN_MODE_CADT : SAIGE_Q,
    RUN_MODE_CADX: SAIGE_DX
}