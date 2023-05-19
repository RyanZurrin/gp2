import math
import pydicom
import datetime
import centaur_reports.constants as const

from centaur_reports.helpers.report_helpers import read_json


class SCWriter(object):

    def __init__(self, algorithm_version):
        self._algorithm_version = algorithm_version
        self.report_info = read_json(const.MSP_SC_REPORT_CONFIG_PATH)

    def create_sc_dataset(self, metadata, im_array):
        """
        Creates a Secondary Capture Dataset given a Series of metadata and an image.
        """
        self.orig_metadata = metadata
        self.im_array = im_array

        self.sc = pydicom.dataset.Dataset()
        self.set_creation_datetime()
        self.set_sc_metadata()

        self.sc.ConversionType = self.report_info['ConversionType']
        self.sc.Manufacturer = self.report_info['Manufacturer']

        self.set_header()
        self.set_original_metadata()
        self.set_pixel_data()

        return self.sc

    def set_pixel_data(self):
        """
        Sets pixel-related attributes in the SC Dataset.
        """
        self.sc.SamplesPerPixel = 1
        self.sc.PhotometricInterpretation = 'MONOCHROME2'
        self.sc.Rows = self.im_array.shape[0]
        self.sc.Columns = self.im_array.shape[1]
        self.sc.BitsAllocated = 16
        max_pixel_val = self.im_array.max()
        self.sc.BitsStored = math.ceil(math.log2(max_pixel_val)) if max_pixel_val != 0 else 1
        self.sc.HighBit = self.sc.BitsStored - 1
        self.sc.PixelRepresentation = 0
        self.sc.PixelData = self.im_array.tostring()

    def set_creation_datetime(self):
        """
        Sets creation date and time in the SC Dataset.
        """
        creation_datetime = datetime.datetime.now()
        self.sc.ContentDate = creation_datetime.strftime('%Y%m%d')
        self.sc.ContentTime = creation_datetime.strftime('%H%M%S.%f')

    def set_header(self):
        """
        Sets SC attributes that are not part of the file metadata, pixel data, or metadata from the original file.
        """
        self.set_creation_datetime()
        self.sc.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        self.sc.SOPInstanceUID = self.generate_uid()
        self.sc.ImageType = self.report_info['ImageType']
        self.sc.Modality = 'MG'
        self.sc.AlgorithmName = self.report_info['AlgorithmName']
        self.sc.AlgorithmVersion = self._algorithm_version


    def set_sc_metadata(self):
        """
        Sets SC file metadata.
        """
        file_meta = pydicom.dataset.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        file_meta.MediaStorageInstanceUID = self.generate_uid()
        file_meta.ImplementationClassUID = self.report_info['dh_unique'] + '2'
        file_meta.ImplementationVersionName = self.report_info['ImplementationVersionName']
        file_meta.SourceApplicationEntityTitle = self.report_info['SourceApplicationEntityTitle']

        self.sc.file_meta = file_meta

    def set_original_metadata(self):
        """
        Sets SC attributes based on metadata from the original file.
        """
        orig_attrs = ['AccessionNumber', 'ImageLaterality', 'PatientAge', 'PatientBirthDate', 'PatientID', 'PatientName',
                      'PatientSex', 'ReferringPhysicianName', 'StudyDate', 'StudyID', 'StudyInstanceUID', 'ViewPosition']
        for attr in orig_attrs:
            setattr(self.sc, attr, self.orig_metadata[attr])

        self.sc.ReferencedSOPInstanceUID = self.orig_metadata.SOPInstanceUID

    def save(self, fpath):
        """
        Saves the SC dataset.
        """
        self.sc.is_little_endian = True
        self.sc.is_implicit_VR = False
        self.sc.file_meta.TransferSyntaxUID = const.EXPLICIT_VR_LITTLE_ENDIAN
        self.sc.save_as(fpath, write_like_original=False)

    def generate_uid(self):
        """
        Generate a UID beginning with the unique DH UID at the start.
        """
        return pydicom.uid.generate_uid(self.report_info['dh_unique'])
