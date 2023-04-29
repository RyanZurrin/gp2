import pandas as pd
import numpy as np
from enum import Enum

class DicomTypeMap(object):
    DXM = 'dxm'
    DBT = 'dbt'
    MAMMO_CAD_SR = 'mammo_cad_sr'

    dicom_dict = {
        'dxm': '1.2.840.10008.5.1.4.1.1.1.2',
        'dbt': '1.2.840.10008.5.1.4.1.1.13.1.3',
        'mammo_cad_sr': '1.2.840.10008.5.1.4.1.1.88.50'
    }

    class Classification(Enum):
        INTELLIGENT_2D = "I2D"
        CVIEW = "C-View"
        HOLOGIC_SLABS = 'Hologic Slabs'
        HOLOGIC_SLICES = 'Hologic Slices'
        UNKNOWN = 'Unknown'


    @staticmethod
    def get_dxm_class_id():
        return DicomTypeMap.get_class_id(DicomTypeMap.DXM)

    @staticmethod
    def get_dbt_class_id():
        return DicomTypeMap.get_class_id(DicomTypeMap.DBT)

    @staticmethod
    def get_mammo_cad_sr_class_id():
        return DicomTypeMap.get_class_id(DicomTypeMap.MAMMO_CAD_SR)

    @staticmethod
    def is_dbt(row):
        return DicomTypeMap.get_type_row(row) == DicomTypeMap.DBT

    @staticmethod
    def get_study_type(metadata_df):
        """
        Get a study type based on the metadata. If there is one DBT file, the study will be
        considered DBT. Otherwise, it will be considered Dxm
        :param metadata_df: dataframe
        :return: str. DicomTypeMap.DBT or DicomTypeMap.DXM
        """
        for ix, row in metadata_df.iterrows():
            if DicomTypeMap.is_dbt(row):
                return DicomTypeMap.DBT
        return DicomTypeMap.DXM

    @staticmethod
    def get_type_row(row):

        if isinstance(row, pd.core.frame.DataFrame):
            assert len(row) == 1, "Numbers of rows is {} but needs to be 1".format(len(row))
            pixel_array_shape = row['pixel_array_shape'].values[0]
            number_of_frames = row['NumberOfFrames'].values[0]
            image_classification = row['ImageClassification'].values[0]
        else:
            pixel_array_shape = row['pixel_array_shape']
            number_of_frames = row['NumberOfFrames']
            image_classification = row['ImageClassification']
        if image_classification == 'I2D':
            return DicomTypeMap.DXM
        if hasattr(pixel_array_shape, '__len__'):
            if len(pixel_array_shape) == 3:
                return DicomTypeMap.DBT
            elif len(pixel_array_shape) == 2:
                return DicomTypeMap.DXM
            else:
                raise ValueError("Pixel array shape is {} but should be 2 or 3".format(len(pixel_array_shape)))
        else:
            if number_of_frames in ['', 1, np.nan]:
                return DicomTypeMap.DXM
            elif isinstance(number_of_frames, (int, float)):
                if number_of_frames == number_of_frames and number_of_frames > 1:
                    return DicomTypeMap.DBT
                else:
                    raise ValueError("Number of frames is nan")
            else:
                raise ValueError("Number of frames is unrecognized type, {}".format(number_of_frames))


    @staticmethod
    def get_type_df(df, dicom_type):
        if dicom_type not in DicomTypeMap.dicom_dict.keys():
            raise NotImplementedError("Not a valid dicom type")
        dicom_type_series = df.apply(DicomTypeMap.get_type_row, axis=1)
        return df[dicom_type_series == dicom_type]

    @staticmethod
    def get_class_id(dicom_type):
        if dicom_type in DicomTypeMap.dicom_dict:
            return DicomTypeMap.dicom_dict[dicom_type]
        else:
            raise NotImplementedError("Dicom type is not implemented.")


    @staticmethod
    def get_image_classification(ds):
        """
        Get image classification based on SeriesDescription and SeriesNumber
        Args:
            ds: Dataset

        Returns: ImageClassification

        """
        series_number = ds.dh_getattribute('SeriesNumber')
        series_description = ds.dh_getattribute('SeriesDescription')

        if series_description is not None:
            if "intelligent 2d" in series_description.lower():
                return DicomTypeMap.Classification.INTELLIGENT_2D


            if 'c-view' in series_description.lower():
                return DicomTypeMap.Classification.CVIEW

        if str(series_number) == '73500000':
            return DicomTypeMap.Classification.HOLOGIC_SLABS

        if str(series_number) == '73200000':
            return DicomTypeMap.Classification.HOLOGIC_SLICES

        return DicomTypeMap.Classification.UNKNOWN


