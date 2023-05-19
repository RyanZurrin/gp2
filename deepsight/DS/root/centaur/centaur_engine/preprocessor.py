import random
import string
import time
import os
import numpy as np

import centaur_engine.constants as const
import centaur_engine.helpers.helper_misc as helper_misc
import deephealth_utils.data.utils as dh_data_utils
import logging
import pandas as pd
import shutil
from functools import partial
from centaur_engine.helpers.helper_preprocessor import make_tmp_dir, set_pixel_array, get_orientation_change, \
    implement_orientation
from deephealth_utils.data.checker import Checker
from deephealth_utils.data.parse_logs import log_line
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from deephealth_utils.data.format_helpers import map_manufacturer

class Preprocessor(object):

    def __init__(self, config, logger=None):

        self.dicom_template_paths = {}
        self.study_dir = None
        self.metadata = None
        self._logger = logger
        self.log_line = None
        self._failed_checks = None
        self._checker_passed = None

        self._config = config
        self.checker = Checker(self._config)
        self.pixel_data = {}

    @property
    def logger(self):
        if self._logger is None:
            # Initialize default console logger
            self._logger = helper_misc.create_logger()
        return self._logger


    def set_logger(self, logger):
        self._logger = logger

    def demo_preprocess(self, base_path, file_list):
        """
        To be called instead of preprocess() when running in demo mode.
        Generate pixel_data from numpy arrays instead of the original DICOM files
        :param file_list: list-str. List of paths to DICOM files
        :return: The study passed all the validations
        """
        self.metadata = pd.read_pickle(file_list[0])
        self.metadata['pixel_array_shape'] = None
        self._failed_checks = []
        self.pixel_data = {}
        failed_ixs = []
        for i, row in self.metadata.iterrows():
            # make sure that self.metadata points to the right temp_dir images
            # attrs = row.to_dict()
            # get the row in metadata
            np_paths = row['np_paths']
            abs_path = os.path.join(base_path, np_paths)
            # p = os.path.join(abs_path, frame_filename)
            file_passed, file_failed_checks, ds = self.checker.check_file(abs_path,
                                                                          external_metadata=self.metadata.loc[i],
                                                                          return_ds=True,
                                                                          logger=self.logger)
            if not file_passed:
                self._failed_checks.extend(file_failed_checks)
                failed_ixs.append(i)
                continue

            self.metadata.at[i, 'np_paths'] = abs_path
            image_frames = {}
            # j = 0
            pixel_array = ds.pixel_array    # Note that pixel_array contains the original numpy images, not preprocessed
            pixel_array_shape = pixel_array.shape
            orientation_change = get_orientation_change(self.metadata.loc[i]['ImageLaterality'],
                                                        self.metadata.loc[i]['PatientOrientation'])

            # If the manufacturer is not Hologic or GE, windowing/array preprocessing happens on the client side.
            manufacturer = self.metadata.loc[i]['Manufacturer']
            manufacturer = map_manufacturer(manufacturer)
            if manufacturer.lower() in ['ge', 'hologic']:
                x = dh_data_utils.dcm_to_preprocessed_array(x=pixel_array,
                                                            attribute_dict=self.metadata.loc[i],
                                                            standardize_window_params=False)
            else:
                x = pixel_array.copy()

            if len(pixel_array_shape) == 2:
                # Just one frame
                x = implement_orientation(x, orientation_change)
                image_frames[0] = x
            elif len(pixel_array_shape) == 3:
                # Implement rotation frame by frame
                for j in range(pixel_array_shape[0]):
                    image_frames[j] = implement_orientation(x[j], orientation_change)
            else:
                raise Exception(f"Unexpected shape ({pixel_array_shape}). Dimensions 2 or 3 expected")

            self.pixel_data[i] = image_frames
            assert pixel_array_shape is not None, "pixel data shape could not be obtained"
            self.metadata.at[i, 'pixel_array_shape'] = pixel_array_shape

        # Remove the failed images from the metadata so that they are not processed
        for i in failed_ixs:
            self.metadata = self.metadata.drop(i)

        # If any file passes, we consider the study passed
        self._checker_passed = len(self.metadata) > 0

        return self._checker_passed

    def preprocess(self, file_list):
        """
        Preprocess the list of DICOM files
        :param file_list: list-str. List of paths to DICOM files
        :return: bool. The study passed all the validations
        """
        input_folders = set([os.path.dirname(f) for f in file_list])
        if len(input_folders) > 1:
            self.logger.warning(log_line(5, "All the files were expected in the same folder. Got: {}".format(file_list)))

        self.study_dir = input_folders.pop()
        self.log_line = partial(log_line, study_dir=self.study_dir)

        self.pixel_data = {}

        save_to_ram = self._config['save_to_ram']
        preprocessed_numpy_folder = self._config['preprocessed_numpy_folder']

        self.logger.info(self.log_line('0', 'Processing study.'))

        self.metadata = []
        self._failed_checks = []

        if self._config['parallelize']:
            import deephealth_utils.misc.parallel_helpers as dh_parallel
            self.queue = dh_parallel.ParallelQueue()

        passed = True
        read_pixels = self._config['process_pixel_data']

        skip_study_checks = self._config['skip_study_checks']

        study_name = os.path.basename(self.study_dir)

        t1 = time.time()
        study_size_check_passed, study_size_dict = self.checker.check_study_size(file_list)
        t2 = time.time()
        self.logger.info(log_line('4', "Image_Preprocess_CheckStudySize:{}s;MEM:{}".format(t2 - t1,
                                                                     dh_data_utils.get_memory_used()),
                                                                     study_name))
        if not study_size_check_passed:
            self._failed_checks.extend(study_size_dict)
            self.logger.info(self.log_line('1.2.1', 'Study skipped, size too large.'))
            self._checker_passed = False
            return False

        for i, dicom_path in enumerate(file_list):
            t1_image = time.time()
            self.logger.debug(self.log_line('-1', "Processing dicom {}".format(dicom_path)))
            self.logger.info(self.log_line('2', str(dicom_path)))

            t1 = time.time()
            size_check_passed, file_size_dict = self.checker.check_file_size(dicom_path)
            t2 = time.time()
            self.logger.info(log_line(4, "Image_Preprocess_CheckFileSize:{}s;MEM:{}".format
                                    (t2 - t1, dh_data_utils.get_memory_used()), "{};{}".format(
                                                                                         study_name,
                                                                                         os.path.basename(dicom_path))))
            if not size_check_passed:
                file_passed = False
                file_failed_checks = file_size_dict
                ds = None
            else:
                t1 = time.time()
                # file_passed, file_failed_checks, ds = self.checker.check_file(dicom_path, self._config['reuse_ds'], self.logger)
                file_passed, file_failed_checks, ds = self.checker.check_file(dicom_path, return_ds=True,
                                                                              logger=self.logger)
                t2 = time.time()
                self.logger.info(log_line(4, "Image_Preprocess_Checker:{}s;MEM:{}".format(t2 - t1, dh_data_utils.get_memory_used()),
                         "{};{}".format(study_name, os.path.basename(dicom_path))))

            if file_passed:
                if read_pixels and not save_to_ram:
                    if preprocessed_numpy_folder is None:
                        os.makedirs(const.TMP_NP_PATH, exist_ok=True)
                        tmp_dir = make_tmp_dir(const.TMP_NP_PATH)
                    else:
                        tmp_dir = os.path.join(preprocessed_numpy_folder,
                                               ''.join(random.choice(string.ascii_lowercase) for _ in range(10)))
                        os.makedirs(tmp_dir)
                else:
                    tmp_dir = None

                dr = DICOMReader(dicom_path, ds, None, self.logger, tmp_dir, read_pixels)
                # if self._config['reuse_ds']:
                #     dr.ds = ds
                # else:
                #     dr.read_ds()

                dr.parse_metadata()
                if read_pixels:
                    if self._config['parallelize']:
                        self.queue.add(set_pixel_array, (dr, len(self.metadata), save_to_ram))
                    else:
                        logging.debug(self.log_line('-1', "Start processing data {}".format(time.time())))
                        t1 = time.time()
                        np_data = set_pixel_array((dr, len(self.metadata), save_to_ram))
                        if save_to_ram:
                            self.pixel_data[len(self.metadata)] = np_data
                        t2 = time.time()
                        self.logger.info(log_line(4, "Image_Preprocess_NumpyOps:{}s;MEM:{}".format(t2 - t1,
                                                                                                   dh_data_utils.get_memory_used()),
                                                  "{};{}".format(study_name, os.path.basename(dicom_path))))

                    assert dr.metadata['Rows'] == str(dr.metadata['pixel_array_shape'][-2]), \
                        "'Rows' {} does not match pixel array shape value {}".format(dr.metadata['Rows'], str(
                            dr.metadata['pixel_array_shape'][-2]))
                    assert dr.metadata['Columns'] == str(dr.metadata['pixel_array_shape'][-1]), \
                        "'Columns' {} does not match pixel array shape value {}".format(dr.metadata['Columns'], str(
                            dr.metadata['pixel_array_shape'][-1]))
                    if dr.metadata['NumberOfFrames'] in ['', '1']:
                        assert len(dr.metadata['pixel_array_shape']) == 2, \
                            "NumberOfFrames is {}, expects ndim of pixel_array_shape to be 2 for pixel_array_shape={}".format(
                                dr.metadata['NumberOfFrames'], dr.metadata['pixel_array_shape'])
                    else:
                        assert len(dr.metadata['pixel_array_shape']) == 3, \
                            "NumberOfFrames is {}, expects ndim of pixel_array_shape to be 3 for pixel_array_shape={}".format(
                                dr.metadata['NumberOfFrames'], dr.metadata['pixel_array_shape'])
                self.logger.info(self.log_line('3.1', 'File successfully processed.'))
            else:
                # File didn't pass the Checker
                self._failed_checks.extend(file_failed_checks)
                self.logger.info(self.log_line('3.2', str(file_failed_checks)))
                # No need for output dir or reading pixels because the file didn't pass the Checker
                dr = DICOMReader(dicom_path, ds, list(file_failed_checks.keys()), self.logger, None, False)
                dr.parse_metadata()
                dr.metadata['pixel_array_shape'] = None

            self.metadata.append(dr.metadata)
            t2_image = time.time()
            self.logger.info(
                log_line(4, "Image_Preprocess_Total:{}s;MEM:{}".format(t2_image - t1_image, dh_data_utils.get_memory_used()),
                         "{};{}".format(dr.metadata['StudyInstanceUID'], dr.metadata['SOPInstanceUID'])))

        # self.pixel_data = np.load('temp.npy', allow_pickle=True).item()
        self.metadata = pd.DataFrame(self.metadata)
        valid_metadata = self.get_metadata(passed_checker_only=True)

        if len(valid_metadata) == 0:
            passed = False
            self.logger.info(self.log_line('1.2.1', 'Study failed: No files passed.'))
        elif not skip_study_checks:
            if self._config['parallelize']:
                _ = self.queue.run()
                data = self.queue.get_return_dict()
                if save_to_ram:
                    self.pixel_data = data
                else:
                    # already done in set_pixel_array
                    pass

            # Run the study checks only on the files that passed the Checker
            study_passed, study_reasons = self.checker.check_study(valid_metadata)
            if not study_passed:
                self._failed_checks.extend(study_reasons)
                self.logger.info(self.log_line('1.2.2', str(study_reasons)))
                passed = False
            else:
                self.logger.info(self.log_line('1.1', 'Study successfully processed.'))

        # PACs will be treated as SACs (one PAC fail will make the whole study fail)
        for reason in self._failed_checks:
            if reason.startswith('PAC-'):
                passed = False
                break

        self._checker_passed = passed
        return passed

    def clean(self, root_folder=None):
        """
        Remove the processed numpy arrays
        :param root_folder: str. Root folder. If None, use const.TMP_NP_PATH
        """
        if root_folder is None:
            root_folder = const.TMP_NP_PATH

        if os.path.isdir(root_folder):
            for the_file in os.listdir(root_folder):
                file_path = os.path.join(root_folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:

                    self.logger.warning(self.log_line('-1', e))

        keys = list(self.pixel_data.keys())
        for k in keys:
            del self.pixel_data[k]

    def get_metadata(self, passed_checker_only=False):
        """
        Get a copy of the current metadata
        Args:
            passed_checker_only (bool): return only the files that passed the Checker

        Returns:
            Dataframe
        """
        metadata = self.metadata[self.metadata['failed_checks'].isnull()] if passed_checker_only else self.metadata
        return metadata.copy()


    def get_pixel_data(self):
        return self.pixel_data

    def checker_passed(self):
        """
        The last study passed the Checker validation
        Returns:
            bool
        """
        return self._checker_passed

    def get_failed_checks(self):
        """
        Get a list with all the failed checks
        Returns:
            list of str
        """
        return self._failed_checks


class DICOMReader(object):
    """
    Reader object responsible for reading DICOM files and returning frames as numpy arrays.
    Args:
        dicom_dir (str): The directory containing DICOM file.
        output_dir (str): The directory to store processed numpy arrays.
    Note:
        Getting number of frames from DICOM attribute is faster than reading the whole DICOM and calling shape.
    """

    def __init__(self, dicom_path, ds, failed_checks, logger, output_dir, process_data):
        self.dicom_path = dicom_path
        self.logger = logger
        self.output_dir = output_dir
        self.process_data = process_data
        self.ds = ds
        self.pixel_array = None
        self.metadata = {'failed_checks': failed_checks}

    def read_ds(self):

        self.ds = dh_data_utils.dh_dcmread(self.dicom_path, stop_before_pixels=not self.process_data)

    def get_ds(self):
        if self.ds is not None:
            return self.ds

    def parse_metadata(self):

        dicom_attributes = [
            'AccessionNumber',
            'Columns',
            'CompressionForce',
            'ContentDate',
            'ContentTime',
            'DetectorBinning',
            'Exposure',
            'FieldOfViewHorizontalFlip',
            'FieldOfViewOrigin',
            'FieldOfViewRotation',
            'FrameOfReferenceUID',
            'ImageLaterality',
            'ImagerPixelSpacing',
            'ImageType',
            'InstitutionName',
            'Manufacturer',
            'ManufacturerModelName',
            'Modality',
            'NumberOfFrames',
            'PatientAge',
            'PatientBirthDate',
            'PatientID',
            'PatientName',
            'PatientOrientation',
            'PatientSex',
            'PixelSpacing',
            'PresentationIntentType',
            'ProtocolName',
            'ReferringPhysicianName',
            'Rows',
            'SeriesDescription',
            'SeriesInstanceUID',
            'SeriesNumber',
            'SOPClassUID',
            'SOPInstanceUID',
            'StudyDate',
            'StudyDescription',
            'StudyID',
            'StudyInstanceUID',
            'StudyTime',
            'ViewPosition',
            'WindowCenter',
            'WindowWidth',
            'WindowCenterWidthExplanation'
            ]
        self.metadata['dcm_path'] = self.dicom_path

        if self.ds is not None:
            for field in dicom_attributes:
                try:
                    field_val = self.ds.dh_getattribute(field)
                    if field_val is None:
                        field_val = ''
                    else:
                        field_val = dh_data_utils.dcm_value_to_string(field_val, field)
                    self.metadata[field] = field_val
                except:
                    self.logger.warning(f"Unexpected error for field {field} in file {self.dicom_path}")
                    self.metadata[field] = None
        else:
            self.logger.warning(f"The file {self.dicom_path} does not seem to be a valid DICOM file. " \
                                 "All the metadata will be set to None")
            for field in dicom_attributes:
                self.metadata[field] = None

        try:
            self.metadata['ImageClassification'] = DicomTypeMap.get_image_classification(self.ds).value
        except:
            self.metadata['ImageClassification'] = None

        self.metadata['np_paths'] = None
        if self.output_dir is not None:
            sop_instance_uid = self.metadata.get('SOPInstanceUID')
            if sop_instance_uid:
                self.metadata['np_paths'] = os.path.join(str(self.output_dir), str(sop_instance_uid))



    def get_numpy_frame(self, numpy_array, index):

        """Returns a single frame as a numpy array from a DICOM file.
        Note: Accessing each index individually takes up much less space than first loading the entire array.
        Args:
            index (int): Index of the frame to access.
        Returns:
            frame
        """

        if numpy_array.ndim == 2:
            frame = numpy_array
        else:
            frame = numpy_array[index]
        frame = dh_data_utils.dcm_to_preprocessed_array(self.ds, frame)

        return frame

    def pxl_to_np(self, pixel_array):

        """Extracts and populates output_dir with numpy arrays from DICOM in dicom_dir.
        Note:
        Args:
            save_to_ram (boolean): If true, return the numpy array instead of saving it
        Returns:
            None
        """
        if self.metadata == {}:
            raise Exception('Metadata required to extract numpys')
        self.pixel_array = pixel_array
        logging.debug(log_line('-1', "Started dcm_to_np"))

        t1 = time.time()
        num_frames = self.metadata['NumberOfFrames']
        logging.debug(log_line('-1', "Ended num_frames in {} s".format(time.time() - t1)))

        if num_frames == '':
            num_frames = 1
        else:
            num_frames = int(num_frames)

        pxl_data = {}
        for i in range(num_frames):
            logging.debug(log_line('-1', "Processing {} {}".format(i, time.time())))
            pxl_data[i] = self.get_numpy_frame(self.pixel_array, i)
        return pxl_data

    def np_to_file(self, pxl_data):
        os.makedirs(self.metadata['np_paths'], exist_ok=True)
        for i, frame in pxl_data.items():
            path_ = os.path.join(self.metadata['np_paths'], 'frame_{}.npy'.format(i))
            np.save(path_, frame)
