import copy
import glob
import os
import json

import centaur_deploy.constants as const_deploy
import centaur_engine.constants as const
import centaur_engine.helpers.helper_results_processing as helper_results_processing
import numpy as np
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from skimage.io import imsave

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Engine(object):
    """Class contains the Engine object.
    Methods:
        set_file_list
        set_output_dir
        preprocess
        check
        evaluate
        save
        run-all
        clean
    Notes: preprocess, check and evaluate studies
    """

    def __init__(self, preprocessor=None, model=None, reuse=False, save_to_ram=True, logger=None, config=None):
        self.file_list = None
        self.output_dir = None
        self.preprocessor = preprocessor
        self.model = model
        self.input_proc_config = self.model.get_input_proc_config() if model is not None else None
        self.results_raw = None
        self.results = None
        self.reuse = reuse
        self.save_to_ram = save_to_ram
        self.report = None
        self.logger = logger
        self.config = config

    def set_file_list(self, file_list):
        """
        Specify the list of files that belong to the study being evaluated
        :param file_list: list of files names that corresponds to the study
        :return: None
        """
        self.file_list = file_list

    def set_output_dir(self, output_dir):
        """
        Specify the output_dir to output the results of the study
        :param output_dir: out_put directory for the results
        :return: None
        """
        self.output_dir = output_dir

    def get_metadata_path(self):
        """
        Get the (expected) path to the metadata file
        :return: str.
        """
        return os.path.join(self.output_dir, const.METADATA_PKL_NAME)

    def get_results_paths(self):
        """
        Get the (expected) paths to the results file (post-processed and raw)
        :return:
        """
        return (os.path.join(self.output_dir, const.CENTAUR_RESULTS_JSON),
                os.path.join(self.output_dir, const.CENTAUR_RESULTS_RAW_JSON))

    def preprocess(self, input_dir=None):
        """
        Preprocess the files in the study so its in a format suitable for model evaluation.
        Args:
            input_dir (str): input directory (special case for DEMO mode)

        Returns:
            (bool). The study passed all the validations and was preprocessed correctly
        """
        if self.config['checker_mode'] == 'demo':
            passed = self.preprocessor.demo_preprocess(input_dir, self.file_list)
        else:
            passed = self.preprocessor.preprocess(self.file_list)
        return passed

    def evaluate(self):
        """
        Pass the preprocessed data into model to get prediction
        :return: None
        """
        metadata_df = self.preprocessor.get_metadata(passed_checker_only=True)
        if self.save_to_ram:
            input_pixel_data = self.preprocessor.get_pixel_data()
            self.model.set_pixel_data(input_pixel_data)

        # Sanity checks
        # obligatory_dicom_fields = ['dcm_path', 'SOPClassUID', 'SOPInstanceUID', 'AccessionNumber',
        #                            'SeriesInstanceUID', 'ViewPosition', 'ImageLaterality',
        #                            'Rows', 'Columns', 'Manufacturer',
        #                            'ManufacturerModelName', 'StudyInstanceUID',
        #                            'FrameOfReferenceUID', 'PatientOrientation',
        #                            'FieldOfViewHorizontalFlip', 'FieldOfViewOrigin', 'FieldOfViewRotation',
        #                            'pixel_array_shape']
        #
        # for c in obligatory_dicom_fields:
        #     assert not metadata_df[c].isnull().any(), f"Field {c} not available in all the rows in the metadata"

        self.results_raw = self.model(metadata_df)

        # Post process the results
        self.results_raw = self._process_raw_results(self.results_raw, metadata_df)
        self.results = self._results_post_processing(self.results_raw, metadata_df)
        helper_results_processing.model_results_sanity_checks(self.results, self.results_raw, metadata_df,
                                                              self.config['results_pproc_max_bbxs_displayed_total'],
                                                              self.config[
                                                                  'results_pproc_min_relative_bbx_size_allowed'],
                                                              self.config[
                                                                  'results_pproc_max_relative_bbx_size_allowed'],
                                                              run_mode=self.config['run_mode'])

    def _add_num_slices_to_proc_info(self, model_results, metadata):
        """
        For 3D images, include a 'num_slices' field in the proc_info section
        :param model_results: dict. Model results that will be modified in_place
        :param metadata: Dataframe.
        :return: model_results will be modified in place
        """
        # For 3D images, include the number of slices in the proc_info
        bt_images_df = DicomTypeMap.get_type_df(metadata, 'dbt')
        for ix, row in bt_images_df.iterrows():
            instance_uid = row['SOPInstanceUID']
            if self.save_to_ram:
                # Get the number of frames from the pixel_data
                pixel_data = self.preprocessor.get_pixel_data()
                # Exclude all the arrays that are not part of image (like synthetic, scores...)
                num_slices = len([k for k in pixel_data[ix].keys() if isinstance(k, int)])
            else:
                # check that no frames were skipped
                slice_frames = glob.glob(row['np_paths'] + "/frame_*.npy")
                num_slices = len(slice_frames)
                get_num = lambda x: int(x.split('frame_')[1].split('.npy')[0])
                sorted_files = sorted(slice_frames, key=get_num)
                max_slice_file = get_num(sorted_files[-1])
                assert num_slices == max_slice_file + 1, \
                    "Number of slice files {} does not match the max slice number {}".format(num_slices, max_slice_file)

            model_results['proc_info'][instance_uid]['num_slices'] = num_slices

    def _process_raw_results(self, model_results, metadata):
        """
        Process raw results:
            - Rotate the coordinates of the images that were flipped before running the model prediction

        :param model_results: dict. Original model results
        :param metadata: Dataframe. Metadata dataframe
        :return: dictionary (copy of model_results with applied changes)
        """
        self._add_num_slices_to_proc_info(model_results, metadata)
        mr = helper_results_processing.sort_bounding_boxes(model_results)
        mr = helper_results_processing.orient_coordinates(mr, metadata)
        return mr

    def _results_post_processing(self, model_results, metadata):
        """
        Model results post processing. Modify the results dict in place.
        - Sort the bounding boxes in a descending score order for each image/transform
        - Maximum of X bounding boxes per image
        - The category for each laterality and for each bounding box cannot be higher than the category for the study

        :param model_results: dict. Original model results (raw)
        :param metadata: Dataframe. Metadata dataframe
        :return: dictionary (copy of model_results with applied changes)
        """
        mr = copy.deepcopy(model_results)
        if self.config['run_mode'] == const_deploy.RUN_MODE_CADX:
            mr = helper_results_processing.assign_cadx_categories(mr, metadata, self.model.config["cadx_thresholds"])
            if DicomTypeMap.get_study_type(metadata) == DicomTypeMap.DBT:
                mr = helper_results_processing.combine_foruid_boxes(mr, metadata, self.model.config,
                                                                    self.preprocessor.get_pixel_data(), self.logger)
            mr = helper_results_processing.cap_bounding_box_category(mr, metadata, logger=self.logger)
            mr = helper_results_processing.cap_bbx_number_per_image(mr,
                                                                    self.config['results_pproc_max_bbxs_displayed_total'],
                                                                    self.config['results_pproc_max_bbxs_displayed_intermediate'])
        elif self.config['run_mode'] == const_deploy.RUN_MODE_CADT:
            # The 3 bbox postprocessing functions break if there are no bbox categories, which is why they are not
            # run if run_mode is CADt
            mr = helper_results_processing.assign_cadt_categories(mr, DicomTypeMap.get_study_type(metadata),
                                                                  self.config['cadt_operating_point_values'])
        elif self.config['run_mode'] == const_deploy.RUN_MODE_DEMO:
            mr = helper_results_processing.assign_cadx_categories(mr, metadata, self.model.config["cadx_thresholds"])
        else:
            raise AssertionError('Expected run_mode to be one of {}, got {}.'.format(
                [const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_DEMO, const_deploy.RUN_MODE_CADT],
                self.config['run_mode']))
        mr = helper_results_processing.fix_bbxs_size(mr,
                                                     self.config['results_pproc_min_relative_bbx_size_allowed'],
                                                     self.config['results_pproc_max_relative_bbx_size_allowed'])
        mr = helper_results_processing.add_percentile_study_scores(mr, metadata, self.model.config)
        return mr


    def save_intermediate_files(self, save_results_post_processed=False, save_results_raw=False, save_synthetics=False):
        """
        Save the model results (postprocessed and raw) in two json files and/or synthetic files
        Args:
            save_results_post_processed (bool): save results.json
            save_results_raw (bool): save results_raw.json
            save_synthetics (bool): save synthetic files
        """
        metadata = self.preprocessor.get_metadata()
        if save_results_raw or save_results_post_processed:
            results_json = dict()
            results_json['metadata'] = metadata.to_json()
            results_json['model_results'] = self.results
            output_file_path, output_raw_file_path = self.get_results_paths()
            if save_results_post_processed:
                with open(output_file_path, 'w') as fp:
                    json.dump(results_json, fp)
            if save_results_raw:
                with open(output_raw_file_path, 'w') as fp:
                    json.dump(self.results_raw, fp)

        if save_synthetics:
            def save_synth_ims(sop_uid, synth, slice_map):
                sop_dir = os.path.join(self.output_dir, sop_uid)
                try:
                    os.makedirs(sop_dir, exist_ok=True)
                except:
                    raise IOError("Unable to create {}".format(sop_dir))
                imsave('{}/synth.png'.format(sop_dir), synth)
                np.save('{}/slice_map.npy'.format(sop_dir), slice_map)

            if self.save_to_ram:
                # Get the number of frames from the pixel_data
                pixel_data = self.preprocessor.get_pixel_data()
                for k, v in pixel_data.items():
                    if 'synth' in v:
                        save_synth_ims(metadata.loc[k]['SOPInstanceUID'], v['synth'], v['slice_map'])
            else:
                for i, path in metadata['np_paths'].items():
                    parent_dir = os.path.dirname(path)
                    if 'synth' in os.listdir(parent_dir):
                        synth = np.load(os.path.join(parent_dir, 'synth', const.SYNTHETIC_IM_FILENAME))
                        slice_map = np.load(os.path.join(parent_dir, 'synth', const.SYNTHETIC_SLICE_MAP_FILENAME))
                        save_synth_ims(os.path.basename(path), synth, slice_map)



    def clean(self):
        """
        delete temporary files
        :return: None
        """
        self.preprocessor.clean()
