import itertools
import cv2
import os
import pandas as pd
import copy
import numpy as np

from centaur_deploy.deploys.config import Config
from centaur_reports.report import Report
from deephealth_utils.misc.utils import post_process_im
from deephealth_utils.ml.detection_helpers import box_coords_to_transformed
from deephealth_utils.data.input_processing import ImageInputProcessor
import deephealth_utils.misc.results_parser as results_parser
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
import deephealth_utils.misc.drawing as draw
from centaur_engine.helpers.helper_category import CategoryHelper
from centaur_engine.helpers import helper_preprocessor, helper_results_processing


class ImagesReport(Report):
    """ Report class: Creates a Report object for
    Methods:
    Notes:
    """

    def __init__(self, model, config, logger=None):
        """
        Constructor
        :param model: Model object
        :param config: Config object
        :param logger:
        """
        super().__init__(config.get_algorithm_version(), logger=logger)
        self._model = model
        self._config = config
        self._input_proc = ImageInputProcessor(**model.get_input_proc_config()).create_input
        self._params_config = self._model.get_params_config()
        self.ram_images = None

        self.box_color = [255, 255, 255]
        self.box_thickness = 10
        self.box_text_size = 2
        self.box_text_thickness = 7
        self.box_text_color = [255, 255, 255]

        self.draw_modality_header_letters = False
        # self.output_dir = None
        # self._view_text_color = [255, 255, 255]
        # self._view_text_size = 4
        # self._view_text_thickness = 4
        # self._view_text_offset_x1 = 20
        # self._view_text_offset_x2 = 350
        # self._view_text_offset_y = 100

        # self._box_color = [255, 255, 255]
        # self._box_thickness = 10
        # self._box_text_size = 1
        # self._modality_y_offest = 40
        # self.category_text_thickness = 4

    def generate(self, study_deploy_results, ram_images=None, instance_uids=None,
                 draw_box=True, draw_one_bbx_only=False):
        """
        Base class to generate a report based on images/bounding boxes
        Args:
            study_deploy_results (StudyDeployResults): Object that contains the results of the study
            ram_images (dict): Dictionary of preprocessed images saved in ram memory
            instance_uids (list-str): List of selected SOPInstanceUIDs to use (by default use all)
            draw_box (bool): whether or not to draw bounding boxes
            draw_one_bbx_only (bool): draw just the most malignant bounding box
        Returns:
            Dictionary of SOPInstanceUID-Numpy arrays with the image for each SOPInstanceUID
        """
        self.ram_images = ram_images
        results = copy.copy(study_deploy_results.results)
        metadata = copy.copy(study_deploy_results.metadata)
        dicom_results = results_parser.get_dicom_results(results)
        proc_info = results_parser.get_proc_info(results)

        combined_results_df = self._get_combined_results_df(metadata, dicom_results)
        maps = self._get_maps(combined_results_df, proc_info, metadata, instance_uids=instance_uids,
                              draw_box=draw_box, draw_one_bbx_only=draw_one_bbx_only)
        return maps

    def _get_combined_results_df(self, metadata_df, dicom_results):
        """
        Generate a dataframe that combines metadata and dicom results
        :param metadata_df:
        :param dicom_results:
        :return:
        """
        # Create an aux dataframe with the results
        results_df = pd.DataFrame(columns=['SOPInstanceUID', 'ix', 'score', 'category', 'coords', 'slice'])

        i = 0
        for instance_uid in metadata_df['SOPInstanceUID']:
            r = dicom_results[instance_uid]['none']
            for j in range(len(r)):
                if "category" in r[j]:
                    cat = r[j]['category']
                else:
                    cat = ''
                assert not pd.isna(r[j]['score']), "score should not be Nan"
                results_df.loc[i] = [instance_uid, j, r[j]['score'], cat, r[j]['coords'],
                                     r[j]['slice'] if 'slice' in r[j] else 0]
                i += 1
        results_df = pd.merge(metadata_df, results_df, on="SOPInstanceUID", suffixes=('', '_'))
        return results_df

    def get_map_helper(self, X, results_instance_df,  proc, draw_one_bbx_only, draw_box, slice_):
        """
        do additional pre-processing of transformed_image, orient/transform the coordinates of the boxes and call the
        _generate_map function for each image
        Args:
            X: numpy array that represent the the pixel values of a dicom image
            results_instance_df: pandas dataframe, contain all results/boxes for that image
            proc: pre-processing information for that image
            draw_one_bbx_only: bool specifies whether or not to draw only one box
            draw_box: bool specifies whether or not to draw boxes
            slice_: int, slice number the box is on

        Returns:

        """
        transformed_image = self._input_proc(X, return_extra_info=False)[0]
        if draw_one_bbx_only:
            # Draw one box at most
            results_instance_df = results_instance_df.iloc[:1]
        num_results = len(results_instance_df)
        if num_results == 0:
            # Just return the transformed image. No bounding boxes to be drawn
            return X

        # There are some bounding boxes
        results_instance_df['coords_transformed'] = [[0, 0, 0, 0]] * num_results
        coords = results_instance_df['coords'].to_list()

        orientation_change = helper_preprocessor.get_orientation_change(
            results_instance_df['ImageLaterality'].values[0],
            results_instance_df['PatientOrientation'].values[0])

        if orientation_change is not None and orientation_change != "unknown":
            # Rotate coordinates to move them to the "preprocessed image coordinates space"
            for i in range(len(coords)):
                coords[i] = helper_results_processing.orient_bounding_box_coords(coords[i],
                                                                                 proc['original_shape'],
                                                                                 orientation_change)

        coords_transformed = box_coords_to_transformed(coords, proc)
        for i in range(len(coords_transformed)):
            ix = int(results_instance_df.iloc[i].name)
            results_instance_df.at[ix, 'coords_transformed'] = coords_transformed[i].tolist()

        image_map = self._generate_map(transformed_image, proc, results_instance_df, slice_, draw_box=draw_box)
        return image_map


    def _get_maps(self, results_df, proc_info, metadata_df, instance_uids=None,
                  draw_box=True, draw_one_bbx_only=False):
        """
        Generate all the images
        Args:
            results_df (Dataframe): Dataframe that contains the metadata + the results dicom info combined
            proc_info (dictionary): Info about image processing
            metadata_df (Dataframe): Original metadata dataframe
            instance_uids (list-str): List of instance uids to filter. If None, all the images will be generated
            draw_box (bool): Draw bounding box/es
            draw_one_bbx_only (bool): Draw just the most malignant bounding box

        Returns:
            (dict). Dictionary of images (with keys as tuples (Modality,Laterality,View))
        """

        def get_slice(row):
            if row.dicom_types == DicomTypeMap.DXM:
                return 0
            try:
                return int(row['slice'])
            except ValueError:
                # no box for this dbt, get the middle slice
                assert pd.isna(row['slice']), "unexpected slice value {}".format(row['slice'])
                assert pd.isna(row['score']), "when slice is Nan, score of Nan is expected got {}".format(row['score'])
                if self.ram_images is None:
                    num_slices = os.listdir(row_meta['np_paths'])
                    return len(num_slices) // 2
                else:
                    ix = metadata_df.loc[metadata_df['SOPInstanceUID'] == row_meta['SOPInstanceUID']].index[0]
                    num_slices = len([k for k in self.ram_images[ix].keys() if isinstance(k, int)])
                    return num_slices // 2

        # creates a df frame that is similar to results_df but also include dicoms that dont have results
        df = pd.concat([metadata_df[~metadata_df['SOPInstanceUID'].isin(results_df.SOPInstanceUID)], results_df])
        df['dicom_types'] = df.apply(DicomTypeMap.get_type_row, axis=1)

        if instance_uids is not None:
            df = df[df['SOPInstanceUID'].isin(instance_uids)]
        df = df.sort_values('score', ascending=False)

        # get the first SOPInstanceUID because the df is sorted so it will select the most malignant
        data = df.groupby(['dicom_types', 'ImageLaterality', 'ViewPosition'])['SOPInstanceUID'].first()
        all_views_lat = list(itertools.product((DicomTypeMap.DXM, DicomTypeMap.DBT), ('L', 'R'), ('CC', 'MLO')))
        maps = dict(zip(all_views_lat, [None]*len(all_views_lat)))



        # iter  through all the latview for each modality
        for key in data.keys():

            modality, lat, view = key
            instance_uid = data[key]
            row_meta = metadata_df[metadata_df['SOPInstanceUID'] == instance_uid].iloc[0]
            instance_df = df[df['SOPInstanceUID'] == instance_uid].sort_values('score', ascending=False)
            results_instance_df = results_df[results_df['SOPInstanceUID'] == instance_uid].sort_values('score', ascending=False)

            # get the slice from the most malignant slice. slice_ is zero for 2D
            slice_ = get_slice(instance_df.iloc[0])

            if self.ram_images is None:
                p = row_meta['np_paths'] + '/frame_{}.npy'.format(slice_)
                X = np.load(p)
            else:
                ix = row_meta.name
                X = self.ram_images[ix][0]


            img_map = self.get_map_helper(X, results_instance_df,  proc_info[instance_uid], draw_one_bbx_only,
                                          draw_box, slice_)
            if modality == DicomTypeMap.DBT:
                maps[(modality, lat, view)] = (instance_uid, slice_), img_map
            else:
                maps[(modality, lat, view)] = (instance_uid, img_map)



        if self.draw_modality_header_letters:
            self._draw_modality_header_letters(img_map, lat, view)

       
        return maps

    def _generate_map(self, image, proc_info, results_df, slice_=0, draw_box=True):
        """
        Given bbox predictions for an image (Laterality+View), draw the image and the bounding boxes
        Args:
            image (numpy array): 2D Image (dxm or 3D slice)
            proc_info (dict): proc_info dictionary included in the results object
            results_df (dataframe): results aggregated in a dataframe
            slice_ (int): slice number (for DBT images only)
            draw_box (bool): whether or not to draw bounding boxes

        Returns:
            (numpy array) image with all the required annotations
        """
        # Main view
        im = image.squeeze().copy()
        # Rescale image (in-place)
        min_ = im.min()
        max_ = im.max()
        im -= min_
        im /= max_ - min_
        im *= 255

        # Draw bounding boxes in the main image
        if draw_box:
            for _, row in results_df.iterrows():
                category = row['category']
                b = row['coords_transformed']
                assert b != [0, 0, 0, 0],"null box found!"

                try:
                    category_text = CategoryHelper.get_category_abbreviation(
                        category=category,
                        run_mode=self._config[Config.MODULE_DEPLOY, 'run_mode']
                    )
                except Exception as e:
                    category_text = ''

                type_row = DicomTypeMap.get_type_row(row)

                # Draw solid in Dxm or in DBT if the box is in the slice displayed
                draw_solid = type_row == DicomTypeMap.DXM or \
                             (type_row == DicomTypeMap.DBT and int(row['slice']) == slice_)

                if draw_solid:
                    # Box in the current slice (main image)
                    cv2.rectangle(im,
                                  (b[0], b[1]), (b[2], b[3]),
                                  self.box_color,
                                  self.box_thickness,
                                  cv2.LINE_AA)
                else:
                    # Projected bounding box (dashed)
                    draw.drawrect(im, b[:2], b[2:], self.box_color, style="dashed", thickness=self.box_thickness)

                pos_x = max(((b[2] - b[0]) // 2) + b[0] - (55 if len(category_text) <= 3 else 60), 0)
                pos_y = b[3] + 60
                if pos_y >= im.shape[0] - 65:
                    # Bbx too low. Place the category on the top of the bounding box
                    pos_y = b[1] - 15
                cv2.putText(im, category_text, (pos_x, pos_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.box_text_size, self.box_text_color, thickness=self.box_text_thickness)

        # Resize the image with crop info
        im = post_process_im(im, self._params_config['map_params'], proc_info)

        # DEBUG
        # import datetime
        # f = "{}.jpg".format(datetime.datetime.strftime(datetime.datetime.now(), "%H%M%S"))
        # cv2.imwrite(f, im)
        # os.system("open {}".format(f))

        return im

    def _draw_modality_header_letters(self, im, laterality, view):
        """
        Draw the corner modality letters (RCC, LCC, etc.)
        :param im: numpy array. Image
        :param laterality: str. 'L' or 'R'
        :param view: str. 'MLO' or 'CC'
        """
        view_text_color = [255, 255, 255]
        view_text_size = 4
        view_text_thickness = 6
        view_text_offset_x1 = 20
        view_text_offset_x2 = 350
        view_text_offset_y = 120
        pos_x = view_text_offset_x1 if laterality == 'R' else im.shape[1] - view_text_offset_x2
        pos_y = view_text_offset_y
        modality = laterality + view
        cv2.putText(im, modality, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,
                    view_text_size, view_text_color, thickness=view_text_thickness)
