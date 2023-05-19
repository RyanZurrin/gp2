import os
import cv2
import imageio

import deephealth_utils.misc.results_parser as results_parser
from centaur_reports.report_images import ImagesReport
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap

class MostMalignantImageReport(ImagesReport):
    """
    Generate one single image with the most malignant lesion (and other lesions visible in the same image)
    """
    def __init__(self, model, config, draw_box=True, logger=None):
        """
        Constructor
        Args:
            model (centaur_engine.models.model.Model): model
            config (Config): config instance
            draw_box (bool): draw most suspicious bounding box
            logger (Logger): global logger object
        """
        super().__init__(model, config, logger=logger)

        self.box_text_size = 0  # Do not draw category text
        self.draw_modality_header_letters = True    # Draw modality letters (LMLO, RCC, etc.)
        self.draw_box = draw_box                    # Draw bounding box in the most malignant image

        self._product = 'qa'
        self._report_prefix = 'mmaligImage'
        self._file_extension = 'png'

    def generate(self, study_deploy_results, ram_images=None, draw_one_bbx_only=True, custom_path=None):
        """
        Generate one single image that contains the most malignant lesion.
        Args:
            study_deploy_results (StudyDeployResults): Object that contains the results of the study
            ram_images (dict): Dictionary of preprocessed images saved in ram memory
            draw_one_bbx_only (bool): In case we are drawing bounding boxes, make sure we just draw
                                      the most malignant one instead of all the bbxs found in the most malignant image
            custom_path (str): full path where the image should be saved to replace the default location
        Returns:
            str. File path to the generated image
        """
        # If there is no result from engine.
        if study_deploy_results.results is None:
            self.logger.info('There is no study_results. Skip to generate {} report.'.format(__name__))
            return None

        iuid, bbx = results_parser.get_most_malignant_lesion(study_deploy_results)
        if "origin" not in bbx or bbx["origin"] == "dxm":
            self.slice = ""
        else:
            self.slice = str(bbx["slice"])

        images = super(self.__class__, MostMalignantImageReport).generate(self, study_deploy_results,
                                                                          ram_images=ram_images,
                                                                          instance_uids=[iuid],
                                                                          draw_box=self.draw_box,
                                                                          draw_one_bbx_only=draw_one_bbx_only)

        row = study_deploy_results.metadata[study_deploy_results.metadata['SOPInstanceUID'] == iuid].iloc[0]
        modality = DicomTypeMap.get_type_row(row)
        lat = row['ImageLaterality']
        view = row['ViewPosition']
        img_info = images[(modality, lat, view)]
        assert img_info is not None, \
            "not image was generated with modality: {}, lat: {}, view: {}".format(modality, lat, view)
        img = img_info[1]
        if custom_path is None:
            # Use default path
            file_name = self.get_output_file_name(study_deploy_results.get_studyUID())
            file_path = os.path.join(study_deploy_results.output_dir, file_name)
        else:
            # Use custom path
            file_path = custom_path
        imageio.imwrite(file_path, img)
        return file_path

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
        view_text_offset_x2_slice = 960
        view_text_offset_y = 120

        if laterality == "R":
            # Left corner
            pos_x = view_text_offset_x1
        elif laterality == "L":
            if not self.slice:
                # Right corner. Just "LCC" text
                pos_x = im.shape[1] - view_text_offset_x2
            else:
                # Right corner. Text like "LCC-Slice XX"
                pos_x = im.shape[1] - view_text_offset_x2_slice
        else:
            raise ValueError(f"Unknown laterality: {laterality}")

        pos_y = view_text_offset_y
        modality = laterality + view
        if self.slice:
            modality += f"-Slice {self.slice}"

        cv2.putText(im, modality, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, view_text_size, view_text_color, thickness=view_text_thickness)
