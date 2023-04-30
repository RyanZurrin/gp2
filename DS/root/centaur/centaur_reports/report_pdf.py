import itertools
import logging
from collections import defaultdict
import shutil
import re
import cv2
import os
import pdf2image
import pdfkit
import subprocess
import copy
import imageio
from bs4 import BeautifulSoup

from centaur_deploy.deploys.config import Config
from centaur_reports.report_most_malignant_image import ImagesReport
import deephealth_utils.misc.results_parser as results_parser
from deephealth_utils.data.dicom_type_helpers import DicomTypeMap
from centaur_engine.helpers.helper_category import CategoryHelper
from deephealth_utils.data.parse_logs import log_line
import centaur_deploy.constants as const_deploy
import centaur_reports.constants as const

class PDFReport(ImagesReport):
    """ PDF + SC report for CADx"""

    def __init__(self, model, config, logger=None):
        """
        Constructor
        :param model: Model object
        :param config: Config object
        :param logger:
        """
        super().__init__(model, config, logger=logger)

        self._temp_html_folder = self._temp_html_report = None
        self._report_params = None

        self._product = const.SAIGE_Q if config.get_run_mode() == const_deploy.RUN_MODE_CADT else const.SAIGE_DX
        self._report_prefix = 'PDF'
        self._file_extension = 'dcm'

    @property
    def _source_folder(self):
        return os.path.abspath(os.path.dirname(__file__)) + "/libs/pdf/"

    @property
    def max_length_texts(self):
        """
        Maximum length for a text in different sections
        :return: dictionary of str(section)-int(length)
        """
        return {
            "patient_info": 17,
            "slice_findings": 35
        }

    def generate(self, study_deploy_results, ram_images=None, clean_temp_files=True):
        """
        Generate a PDF report + an SC report embedding the same PDF file
        Args:
            study_deploy_results (StudyDeployResults): Object that contains the results of the study
            ram_images (dict): Dictionary of preprocessed images saved in ram memory
            clean_temp_files (bool): Remove temporary html files once the pdf has been created

        Returns:
            dictionary of str ('pdf_output','dcm_output'): Paths to the generated reports
        """

        ## AUX FUNCTIONS
        # def get_num_findings(modality, laterality, view):
        #     laterality = laterality.upper()
        #     view = view.upper()
        #     key = (modality, laterality, view)
        #     if key not in instance_uids:
        #         return 0
        #     instance_uid = instance_uids[key]
        #     if modality == "dbt":
        #         instance_uid = instance_uid[0]
        #     return len(combined_results_df[combined_results_df['SOPInstanceUID']==instance_uid])

        def get_dbt_slice(laterality, view):
            laterality = laterality.upper()
            view = view.upper()
            key = ('dbt', laterality, view)
            if key not in instance_uids:
                return None
            return instance_uids[key][1]

        def get_dbt_all_slices(laterality, view):
            laterality = laterality.upper()
            view = view.upper()
            key = ('dbt', laterality, view)
            if key not in instance_uids:
                return None
            instance_uid = instance_uids[key][0]
            l = combined_results_df[combined_results_df['SOPInstanceUID'] == instance_uid].sort_values('slice')[
                'slice'].unique()
            if len(l) == 0:
                return "[-]"
            return "[{}]".format(",".join(map(lambda f: str(int(f)), l)))

        ## END AUX FUNCTIONS
        results = copy.copy(study_deploy_results.results)
        metadata = copy.copy(study_deploy_results.metadata)
        output_dir = study_deploy_results.output_dir
        self.ram_images = ram_images
        # metadata['dcm_path'] = "/root/test_datasets/DS_01/data/" + metadata['dcm_path']
        # metadata['np_paths'] = "/root/test_datasets/DS_01/baseline/" + metadata['np_paths']
        file_name = self.get_output_file_name(study_deploy_results.get_studyUID())

        study_results = results_parser.get_study_results(results)
        dicom_results = results_parser.get_dicom_results(results)
        proc_info = results_parser.get_proc_info(results)

        combined_results_df = self._get_combined_results_df(metadata, dicom_results)
        maps = self._get_maps(combined_results_df, proc_info, metadata)

        # Create temp files folder
        study_uid = metadata['StudyInstanceUID'][0]

        # Create temp html folder
        self._temp_html_folder = os.path.join(output_dir,
                                  f'{self._config[Config.MODULE_REPORTS, "temp_html_files_dir"]}_{self._report_prefix}')

        if os.path.isdir(self._temp_html_folder):
            shutil.rmtree(self._temp_html_folder)
        os.makedirs(self._temp_html_folder)
        self._temp_html_report = os.path.join(self._temp_html_folder, "report.html")
        self._temp_jpeg = os.path.join(self._temp_html_folder, "temp_pdf.jpeg")

        # merged_df = pd.merge(metadata, how='left', )
        default_img_path = os.path.realpath(os.path.dirname(__file__)) + "/libs/pdf/dh_resources/dh_na_img2.png"
        report_imgs_paths = defaultdict(lambda: default_img_path)
        instance_uids = {}
        for modality, lat, view in itertools.product(('dxm', 'dbt'), ('L', 'R'), ('CC', 'MLO')):
            key = (modality, lat, view)
            if maps[key] is not None:
                instance_uids[key] = maps[key][0]
                # Save the image
                img_name = self._config[Config.MODULE_REPORTS, "temp_imgs_filename_template"].format(
                    modality=modality, laterality=lat, view=view)
                report_imgs_paths[key] = os.path.join(self._temp_html_folder, img_name)
                im = maps[key][1]
                imageio.imwrite(report_imgs_paths[key], im)
        self.logger.debug(log_line(-1, "Images saved to {}".format(output_dir)))

        if len(study_uid) > self.max_length_texts['patient_info']:
            ix = self.max_length_texts['patient_info'] - 3
            study_uid = "..." + study_uid[-ix:]

        # The patient metadata will be the same for all the rows
        patient_info = metadata.iloc[0]
        modality = 'DBT' if len(DicomTypeMap.get_type_df(metadata, 'dbt')) > 0 else 'DM'

        self._report_params = {}

        try:
            self._report_params['dh-patient-name'] = patient_info['PatientName']
        except:
            self.logger.warning(log_line(-1, 'Patient name not found or wrong format'))
            self._report_params['dh-patient-name'] = None

        try:
            self._report_params['dh-pid'] = patient_info['PatientID']
        except:
            self.logger.warning(log_line(-1, 'PatientID name not found or wrong format'))
            self._report_params['dh-pid'] = None

        try:
            date_ = patient_info['StudyDate']
            if date_ is not None:
                # YYYY/MM/DD
                date_ = "{}/{}/{}".format(date_[:4], date_[4:6], date_[6:])
            self._report_params['dh-exam-date'] = date_
        except:
            logging.warning('StudyDate not found or wrong format')
            self._report_params['dh-exam-date'] = None

        self._report_params['dh-sid'] = study_uid
        self._report_params['dh-modality'] = modality
        run_mode = self._config[Config.MODULE_DEPLOY, 'run_mode']
        self._report_params['dh-suspicious-level-case'] = CategoryHelper.get_category_text(study_results['total']['category'],
                                                                                           run_mode=run_mode)
        self._report_params['dh-suspicious-level-left'] = \
            CategoryHelper.get_category_text(study_results['L']['category'], run_mode=run_mode) if 'L' in study_results else None
        self._report_params['dh-suspicious-level-right'] = \
            CategoryHelper.get_category_text(study_results['R']['category'], run_mode=run_mode) if 'R' in study_results else None
        self._report_params['dh-study-percentile'] = study_results['total']['postprocessed_percentile_score']

        for lat, view in itertools.product(('l', 'r'), ('mlo', 'cc')):
            self._report_params['dh-img-dxm-{}-{}'.format(lat, view)] = \
                report_imgs_paths[('dxm', lat.upper(), view.upper())],
            dbt = report_imgs_paths[('dbt', lat.upper(), view.upper())]
            self._report_params['dh-img-dbt-{}-{}'.format(lat, view)] = dbt
            if dbt is None:
                # No DBT image. Remove all slice text
                self._report_params['dh-findings-main-slice-dbt-{}-{}'.format(lat, view)] = \
                    self._report_params['dh-findings-all-slices-dbt-{}-{}'.format(lat, view)] = \
                    ''
            else:
                self._report_params['dh-findings-main-slice-dbt-{}-{}'.format(lat, view)] = \
                    "Slice displayed: {}".format(get_dbt_slice(lat, view))
                self._report_params['dh-findings-all-slices-dbt-{}-{}'.format(lat, view)] = \
                    "Slices of findings: {}".format(get_dbt_all_slices(lat, view))

        self._create_html()
        pdf_output = os.path.join(output_dir, file_name.replace('.dcm', '.pdf'))
        self._create_pdf(pdf_output)
        dcm_output = os.path.join(output_dir, file_name)

        #return list of images corresponding to the page of the pdf get the first page
        pdf_img = pdf2image.convert_from_path(pdf_output)[0]
        pdf_img.save(self._temp_jpeg)

        cmd_str = 'img2dcm -sef {} {} {} '.format(metadata.iloc[0]['dcm_path'], self._temp_jpeg, dcm_output)
        out_code = subprocess.call(cmd_str, shell=True)
        assert out_code == 0, "There was an error in the img2dcm. Command: {}".format(cmd_str)
       
        if clean_temp_files:
            shutil.rmtree(self._temp_html_folder)
            self.logger.debug(log_line(-1, '{} folder removed'.format(self._temp_html_folder)))

        return {'pdf_output': pdf_output, 'dcm_output': dcm_output}

    def _create_html(self):
        """
        Create the html file that will be used to generate the PDF.
        If the process finishes ok, self._temp_html_report will contain the path to the generated file
        :param report_params: dictionary. Parameters used for the html generation
        """
        html_template = self._source_folder + 'index.html'

        # Read the template
        with open(html_template, 'r') as fp:
            s = fp.read()
            # Append the root path to the dh_resources folder to convert them into absolute paths
            s = s.replace("dh_resources/", self._source_folder + "dh_resources/")
            # Read new html for parsing
            soup = BeautifulSoup(s, 'lxml')

        for key, val in list(self._report_params.items()):
            # logging.debug("{}={}".format(key, val))
            if key == "dh-study-percentile":
                # Special case. Score bar
                # Get the offset based on the bar height and score
                despl = self._get_score_bar_despl(val)  # Move the arrow accordingly
                arrow_tag = soup.find(id="dh-score-arrow")
                arrow_tag.attrs['style'] += ";margin-top: {}px".format(despl)
            # finds element
            tag = soup.find(id=key)
            if tag is not None and val is not None:
                if tag.name == 'img':
                    # Image source
                    tag.attrs['src'] = val
                else:
                    # Just update the text
                    tag.string = val

        # Truncate patient info texts
        patient_info_div = soup.find(id="dh-patient-info-div")
        for tag in patient_info_div.find_all("h6"):
            tag.string = self._truncate_text(tag.string, self.max_length_texts['patient_info'])

        # Truncate slice finding
        for tag in soup.find_all(id=re.compile("dh-findings")):
            tag.string = self._truncate_text(tag.string, self.max_length_texts['slice_findings'])

        # output HTML file (temp)
        with open(self._temp_html_report, 'w') as outfile:
            outtext = soup.prettify(formatter='minimal')
            # to output file w/ pretty printing formatting
            # outtext = str(outtext.encode(encoding="utf-8", errors="strict"))
            outfile.write(outtext)
        logging.debug(log_line(-1, "{} generated".format(self._temp_html_report)))

    def _create_pdf(self, output_pdf_path):
        """
        Create the PDF file once the HTML has been generated
        :param output_pdf_path: str. Output file path
        """
        options = {
            # 'page-size': 'Letter',
            # 'disable-smart-shrinking': None,
            'dpi': 1200,
            'viewport-size': '1280x1024',
            # 'dpi': 400,
            # 'zoom': 0.75,
            # 'zoom': 1.25,
            # 'print-media-type': None,
            'margin-top': '0.0in',
            'margin-right': '0.0in',
            'margin-bottom': '0.0in',
            'margin-left': '0.0in',
            'page-width': '8.7in',
            'page-height': '6.5in',
            'encoding': 'UTF-8',
            # 'orientation': 'Landscape',
            'custom-header': [
                ('Accept-Encoding', 'gzip')
            ],
            'no-outline': None,
        }
        # Using PDFKit to generate PDF from HTML
        temp_pdf_path = output_pdf_path + ".uncropped.pdf"
        pdfkit.from_file(self._temp_html_report, temp_pdf_path, options=options)
        # Crop the white margins in the pdf file
        proc = subprocess.run("pdf-crop-margins -p 0 {} -o {}".format(temp_pdf_path, output_pdf_path), shell=True)
        proc.check_returncode()
        # Remove uncropped pdf
        os.remove(temp_pdf_path)
        print('{} generated'.format(output_pdf_path))

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

    def _get_score_bar_despl(self, study_score_percentile, min_value=0, max_value=232):
        """
        Get the offset in px that the score arrow needs to be displaced
        :param study_score_percentile: float 0-100
        :param min_value: int. Min px offset
        :param max_value: int. Max px offset
        :return: int. Number of pixels
        """
        range = max_value - min_value
        return int((1 - (study_score_percentile / 100)) * range) + min_value

    def _truncate_text(self, text, max_length):
        """
        Truncate the text that is displayed in a section (add '...' at the end)
        :param text: str. Text to validate
        :param max_length: int. Maximum length allowed in the text
        :return: str. Truncated string (if necessary)
        """
        if not isinstance(text, str):
            text = str(text)
        if len(text) <= max_length:
            # No need to truncate
            return text
        # Truncate
        return text[:(max_length - 2)] + "..."
