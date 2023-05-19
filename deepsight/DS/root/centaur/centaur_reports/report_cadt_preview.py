import logging
import shutil
import os
import pdf2image
import pdfkit
import subprocess

from bs4 import BeautifulSoup

from centaur_deploy.deploys.config import Config
from centaur_reports.report_most_malignant_image import ImagesReport
import deephealth_utils.misc.results_parser as results_parser
from deephealth_utils.data.parse_logs import log_line
from centaur_reports.report_most_malignant_image import MostMalignantImageReport
import centaur_reports.constants as const

class CADtPreviewImageReport(ImagesReport):
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
        super().__init__(model, config, logger)

        self._product = const.SAIGE_Q
        self._report_prefix = 'Preview'
        self._file_extension = 'dcm'

        self._temp_html_folder = self._temp_html_report = None
        self._report_params = None
        self._most_malig_img_report = MostMalignantImageReport(model, config, draw_box=False, logger=logger)


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
            "patient_info": 16
        }

    def generate(self, study_deploy_results, ram_images=None, clean_temp_files=True):
        """
        Generate a preview image report for CADt
        Args:
            study_deploy_results (StudyDeployResults): Object that contains the results of the study
            ram_images (dict): Dictionary of preprocessed images saved in ram memory
            clean_temp_files (bool): remove temporary files

        Returns:
            str-dict. Dictionary with a single entry ('dcm_output') with the path to the generated file
        """
        # If there is no result from engine.
        if study_deploy_results.results is None:
            self.logger.info('There is no study_results. Skip to generate {} report.'.format(__name__))
            return None

        study_results = results_parser.get_study_results(study_deploy_results.results)
        cat = study_results['total']['category']

        # category 0 means not suspicious thus doesnt need to create a preview image
        # study is suspicious
        if cat == 0:
            self.logger.info('Study not suspicious')
            return None

        output_dir = study_deploy_results.output_dir

        # Create temp html folder
        self._temp_html_folder = os.path.join(output_dir,
                                  f'{self._config[Config.MODULE_REPORTS, "temp_html_files_dir"]}_{self._report_prefix}')
        if os.path.isdir(self._temp_html_folder):
            shutil.rmtree(self._temp_html_folder)
        os.makedirs(self._temp_html_folder)

        # Create the MostMalignantImage in the temp folder, since it's just a temp file
        mm_report_path = os.path.join(
            self._temp_html_folder,
            self._most_malig_img_report.get_output_file_name(study_deploy_results.get_studyUID())
        )
        self._most_malig_img_file_path = self._most_malig_img_report.generate(study_deploy_results, ram_images,
                                                                              custom_path=mm_report_path)
        assert self._most_malig_img_file_path == mm_report_path, \
            "The expected path for the MostMalignantImage report does not match the expected. " \
            f"Expected: {mm_report_path}; Got: {self._most_malig_img_file_path}"
        self.ram_images = ram_images
        metadata = study_deploy_results.metadata
        file_name = self.get_output_file_name(study_deploy_results.get_studyUID())

        self._temp_html_report = os.path.join(self._temp_html_folder, "report.html")
        self._temp_jpeg = os.path.join(self._temp_html_folder, "temp_pdf.jpeg")

        # The patient metadata will be the same for all the rows
        patient_info = metadata.iloc[0]
        #modality = 'DBT' if len(DicomTypeMap.get_type_df(metadata, 'dbt')) > 0 else 'DM'

        self._report_params = {}

        if 'PatientName' in patient_info:
            self._report_params['dh-patient-name'] = patient_info['PatientName']
        else:
            self.logger.warning(log_line(-1, 'Patient name not found or wrong format'))
            raise ValueError('Patient name not found or wrong format')

        if 'PatientID' in patient_info:
            self._report_params['dh-pid'] = patient_info['PatientID']
        else:
            self.logger.warning(log_line(-1, 'PatientID name not found or wrong format'))
            raise ValueError("PatientID name not found or wrong format")

        if 'AccessionNumber' in patient_info:
            self._report_params['dh-accession-number'] = patient_info['AccessionNumber']
        else:
            self.logger.warning(log_line(-1, 'Accession Number not found or wrong format'))
            raise ValueError("Accession Number not found or wrong format")

        if 'StudyDate' in patient_info:
            date_ = patient_info['StudyDate']
            if date_ is not None:
                # YYYY/MM/DD
                date_ = "{}/{}/{}".format(date_[:4], date_[4:6], date_[6:])
            self._report_params['dh-exam-date'] = date_
        else:
            logging.warning('StudyDate not found or wrong format')
            raise ValueError('StudyDate not found or wrong format')

        self._report_params["dh-most-malig"] = self._most_malig_img_file_path

        self._create_html()
        pdf_output = os.path.join(self._temp_html_folder, file_name.replace('.dcm', '.pdf'))
        self._create_pdf(pdf_output)

        dcm_output = os.path.join(output_dir, file_name)

        pdf_img = pdf2image.convert_from_path(pdf_output)[0]
        pdf_img.save(self._temp_jpeg)

        cmd_str = 'img2dcm -sef {} {} {} '.format(metadata.iloc[0]['dcm_path'], self._temp_jpeg, dcm_output)
        out_code = subprocess.call(cmd_str, shell=True)
        assert out_code == 0, "There was an error in the img2dcm. Command: {}".format(cmd_str)
        self.logger.info(log_line(-1, f"{dcm_output} generated"))

        if clean_temp_files:
            shutil.rmtree(self._temp_html_folder)
            self.logger.debug(log_line(-1, '{} folder removed'.format(self._temp_html_folder)))
        return {'dcm_output': dcm_output}

    def _create_html(self):
        """
        Create the html file that will be used to generate the PDF.
        If the process finishes ok, self._temp_html_report will contain the path to the generated file
        :param report_params: dictionary. Parameters used for the html generation
        """
        html_template = self._source_folder + 'index_preview.html'

        # Read the template
        with open(html_template, 'r') as fp:
            s = fp.read()
            # Append the root path to the dh_resources folder to convert them into absolute paths
            s = s.replace("dh_resources/", self._source_folder + "dh_resources/")
            # Read new html for parsing
            soup = BeautifulSoup(s, 'lxml')

        image_changed = False
        for key, val in list(self._report_params.items()):
            tag = soup.find(id=key)
            if tag is not None and val is not None:
                if tag.name == 'img':
                    # Image source
                    tag.attrs['src'] = val
                    if key =="dh-most-malig":
                        image_changed =True
                else:
                    # Just update the text
                    tag.string = val

        if not image_changed:
            raise ValueError("Place holder image is not replaced with correct image.")

        # Truncate patient info texts
        patient_info_div = soup.find(id="dh-patient-info-div")
        for tag in patient_info_div.find_all("h6"):
            tag.string = self._truncate_text(tag.string, self.max_length_texts['patient_info'])



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
        }

        # Using PDFKit to generate PDF from HTML
        temp_pdf_path = output_pdf_path + ".uncropped.pdf"
        pdfkit.from_file(self._temp_html_report, temp_pdf_path, options=options)
        # Crop the white margins in the pdf file
        proc = subprocess.run("pdf-crop-margins -p 0 {} -o {}".format(temp_pdf_path, output_pdf_path), shell=True)
        proc.check_returncode()
        # Remove uncropped pdf
        os.remove(temp_pdf_path)
        self.logger.debug(log_line(-1, '{} generated'.format(output_pdf_path)))


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
