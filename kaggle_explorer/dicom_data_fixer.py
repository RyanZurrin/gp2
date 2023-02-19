import random
import string
import time
import argparse
import pydicom
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset
import pandas as pd
import os

DICOM_DIR = r'/raid/mpsych/kaggle_mammograms/original/train_images/10006/'
OUTPUT_DIR = None
CSV_PATH = r'/raid/mpsych/kaggle_mammograms/train.csv'
PARAMETERS = {
    'PatientName': '',
    'PatientBirthDate': '',
    'PatientSex': '',
    'StudyDate': '',
    'StudyTime': '',
    'SeriesDate': '',
    'ContentDate': '',
    'InstitutionName': '',
    'AcquisitionDeviceProcessingCode': '',
    'PartialView': '',
    'PartialViewDescription': '',
    'StudyPriorityID': '',
    'ScheduledStudyStartTime': '',
    'RouteOfAdmissions': '',
    'RequestedProcedurePriority': '',
    'SOPClassUID': '1.2.840.10008.5.1.4.1.1.1.2',
    'Manufacturer': 'Hologic',
    'ManufacturerModelName': 'Selenia Dimensions',
    'ViewCodeSequence': '',
}

# use argparse to get the parameters from the command line if you want both short and long options
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dicom_dir', help='Path to dicom directory',
                    type=str, default=DICOM_DIR)
parser.add_argument('-o', '--output_dir', help='Path to output directory',
                    type=str, default=OUTPUT_DIR)
parser.add_argument('-c', '--csv_path', help='Path to csv file',
                    type=str, default=CSV_PATH)
parser.add_argument('-p', '--parameters', help='Parameters to change',
                    type=str, default=PARAMETERS)
args = parser.parse_args()


class DicomDataFixer:
    def __init__(self,
                 dicom_dir,
                 csv_path,
                 output_dir=None,
                 inplace=False,
                 **kwargs):
        self.dicom_dir = dicom_dir
        self.image_paths = self.get_image_paths()
        self.output_dir = output_dir
        self.csv_path = csv_path
        self.inplace = inplace
        self.csv_df = self.read_csv()
        self.parameters = kwargs

    def read_csv(self):
        print('Reading csv file: {}'.format(self.csv_path))
        csv_data = pd.read_csv(self.csv_path)
        return csv_data

    def get_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.dicom_dir):
            for file in files:
                if file.endswith(".dcm"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def get_age(self, dicom_data):
        inst_num = dicom_data.InstanceNumber
        age = self.csv_df.loc[self.csv_df['image_id'] == inst_num, 'age'].iloc[
            0]
        age = age * 12
        return str(age) + 'm'

    def get_veiw(self, dicom_data):
        inst_num = dicom_data.InstanceNumber
        view = \
            self.csv_df.loc[self.csv_df['image_id'] == inst_num, 'view'].iloc[0]
        return view

    def get_breast_implant_present(self, dicom_data):
        inst_num = dicom_data.InstanceNumber
        breast_implant_present = \
            self.csv_df.loc[
                self.csv_df['image_id'] == inst_num, 'implant'].iloc[0]
        if breast_implant_present == 1:
            return 'YES'
        else:
            return 'NO'

    def get_view_code_sequence_value(self, dicom_data):
        inst_num = dicom_data.InstanceNumber
        view = \
            self.csv_df.loc[self.csv_df['image_id'] == inst_num, 'view'].iloc[0]
        if view == 'CC':
            return "R-10242"
        elif view == 'MLO':
            return "R-10226"

    def get_view_code_meaning_value(self, dicom_data):
        inst_num = dicom_data.InstanceNumber
        view = \
            self.csv_df.loc[self.csv_df['image_id'] == inst_num, 'view'].iloc[0]
        if view == 'CC':
            return "cranio-caudal"
        elif view == 'MLO':
            return "mediolateral-oblique"

    def generate_accession_number(self):
        """
        Generate a random AccessionNumber of 12 characters all uppercase
        Returns
        -------
        accession_number : str
            A random AccessionNumber of 12 characters all uppercase
        """
        accession_number = ''.join(random.choices(string.ascii_uppercase,
                                                  k=12))
        return accession_number

    def get_patient_orientations(self, dicom_data):
        """
        Get the patient orientation from the csv file if laterality is R then
        use ["P", "L"], if laterality is L then use ["A", "R"]
        Parameters
        ----------
        dicom_data : pydicom.dataset.FileDataset
            The dicom data
        Returns
        -------
        patient_orientation : str
            The patient orientation
        """
        inst_num = dicom_data.InstanceNumber
        laterality = self.csv_df.loc[
            self.csv_df['image_id'] == inst_num, 'laterality'].iloc[0]
        if laterality == 'R':
            patient_orientation = ["P", "L"]
        else:
            patient_orientation = ["A", "R"]
        return patient_orientation

    def fix_dicoms(self):
        t0 = time.time()
        count = 0
        print(f'Fixing {len(self.image_paths)} dicom files in {self.dicom_dir}')
        for dicom_loc in self.image_paths:
            dicom_data = pydicom.dcmread(dicom_loc, force=True)
            dicom_data.PatientAge = self.get_age(dicom_data)
            dicom_data.ViewPosition = self.get_veiw(dicom_data)
            dicom_data.BreastImplantPresent = self.get_breast_implant_present(
                dicom_data)
            dicom_data.PatientOrientation = self.get_patient_orientations(
                dicom_data)
            dicom_data.AccessionNumber = self.generate_accession_number()

            dsv = Dataset()
            dsv.CodeValue = self.get_view_code_sequence_value(dicom_data)
            dsv.CodeMeaning = self.get_view_code_meaning_value(dicom_data)

            dicom_data.ViewCodeSequence = Sequence([dsv])

            for key, value in self.parameters.items():
                setattr(dicom_data, key, value)
                # save the dicom data
                if self.inplace:
                    dicom_data.save_as(dicom_loc)
                else:
                    dicom_data.save_as(os.path.join(self.output_dir,
                                                    os.path.basename(
                                                        dicom_loc)),
                                       write_like_original=False)
            if count % 100 == 0:
                print(
                    f'Finished fixing {count} dicom files in '
                    f'{time.time() - t0} seconds.')
            count += 1

        print(f'Finished fixing {count} dicom files in {time.time() - t0} '
              f'seconds.')


if __name__ == '__main__':
    # create the object
    fixer = DicomDataFixer(args.dicom_dir, args.csv_path, args.output_dir,
                           args.inplace, **vars(args))

    # fix the dicoms
    fixer.fix_dicoms()
