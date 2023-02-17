import pydicom
import matplotlib.pyplot as plt


class DicomCopyMachine:
    """
    This class will take two lists or two dicoms A and B and copy all the header
    information from A and pixel information from B and make a new dicom C
    and save it as new dicom.
    data on each
    dicom intact.
    """

    def __init__(self, dicom_a, dicom_b, save_path=None):
        """
        Parameters
        ----------
        dicom_a : pydicom.dataset.FileDataset
            The dicom data
        dicom_a : pydicom.dataset.FileDataset
            The dicom data
        save_path : str
            The path to save the new dicom to
        """
        self.dicom_A = dicom_a
        self.dicom_B = dicom_b
        self.save_path = save_path
        self.dicom_C = self.copy_dicom()

    def copy_dicom(self):
        """
        Copy the dicom A header info and replace the pixel data and pixel
        transfer syntax with the pixel data and transfer syntax from
        dicom B. This will use proper encapsulation as the pixel data is compressed.

        Jpeg fromat needs to start with 0xFFD8 and end with 0xFFD9
        Jpeg2000 needs to start with 0xFF4F and end with 0xFFD9 so make sure
        this is the case.

        Returns
        -------
        new_dicom : pydicom.dataset.FileDataset
            The new dicom
        """


    def view_dicom(self):
        """
        View the dicom using matplotlib
        """
        plt.imshow(self.dicom_C.pixel_array, cmap=plt.cm.bone)
        plt.show()
