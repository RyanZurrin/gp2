import subprocess
from pathlib import Path
import io
import pydicom
import matplotlib.pyplot as plt
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid
from PIL import ImageDraw
from PIL.FontFile import WIDTH
from PIL.Image import Image
import gdcm
from .dicom_data_fixer import DicomDataFixer


class DicomCopyMachine:
    """
    This class will take two lists or two dicoms A and B and copy all the header
    information from A and pixel information from B and make a new dicom C
    and save it as new dicom.
    data on each
    dicom intact.
    """

    def __init__(self, dicom_a, dicom_b, save_path=None, compression_type=None):
        """
        Parameters
        ----------
        dicom_a : pydicom.dataset.FileDataset
            The dicom data
        dicom_a : pydicom.dataset.FileDataset
            The dicom data
        save_path : Path
            The path to save the new dicom to
        """
        self.dicom_A = dicom_a
        self.dicom_B = dicom_b
        self.save_path = Path(save_path)
        self.compression_type = compression_type
        self.dicom_C = self.copy_dicom()

        print(f'New dicom saved to {self.save_path}')

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
        new_dicom = self.dicom_A

        new_dicom.file_meta.TransferSyntaxUID = \
            self.dicom_B.file_meta.TransferSyntaxUID
        # set all the (0028, XXXX) tags to the same as dicom B
        # samples per pixel
        new_dicom.SamplesPerPixel = self.dicom_B.SamplesPerPixel
        # photometric interpretation
        new_dicom.PhotometricInterpretation = \
            self.dicom_B.PhotometricInterpretation
        # rows
        new_dicom.Rows = self.dicom_B.Rows
        # columns
        new_dicom.Columns = self.dicom_B.Columns
        # bits allocated
        new_dicom.BitsAllocated = self.dicom_B.BitsAllocated
        # bits stored
        new_dicom.BitsStored = self.dicom_B.BitsStored
        # high bit
        new_dicom.HighBit = self.dicom_B.HighBit
        # pixel representation
        new_dicom.PixelRepresentation = self.dicom_B.PixelRepresentation
        # pixel padding value
        new_dicom.PixelPaddingValue = self.dicom_B.PixelPaddingValue
        # pixel intensity relationship
        new_dicom.PixelIntensityRelationship = \
            self.dicom_B.PixelIntensityRelationship
        # pixel intensity relationship sign
        new_dicom.PixelIntensityRelationshipSign = \
            self.dicom_B.PixelIntensityRelationshipSign
        # window center
        new_dicom.WindowCenter = self.dicom_B.WindowCenter
        # window width
        new_dicom.WindowWidth = self.dicom_B.WindowWidth
        # rescale intercept
        new_dicom.RescaleIntercept = self.dicom_B.RescaleIntercept
        # rescale slope
        new_dicom.RescaleSlope = self.dicom_B.RescaleSlope
        # rescale type
        new_dicom.RescaleType = self.dicom_B.RescaleType
        # VOI LUT Function
        new_dicom.VOILUTFunction = self.dicom_B.VOILUTFunction
        # partial view
        new_dicom.PartialView = self.dicom_B.PartialView
        # Lossy Image Compression
        new_dicom.LossyImageCompression = self.dicom_B.LossyImageCompression

        # use gdcm to decompress the pixel data and then re-compress it with
        # the new compression type and set the pixel data to the new compression
        # type
        # decompress the pixel data
        decompressed_pixel_data = gdcm.ImageReader()
        decompressed_pixel_data.SetFileName(self.dicom_B.filename)
        decompressed_pixel_data.Read()
        decompressed_pixel_data = decompressed_pixel_data.GetImage()
        decompressed_pixel_data = decompressed_pixel_data.GetBuffer()
        # set the pixel data to the new compression

        new_dicom.PixelData = decompressed_pixel_data

        new_dicom.PixelData = self.dicom_B.PixelData

        # if self.compression_type is None:
        #     # if none use the compression from A and pixel data from B
        #     self.compression_type = self.dicom_A.file_meta.TransferSyntaxUID
        # # set the transfer syntax to the new compression
        # new_dicom.file_meta.TransferSyntaxUID = self.compression_type
        # # set the pixel data to the new compression
        # new_dicom.PixelData = encapsulate([self.dicom_B.PixelData])


        # add the InstanceNumber tag from dicom B
        new_dicom.InstanceNumber = self.dicom_B.InstanceNumber

        # remove the VOI LUT Sequence tag from dicom C
        if hasattr(new_dicom, 'VOILUTSequence'):
            del new_dicom.VOILUTSequence

        if self.save_path is not None:
            new_dicom.save_as(self.save_path, write_like_original=False)
        return new_dicom

    def change_compression(self, new_compression=None):
        """
        Change the compression of the dicom C to the new compression
        """

        # save the dicom

    def dicom_fixer(self):
        """
        Calls the DicomDataFixer class to fix dicom C using python to call using
        terminal commands
        """
        # python /hpcstor6/scratch01/r/ryan.zurrin001/scripts/dicom_data_fixer.py -d save_path -o save_path
        save_path = Path(self.save_path)
        print(f'Running dicom fixer on files located at {save_path.parent}')
        subprocess.run(['python',
                        '/hpcstor6/scratch01/r/ryan.zurrin001/scripts/dicom_data_fixer.py',
                        '-d', save_path, '-o', save_path.parent])

    def view_dicom(self):
        """
        View the dicom using matplotlib
        """
        plt.imshow(self.dicom_C.pixel_array, cmap='gray')
        plt.show()
