from PIL import Image
import os
import numpy as np
import cv2
from deephealth_utils.misc import results_parser
from deephealth_utils.data.utils import dcm_to_preprocessed_array
from centaur.centaur_engine.helpers.helper_preprocessor import get_orientation_change
from centaur.centaur_engine.helpers.helper_results_processing import orient_bounding_box_coords


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    COPIED FROM SCIPY SOURCE CODE, NOW DEPRECATED
    """
    if data.dtype == np.uint8:
        return data
    if high < low:
        raise ValueError("`high` should be larger than `low`.")
    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()
    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1
    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


def plot_box(im, coords, slice_num, color, thickness, score, additional_text='', text_color=(255,255,255), text_scale=1):
    """
    Plots bounding boxes on images in place and adding text indicating score and any other additional texts
    Args:
        im: numpy array,  representing image pixels
        coords: tuple or list of integers, coordinates of the boxes (x1, y1, x2, y2)
        slice_num: int, slice number of the box
        color: tuple of ints, color of the bounding box represented as a tuple of ints of length three
        thickness: int, thickness of the box
        score: float, score of the bounding box
        additional_text: str, any additional text to be added next to all bounding boxes
        text_color: color of the text
        text_scale: font size of the text

    Returns:

    """
    x1, y1, x2, y2 = coords
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # values of bounding boxes has to make sense
    assert 0 <= x1 <= x2 <= im.shape[1], "Got unexpected x values"
    assert 0 <= y1 <= y2 <= im.shape[0], "Got unexpected y values"
    cv2.rectangle(im, (x1, y1), (x2, y2), thickness=thickness, color=color)
    cv2.putText(
        im,
        "{}, {}, {}".format(score, slice_num, additional_text),
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
    )




def plot_dicom(img, boxes, gt_boxes={}, pred_color= (255, 0, 0), gt_color= (0, 255, 0),
               plot_gt=True, thickness=3, is_dbt=False, slice_number=None):
    """
    Plot on specific dicom, plots the predicted boxes and ground truth boxes if specified
    Args:
        img: numpy array, representing image pixels,or str specifying the path of dicom
        boxes: dictionary, representation of a bounding box
        gt_boxes: pandas DataFrame, that contains the ground truth bounding boxes or a list of dictionaries
        pred_color: tuple of int, specifying the color to plot for the predicted bounding boxes
        gt_color: tuple of int, specifying the color to plot for the ground truth bounding boxes
        plot_gt: bool, specifying whether or not to also plot the ground truth bounding boxes
        thickness: int, specifying the thickness of the bounding box lines
        is_dbt: bool, specifying whether or not the dicom is dbt
        slice_number: if the file is dbt, slice_number specifies which slice to plot.

    Returns:

    """
    # check if its a img pixels passed in or img path
    if isinstance(img,str):

        pixels = dcm_to_preprocessed_array(img)
        # if the dicom is dbt we take a slice of it
        if is_dbt:
            assert pixels.ndim==3, "Expected number of dimensions for DBT pixels to 3, \
            got {} dimensions instead".format(pixels.ndim)
            assert slice_number is not None, "please specify a slice_number"
            pixels = pixels[slice_number]
    else:
        pixels = img

    im_cp = bytescale(pixels.copy())

    # images are expected to have 3 channels
    im_cp = np.stack([im_cp, im_cp, im_cp], axis=-1)



    # plot all boxes
    for box in boxes:
        if "text" in box:
            additonal_text = box["text"]
        else:
            additonal_text = ''
        plot_box(
            im_cp,
            box["coords"],
            box["slice"],
            pred_color,
            thickness,
            round(box["score"], 4),
            additional_text=additonal_text
        )

    # if we are not plotting ground truth return the image and we are done
    if not plot_gt:
        return im_cp

    if len(gt_boxes) > 0:
        for box in gt_boxes:
            plot_box(
                im_cp,
                box["coords"],
                box["slice"],
                gt_color,
                thickness,
                box["score"],
            )

    return im_cp

def plot_box_on_synthetics(processed_study,
                            gt_boxes={},
                            pred_color=(255, 0, 0),
                            gt_color=(0,255,0),
                            save_dir=None,
                            mode="ground_truth",
                            bt_dicoms=None):

    """
    Given the processed_study,
    Args:
        processed_study: study_results object, contains results of centaur runs
        study_instance_uid: str, specifying the study
        gt_df: pandas DataFrame, specifiying ground truth bounding boxes
        pred_color:tuple of int, specifying the color to plot for the predicted bounding boxes
        gt_color: tuple of int, specifying the color to plot for the ground truth bounding boxes
        save_dir: str, location to store images
        mode: str, specifying what kind of boxes to plot
        bt_dicoms, list or NoneType, specifying which files are DBT studies

    Returns: list of images

    """

    # only BT files have synthetic
    if bt_dicoms is None:
        # get all the file names of that study
        input_file_names = [x.split("/")[-1] for x in processed_study.input_files]
        bt_dicoms = [x[3:] for x in input_file_names if x[:3] == 'BT.']


    all_images = []

    # get both the processed results and raw results
    results_processed = results_parser.load_results_json(os.path.join(processed_study.study_output_dir, 'results.json'))
    results_raw = results_parser.load_results_json(os.path.join(processed_study.study_output_dir, 'results_raw.json'))

    # if we are plotting non_NMS we need to use the raw results otherwise we are using the processed_results
    if mode == "non_NMS":
        results_dict = results_raw
    else:
        results_dict = results_processed

    # for each BT_dicom plot the boxes on synthetic images
    for dicom in bt_dicoms:
        synthetic_path = os.path.join(processed_study.study_output_dir, dicom, "synth.png")

        # not all BT files are ran by centaur, so BT files are skipped then we are also skipping those
        if not os.path.exists(synthetic_path):
            continue

        synthetic_image = Image.open(synthetic_path)
        synthetic_pixels = np.array(synthetic_image)

        
        if "model_results" in results_dict.keys():
            boxes = results_dict["model_results"]["dicom_results"][dicom]["none"]
        else:
            boxes = results_dict["dicom_results"][dicom]["none"]

        metadata = results_parser.get_metadata(results_processed)
        metadata = metadata[metadata["SOPInstanceUID"] == dicom]
        assert len(metadata) == 1, "multiple rows in metadata for one SOPInstanceUID"

        # check if boxes need to be roated
        laterality = metadata.ImageLaterality.values[0]
        patient_orientation = metadata.PatientOrientation.values[0]



        orientation_change = get_orientation_change(laterality, patient_orientation)

        for box in boxes:
            box["coords"] = orient_bounding_box_coords(box["coords"], synthetic_pixels.shape, orientation_change)


        # plotting both prediction boxes and ground truth boxes
        if mode == "ground_truth":
            if dicom in gt_boxes:
                gt_df_dicom = gt_boxes[dicom]

            else:
                gt_df_dicom = {}

            for box in gt_df_dicom:
                box["coords"] = orient_bounding_box_coords(box["coords"], synthetic_pixels.shape, orientation_change)

            plotted_img = plot_dicom(synthetic_pixels, boxes, gt_df_dicom,
                                         pred_color, gt_color, plot_gt=True)

            suffix = "_boxes.png"

        # plotting prediction boxes only (with NMS algorithm)
        elif mode == "NMS":
            plotted_img = plot_dicom(synthetic_pixels, boxes,
                                     pred_color, gt_color, plot_gt=False)
            
            suffix = "_with_NMS.png"
        # plotting prediction boxes only (without NMS algorithm)
        elif mode == "non_NMS":
            plotted_img = plot_dicom(synthetic_pixels, boxes,
                                     pred_color=(0, 255, 0), gt_color=(0, 0, 255), plot_gt=False)

            suffix = "_without_NMS.png"
        else:
            raise ValueError("mode {} not recognized".format(mode))

        im = Image.fromarray(plotted_img)
        all_images.append(im)
        if save_dir is not None:
            image_path = os.path.join(save_dir, dicom + suffix)
            im.save(image_path)

    return all_images
