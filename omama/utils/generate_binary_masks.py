import cv2
import os

input_folder = "/Users/kamisettyvivek/Downloads/SN7_buildings_train/train/L15-0331E-1257N_1327_3160_13/images"
output_folder = "/Users/kamisettyvivek/Downloads/SN7_buildings_train/train/L15-0331E-1257N_1327_3160_13/binary_masks"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set the threshold value (adjust as needed)
threshold = 127

# Iterate over each image in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        # Load the image
        img = cv2.imread(os.path.join(input_folder, file_name), cv2.IMREAD_GRAYSCALE)
        
        # Apply binary thresholding to generate the binary mask
        ret, binary_mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        
        # Save the binary mask to the output folder with the same name as the original image
        cv2.imwrite(os.path.join(output_folder, file_name), binary_mask)
