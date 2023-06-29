import cv2 
import os
import numpy as np

#Directory of images to mask and resize
IMAGE_DIR = r"C:\Users\Ebuka Amadi-Obi\Documents\GitHub\BreadBuddy\cropped_dataset\capacitor\images\\"

# Directory of output files
OUTPUT_DIR = r"C:\Users\Ebuka Amadi-Obi\Documents\GitHub\BreadBuddy\synthetic_data\capacitor\\"

# Minimum and maximum image dimensions
SUBJECT_MIN = 40
SUBJECT_MAX = 100

for img_filename in os.listdir(IMAGE_DIR):

    img_path = IMAGE_DIR + img_filename

    # Read in image
    img = cv2.imread(img_path)

    # Get black & white version of image for thresholding
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Make mask using adaptive thresholding
    mask = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,21,10)

    img_h, img_w = img.shape[0], img.shape[1]
    
    longest, shortest = max(img_h, img_w), min(img_h, img_w)

    if SUBJECT_MIN>longest or SUBJECT_MAX<longest:

        longest_new = np.random.randint(SUBJECT_MIN,SUBJECT_MAX)
        shortest_new = int(shortest * (longest_new / longest))
        
        if img_h < img_w:
            img_h, img_w = longest_new, shortest_new
        else:
            img_h, img_w = shortest_new, longest_new
         
    # Resize image
    img_resized = cv2.resize(img, (img_h,img_w), interpolation = cv2.INTER_AREA)

    # Resize mask
    mask_resized = cv2.resize(mask, (img_h,img_w), interpolation = cv2.INTER_AREA)   

    first_name, ext = os.path.splitext(img_filename)

    # Mask directory
    mask_output = os.path.join(OUTPUT_DIR + "masks\\", first_name+"_mask.jpg")
    # Create directory if not already exists
    if not os.path.exists(OUTPUT_DIR + "masks\\"):
        os.makedirs(OUTPUT_DIR + "masks\\")

    # Image directory
    image_output = os.path.join(OUTPUT_DIR + "images\\", first_name+"_resized.jpg")
    # Create directory if not already exists
    if not os.path.exists(OUTPUT_DIR + "images\\"):
        os.makedirs(OUTPUT_DIR + "images\\")

    # Write files
    cv2.imwrite(mask_output, mask_resized)
    cv2.imwrite(image_output, img_resized)

    print('Masked and resized', img_filename)