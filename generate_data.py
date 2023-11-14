import cv2
import numpy as np
import os


# ---------------------------------------------------------------------------- #
# This script takes data masks and cropped image data of hand-drawn components
# and adds them to backgrounds to make synthetic data for training my
# object detection model.
#
# This script assumes the proper file stricture and naming scheme       
# ---------------------------------------------------------------------------- #


# File Paths
# Path to images folder containing cropped images and masks
IMAGES_DIR = r"C:\Users\Ebuka Amadi-Obi\Documents\GitHub\BreadBuddy\synthetic_data\\"
# Path to background images
BG_DIR = r"C:\Users\Ebuka Amadi-Obi\Documents\GitHub\BreadBuddy\synthetic_data\bg\\"
# Directory to output resulting images
OUTPUT_DIR = r"C:\Users\Ebuka Amadi-Obi\Documents\GitHub\BreadBuddy\synthetic_data\synth_data_output\\"
# Output Image dimensions
DIM = (640,640)
# Range of subject image dimensions (longest side)
SUBJECT_MIN=40
SUBJECT_MAX=100
# List of classes to be used in image generation
CLASSES = [
    "and",
    "capacitor"
]
NO_IMAGES = 10

# Input background image, subject, and, subject mask, returns background with image as well as annotation dimensions
def add_image(bg, img, mask, annotations, rotate = False):

    # Height and width of background
    bg_h = bg.shape[0]
    bg_w = bg.shape[1]
    
    # Randomly rotate image
    if (rotate == True):
        rotation = np.random.randint(0,2)
        match rotation:
            case 1:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            case 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_h = img.shape[0]
    img_w = img.shape[1]

    # Random values for location of top left corner of image
    img_location_x = np.random.randint(0, bg_w-img_w)
    img_location_y = np.random.randint(0, bg_h-img_h) 

    x_min = img_location_x
    y_min = img_location_y
    x_max = img_location_x+img_w
    y_max = img_location_y+img_h

    center_x = ((x_min+x_max)/2)/bg_w
    center_y = ((y_min+y_max)/2)/bg_h

    # Standardized values for image width and height
    img_w_adj = img_w/bg_w
    img_h_adj = img_h/bg_h 

    annotation_list = annotations.split("\n")
        
    # Make image layer to be overlayed on background
    img_layer = np.zeros((bg.shape[0], bg.shape[1], 3), dtype = "uint8")

    #add image to image layer
    img_layer[y_min:y_max, x_min:x_max] = img[0:img_h, 0:img_w]

    # Loop through pixels, find unmasked spots and replace background with image pixels
    for x in range(img_w):
        for y in range(img_h):
            img_pix = img[y,x]
            mask_value = mask[y,x]
            if all(mask_value > [128,128,128]):
                bg[img_location_y+y,img_location_x+x] = img_pix  

    return bg, center_x, center_y, img_w_adj, img_h_adj

no_bg_files = len(os.listdir(BG_DIR))

for j in range(NO_IMAGES):

    annotations = str()

    bg_index = np.random.randint(0,no_bg_files)

    # Get background filename and read in as CV2 RGB matrix
    bg_filename = os.listdir(BG_DIR)[bg_index]
    bg = cv2.imread(BG_DIR + bg_filename)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    for i in range(8):

        class_index = np.random.randint(0,2)
        #class_index = 1

        # Get image filename and read in as CV2 RGB matrix
        image_filename = os.listdir(IMAGES_DIR +CLASSES[class_index]+ r"\images")[i]         #test index to be swapped with random value
        img = cv2.imread(IMAGES_DIR +CLASSES[class_index]+ r"\images\\" + image_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("overlaying" + image_filename)

        # Get mask filename and read in as CV2 RGB matrix
        mask_filename = os.listdir(IMAGES_DIR +CLASSES[class_index]+r"\masks")[i]            #this index should be the same as masks one
        mask = cv2.imread(IMAGES_DIR +CLASSES[class_index]+ r"\masks\\" + mask_filename)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Resize background
        bg = cv2.resize(bg, DIM, interpolation = cv2.INTER_AREA)

        # Add image to background
        bg, center_x, center_y, img_w, img_h = add_image(bg, img, mask, annotations)

        # Add annotation to annotation list
        img_annotation = (center_x, center_y, img_w, img_h)
        img_annotation_str = str(img_annotation).translate(str.maketrans ('', '', "(,)"))
        class_index_str = str(class_index)
        annotations = annotations + class_index_str +" "+ img_annotation_str + "\n"
        
    # Create directories if not already exists
    if not os.path.exists(OUTPUT_DIR + r"images\\"):
        os.makedirs(OUTPUT_DIR + r"images\\")
    cv2.imwrite(OUTPUT_DIR + r"images\\"+ str(j)+"_image.jpg", bg)

    if not os.path.exists(OUTPUT_DIR + r"labels\\"):
        os.makedirs(OUTPUT_DIR + r"labels\\")    
    f = open(OUTPUT_DIR + r"labels\\"+ str(j)+"_annotation.txt", "w")
    f.write(annotations)
    f.close()

    """cv2.imshow("overlaid", bg)
    cv2.waitKey()"""



