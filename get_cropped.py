import cv2
import os

# ---------------------------------------------------------------------------- #
# This program takes pre-annotated image data and crops on the annotations to 
# create smaller images to be masked and used in the creation of synthetic
# object deteciton data
# ---------------------------------------------------------------------------- #


DATASET_PATH = r"C:\Users\Ebuka Amadi-Obi\Documents\GitHub\BreadBuddy\source_datasets\digi_smybols\\"
OUTPUT_PATH = r"C:\Users\Ebuka Amadi-Obi\Documents\GitHub\BreadBuddy\cropped_images\digi-sym\\"
CLASSLIST = ['AC_Source', 'BJT', 'Battery', 'Capacitor', 'Current_Source', 'DC_Source', 'Diode', 'Ground', 'Inductor', 'MOSFET', 'Resistor', 'Voltage_Source']
CLASSNUMBER = len(CLASSLIST)
CROP_BUFFER = 0

# Dict to hold number of images in each folder according to class
file_number_dict = dict(zip(range(CLASSNUMBER),([0]*CLASSNUMBER)))
        
for image_file in os.listdir(DATASET_PATH+r"images"):
    print("\nCropping images in " + image_file)

    # Load the image
    image = cv2.imread(DATASET_PATH+r"images\\"+image_file)

    # Convert the YOLO annotations to pixel coordinates
    image_height, image_width, _ = image.shape

    #Get corresponding annotation name in image folder
    name,ext = os.path.splitext(image_file)
    annot_file = open(DATASET_PATH+r"labels\\"+name+".txt", "r")

    #loop through lines in annotation
    for line in annot_file:

        #get bounding box values
        annot = line.split()
        class_index = annot[0]
        x_center = float(annot[1]) * float(image_width)
        y_center = float(annot[2]) * float(image_height)
        width = float(annot[3]) * float(image_width)
        height = float(annot[4]) * float(image_height)

        #cropped image size
        x_min = int(x_center - width / 2) - CROP_BUFFER
        y_min = int(y_center - height / 2) - CROP_BUFFER
        x_max = int(x_center + width / 2) + CROP_BUFFER
        y_max = int(y_center + height / 2) + CROP_BUFFER

        #if padding on bounding box goes outside image borders, set value to maximum possible
        if (x_min < 0):
            x_min = 0

        if (y_min < 0):
            y_min = 0

        if (x_max > image_width):
            x_max = image_width

        if (y_max > image_height):
            y_max = image_height

        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Get full paths
        output_image_path = os.path.join(OUTPUT_PATH,CLASSLIST[int(class_index)])
        #label_path = os.path.join(OUTPUT_PATH+"labels",CLASSLIST[int(class_index)])

        # Make directory for new images
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)

        # Get number for second half of filename
        file_number = file_number_dict[int(class_index)]

        # Add class name and file number together along with extension to form filename
        output_image_filename = class_index+"_"+str(file_number)+".jpg" 
        #output_label_filename = class_index+"_"+str(file_number)+".txt"

        # Increment class' number of files in dict - another file has been written
        file_number_dict[int(class_index)] = file_number+1

        # Write image file
        cv2.imwrite((output_image_path +"\\"+ output_image_filename), cropped_image)   

        print("     Wrote: " + output_image_filename)    

print("All Done!")