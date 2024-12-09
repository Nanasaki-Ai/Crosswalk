# Generate black-and-white video frames from trajectory data
# Black represents the background, white represents pedestrians
# Compared to the basic version, pedestrians' initial brightness is 128
# This code will be used later in preprocessing_att_region.py
# to further add a single vehicle with brightness 255
# tools.py contains the functions used

import os
from tqdm import tqdm
from PIL import Image, ImageDraw
from tools import *

def create_trajectory_images(data_dict, folder_name, image_size=(1200, 1100)):
    os.makedirs(folder_name, exist_ok=True)  # Create the corresponding folder B
    
    for frame_id, boxes in tqdm(data_dict.items(), desc=f"Processing {folder_name}"):
        # Create a single-channel image, initialized to 0 (black)
        frame_image = Image.new('L', image_size, 0)
        
        # Get the drawing handle
        draw = ImageDraw.Draw(frame_image)
        
        # Iterate over all boxes in the current frame
        for box in boxes:
            # Get the coordinates of the box
            left = box['left']
            top = box['top']
            right = box['right']
            bottom = box['bottom']
            cls = box['label']
            
            if cls == 1:
                # Draw a rectangle: set the box area to 128 (gray)
                draw.rectangle([left, top, right, bottom], fill=128)
        
        # Format frame_id as an 8-digit string
        formatted_frame_id = f"{frame_id - 1:08d}.jpg"

        # Save the image to the corresponding folder
        frame_image.save(os.path.join(folder_name, formatted_frame_id))

def process_trajectory_folder(folder_trajectory, folder_output):
    # Ensure the output folder exists
    os.makedirs(folder_output, exist_ok=True)
    
    # Iterate over all txt files in the folder
    for file_name in os.listdir(folder_trajectory):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_trajectory, file_name)
            data_dict = read_trajectory_txt(file_path)  # Read the trajectory file
            
            # Create the corresponding subfolder in the output folder
            base_name = os.path.splitext(file_name)[0]
            output_folder = os.path.join(folder_output, base_name)
            
            # Create trajectory images
            create_trajectory_images(data_dict, output_folder)

# Please replace the following paths with the actual paths
folder_trajectory = 'tracking_output'
folder_output = 'generated_ped_images'

# Start processing the trajectory folder
process_trajectory_folder(folder_trajectory, folder_output)
