# Preprocess trajectory data
# Read the IDs of the vehicles of interest based on the preprocessed trajectory files
# Start when a vehicle of interest enters the crosswalk area
# End when a vehicle of interest leaves the crosswalk area
# The files generated here can be used for
# Matching with ground truth labels and comparative experiments
# tools.py contains the functions used

import os
from tqdm import tqdm

from tools import *

# Path settings
folder_trajectory = 'tracking_output'  # Path to the trajectory folder
folder_filtering = 'filtering_output'  # Path to the preprocessed trajectory folder
folder_filtering_woatt = 'filtering_woatt_output'  # Path to the preprocessed trajectory folder

# Ensure the output folders exist
os.makedirs(folder_filtering, exist_ok=True)
os.makedirs(folder_filtering_woatt, exist_ok=True)

# Upper crosswalk area
# The upper and lower line segments are represented by two endpoints,
# note that the pedestrian area is represented by the vertices of a quadrilateral
line_segment_top1 = ((65.0, 153.0), (1028.0, 165.0))
line_segment_top2 = ((57.0, 331.0), (1049.0, 313.0))
poly_top = [[0.0, 120.0], [1200.0, 120.0], [1200.0, 480.0], [0.0, 480.0]]

# Lower crosswalk area
line_segment_bot1 = ((338.0, 589.0), (937.0, 377.0))
line_segment_bot2 = ((398.0, 676.0), (951.0, 458.0))
poly_bot = [[120.0, 480.0], [1100.0, 240.0], [1100.0, 600.0], [240.0, 1000.0]]

# Get a list of all .txt files in the trajectory folder
txt_files = [f for f in os.listdir(folder_trajectory) if f.endswith('.txt')]

num = 0

# Iterate through all files using tqdm
for file_name in tqdm(txt_files, desc='Processing files'):    
    
    # Construct the path to the file to be read
    file_path = os.path.join(folder_trajectory, file_name)
    filtering_file_top_path = os.path.join(folder_filtering, f"{file_name.split('.')[0]}_top.txt")
    filtering_file_bot_path = os.path.join(folder_filtering, f"{file_name.split('.')[0]}_bot.txt")

    # Read the original trajectory and preprocessed trajectory data
    result_dict = read_trajectory_txt(file_path)
    result_dict_top = read_filtering_trajectory_txt(filtering_file_top_path)
    result_dict_bot = read_filtering_trajectory_txt(filtering_file_bot_path)

    # Construct the path to the output file
    output_file_top_path = os.path.join(folder_filtering_woatt, f"{file_name.split('.')[0]}_woatt_top.txt")
    output_file_bot_path = os.path.join(folder_filtering_woatt, f"{file_name.split('.')[0]}_woatt_bot.txt")
        
    # Open output file
    output_file_top = open(output_file_top_path, "w")
    output_file_bot = open(output_file_bot_path, "w")
    
    att_veh_top_ids = set()  # Use Set to automatically remove duplicates
    att_veh_bot_ids = set()  
    
    for frame_id, frame_data in result_dict_top.items():
        for box in frame_data:  # Traverse the bounding box in each frame
            att_veh_top_ids.add(box['id'])  # Add the id to the set
        
    for frame_id, frame_data in result_dict_bot.items():
        for box in frame_data:  
            att_veh_bot_ids.add(box['id'])  

    att_veh_top = list(att_veh_top_ids)
    att_veh_bot = list(att_veh_bot_ids)       

    num += len(att_veh_top)
    num += len(att_veh_bot)
    # print('top', att_veh_top)
    # print('bot', att_veh_bot)
    
    for frame_id, frame_data in result_dict.items():
        output_file_top.write(f"{frame_id}\n")
        output_file_bot.write(f"{frame_id}\n")

        att_id_top = []
        att_id_bot = []
        
        for box in frame_data:
            if box['label'] == 0:  # Vehicle
                veh_top_left = (box['left'], box['top'])
                veh_bottom_right = (box['right'], box['bottom'])
                
                result_top = is_intersection(line_segment_top1, veh_top_left, veh_bottom_right) or \
                             is_intersection(line_segment_top2, veh_top_left, veh_bottom_right)            
                result_bot = is_intersection(line_segment_bot1, veh_top_left, veh_bottom_right) or \
                             is_intersection(line_segment_bot2, veh_top_left, veh_bottom_right)
                             
                if result_top and (box['id'] in att_veh_top):
                    att_id_top.append(box['id'])
                if result_bot and (box['id'] in att_veh_bot):
                    att_id_bot.append(box['id'])

        output_file_top.write(f"{len(att_id_top)}\n")
        output_file_bot.write(f"{len(att_id_bot)}\n")
    
        for box in frame_data:
            if box['id'] in att_id_top:
                output_file_top.write(f"{box['id']} {box['left']} {box['top']} {box['right']} {box['bottom']}\n")
            if box['id'] in att_id_bot:
                output_file_bot.write(f"{box['id']} {box['left']} {box['top']} {box['right']} {box['bottom']}\n")  

print(num)                