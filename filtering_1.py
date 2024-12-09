# Preprocess trajectory data
# First, select the vehicles of interest, which are those that may not yield to pedestrians, and filter the objects of interest
# Start recording the vehicle IDs of interest when the vehicle enters the crosswalk area and there are pedestrians in the area
# End when the vehicle leaves the area or when the last pedestrian leaves the area
# tools.py contains the functions used

import os
from tqdm import tqdm

from tools import *

# Path settings
folder_trajectory = 'tracking_output'  # Path to the folder containing the original trajectory files
folder_filtering = 'filtering_output'  # Path to the folder containing the filtered trajectory files

# Ensure the output folder exists
os.makedirs(folder_filtering, exist_ok=True)

# Upper crosswalk area (the upper and lower line segments are represented by two endpoints,
# note that the pedestrian area is represented by the vertices of a quadrilateral)
line_segment_top1 = ((65.0, 153.0), (1028.0, 165.0))
line_segment_top2 = ((57.0, 331.0), (1049.0, 313.0))

poly_top = [[0.0, 120.0], [1200.0, 120.0], [1200.0, 480.0], [0.0, 480.0]]
# Lower crosswalk area
line_segment_bot1 = ((338.0, 589.0), (937.0, 377.0))
line_segment_bot2 = ((398.0, 676.0), (951.0, 458.0))
poly_bot = [[120.0, 480.0], [1100.0, 240.0], [1100.0, 600.0], [240.0, 1000.0]]

# Get a list of all .txt files in folder A
txt_files = [f for f in os.listdir(folder_trajectory) if f.endswith('.txt')]

# Iterate through all files using tqdm
for file_name in tqdm(txt_files, desc='Processing files'):
    file_path = os.path.join(folder_trajectory, file_name)
    
    # Read trajectory data
    result_dict = read_trajectory_txt(file_path)
    # print(result_dict)
    # Create the output file path
    output_file_top_path = os.path.join(folder_filtering, f"{file_name.split('.')[0]}_top.txt")
    output_file_bot_path = os.path.join(folder_filtering, f"{file_name.split('.')[0]}_bot.txt")

    # Open the output files
    # with open(output_file_top_path, "w") as output_file_top, open(output_file_bot_path, "w") as output_file_bot:
    output_file_top = open(output_file_top_path, "w")
    output_file_bot = open(output_file_bot_path, "w")
        
    for frame_id, frame_data in result_dict.items():
        output_file_top.write(f"{frame_id}\n")
        output_file_bot.write(f"{frame_id}\n")
        
        att_veh_top = []
        att_ped_top = []
        att_id_top = []
        att_veh_bot = []
        att_ped_bot = []
        att_id_bot = []
        
        for box in frame_data:        
            if box['label'] == 0:  # Vehicle
                veh_top_left = (box['left'], box['top'])
                veh_bottom_right = (box['right'], box['bottom'])
                
                result_top = is_intersection(line_segment_top1, veh_top_left, veh_bottom_right) or \
                             is_intersection(line_segment_top2, veh_top_left, veh_bottom_right)            
                result_bot = is_intersection(line_segment_bot1, veh_top_left, veh_bottom_right) or \
                             is_intersection(line_segment_bot2, veh_top_left, veh_bottom_right)
                
                if result_top:
                    att_veh_top.append(box['id'])
                if result_bot:
                    att_veh_bot.append(box['id'])
                    
            elif box['label'] == 1:  # Pedestrian
                point = [(box['left'] + box['right']) / 2, (box['top'] + box['bottom']) / 2]
                
                result_top = is_in_poly(point, poly_top)
                result_bot = is_in_poly(point, poly_bot)
                
                if result_top:
                    att_ped_top.append(box['id'])  
                if result_bot:
                    att_ped_bot.append(box['id'])
        
        # Filter IDs that are both in vehicles and pedestrians lists
        att_id_top = att_veh_top if att_ped_top else []
        att_id_bot = att_veh_bot if att_ped_bot else []
        
        # print(frame_id, att_id_top)
        
        output_file_top.write(f"{len(att_id_top)}\n")
        output_file_bot.write(f"{len(att_id_bot)}\n")
        
        # Correctly iterate the boxes to write the records of interest
        for box in frame_data:
            if box['id'] in att_id_top:
                output_file_top.write(f"{box['id']} {box['left']} {box['top']} {box['right']} {box['bottom']}\n")
            if box['id'] in att_id_bot:
                output_file_bot.write(f"{box['id']} {box['left']} {box['top']} {box['right']} {box['bottom']}\n")
