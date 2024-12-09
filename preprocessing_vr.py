# Preprocessing to obtain video feature representations.
# For each video:
# 1. Read label values (iterate through JSON files),
#    and determine when each offending vehicle (group_id) enters and exits.
# 2. Read the trajectory data from the filtering_woatt_output folder,
#    for the top and bottom regions, determine when each vehicle (ID) enters and exits.
#    Read the trajectory data from the filtering_output folder,
#    and determine during which time intervals each vehicle has a pedestrian in the top and bottom regions.
# 3. Match (spatial-temporal matching) using the ground truth and detection values.
# 4. Using the detection values as the baseline, construct feature images based on the matched data,
#    downsample uniformly (first take 32 frames) according to the start and end frames, and save them.
# Region cropping: Crop according to top-left, top-right, bottom-left, and bottom-right.
# tools.py contains the functions used.

import os
from tqdm import tqdm
from tools import *

# Existing folders
label_file = 'label_txt'   # Path to the label value folder, contains several txt files, from video_001.txt to video_120.txt
det_file = 'filtering_woatt_summary' # Simplified detection value folder, used for matching

video_file = 'intersection-video'  # Path to the video frames folder, contains several subfolders, from video_001 to video_120
trajectory_file = 'generated_images'  # Path to the trajectory video frames folder, contains several subfolders, from video_001 to video_120

# folder_trajectory = 'tracking_output'  # Path to the original trajectory folder
folder_filtering_woatt = 'filtering_woatt_output'  # Path to the preprocessed trajectory folder
# For each video, e.g., video_001, it contains video_001_woatt_bot.txt and video_001_woatt_top.txt
folder_filtering = 'filtering_output'  # Path to the preprocessed trajectory folder, contains several txt files
# For each video, e.g., video_001, it contains video_001_bot.txt and video_001_top.txt

rgb_features_output = 'vr'   # Video representation folder
# You may find 'rgb_volumes_region' and 'vr' in our code and preprocessor files,
# they mean the same thing, i.e., video represention.
if not os.path.exists(rgb_features_output):
    os.makedirs(rgb_features_output)

video_num = [f for f in os.listdir(trajectory_file)]
# print(video_num)

poly_right = [[516.0, 13.0], [546.0, 462.0], [1073.0, 1094.0], [1197.0, 1091.0], [1191.0, 4.0]]

crop_lb = (200, 380, 790, 950)
crop_rb = (490, 250, 1040, 750)
crop_lt = (20, 50, 580, 470)
crop_rt = (480, 50, 1100, 470)

# Read dictionaries and annotations
for file_name in tqdm(video_num, desc='Processing files'):    
    file_number = file_name.split('_')[-1] 
    
    # Construct the path to read the file
    label_path = os.path.join(label_file, f"{file_name}.txt")
    top_det_path = os.path.join(det_file, f"{file_name}_woatt_top.txt")
    bot_det_path = os.path.join(det_file, f"{file_name}_woatt_bot.txt")
    
    top_path = os.path.join(folder_filtering, f"{file_name}_top.txt")
    bot_path = os.path.join(folder_filtering, f"{file_name}_bot.txt")
    top_woatt_path = os.path.join(folder_filtering_woatt, f"{file_name}_woatt_top.txt")
    bot_woatt_path = os.path.join(folder_filtering_woatt, f"{file_name}_woatt_bot.txt")
    
    video_path = os.path.join(video_file, file_name)
    trajectory_path = os.path.join(trajectory_file, file_name)
    
    # Read preprocessed trajectory data    
    top_dict = read_filtering_trajectory_txt(top_path)
    bot_dict = read_filtering_trajectory_txt(bot_path) 
    top_woatt_dict = read_filtering_trajectory_txt(top_woatt_path)
    bot_woatt_dict = read_filtering_trajectory_txt(bot_woatt_path)

    att_top_ids = set()  # Use sets to automatically remove duplicates
    att_bot_ids = set()  
    
    # Read the label file to get the true value, the time and
    # spatial coordinates of each group_id entering and leaving the zebra crossing area
    # Store the results in label_dict
    
    label_dict = read_label_txt(label_path)       
    # The label file provides the spatiotemporal information (ground truth)
    # of each group_id at the initial frame and the end frame
    
    top_det_dict = read_label_txt(top_det_path)   
    # Provides the spatiotemporal information (detection value)
    # of the initial frame and the end frame of each ID in the upper half area
    bot_det_dict = read_label_txt(bot_det_path)   
    # Provides the spatiotemporal information (detection value)
    # of the initial frame and the end frame of each ID in the lower half area
        
    for box_id in top_det_dict.keys():
        att_top_ids.add(box_id) 
    # print(att_top_ids)
    for box_id in bot_det_dict.keys():
        att_bot_ids.add(box_id)
    
    matched_top_ids = set()  # Initialize the matched id set
    matched_bot_ids = set()
    
    # Perform spatiotemporal matching
    for group_id, label_boxes in label_dict.items():
        label_start, label_end = label_boxes[0]['frame'], label_boxes[1]['frame']
        label_sbox = {'top': label_boxes[0]['top'], 'left': label_boxes[0]['left'],
                     'bottom': label_boxes[0]['bottom'], 'right': label_boxes[0]['right']}
        label_ebox = {'top': label_boxes[1]['top'], 'left': label_boxes[1]['left'],
                     'bottom': label_boxes[1]['bottom'], 'right': label_boxes[1]['right']}
                     
        best_match_id = None
        best_eiou = 0

        downward = label_sbox['top'] < label_ebox['top']               

        region_top = (downward and (label_sbox['bottom'] < 380)) or \
                     (not downward and (label_ebox['top'] < 200)) or \
                     (not downward and (label_ebox['top'] < 245) and (label_ebox['bottom'] < 510))
        
        # Select the detection value dictionary (upper/lower half)
        det_dict = top_det_dict if region_top else bot_det_dict
        matched_ids = matched_top_ids if region_top else matched_bot_ids
        not_matched_det_ids = att_top_ids if region_top else att_bot_ids
     
        if region_top and downward:
            crop_rectangle = crop_rt
        elif region_top and not downward:
            crop_rectangle = crop_lt
        elif not region_top and downward:
            crop_rectangle = crop_rb
        elif not region_top and not downward:
            crop_rectangle = crop_lb
            
        for box_id, det_boxes in det_dict.items():
            if box_id in matched_ids:
                continue  # Skip matched ids
            
            det_start, det_end = det_boxes[0]['frame'], det_boxes[1]['frame']
            det_sbox = {'top': det_boxes[0]['top'], 'left': det_boxes[0]['left'],
                       'bottom': det_boxes[0]['bottom'], 'right': det_boxes[0]['right']}
            det_ebox = {'top': det_boxes[1]['top'], 'left': det_boxes[1]['left'],
                       'bottom': det_boxes[1]['bottom'], 'right': det_boxes[1]['right']}
                       
            tiou = calculate_tiou(det_start, det_end, label_start, label_end)
            siou_s = calculate_siou(det_sbox, label_sbox)
            siou_e = calculate_siou(det_ebox, label_ebox)
            eiou = tiou * (siou_s + siou_e)
            
            if eiou > best_eiou:
                best_eiou = eiou
                best_match_id = box_id
                start_frame, end_frame = det_start, det_end
        
        if best_match_id is not None:
            matched_ids.add(best_match_id)  # Add to the matched id set
            not_matched_det_ids.discard(best_match_id)
            
        save_folder = f'V{file_number}I{best_match_id:05d}S{0 if region_top else 1}D{0 if downward else 1}R0A0'
        rgb_save_path = os.path.join(rgb_features_output, save_folder)
        if os.path.exists(rgb_save_path):
            # print("File already exists, skipping...")
            continue          
        crop_images(video_path, rgb_save_path, crop_rectangle, start_frame-1, end_frame-1, frame=32)
        
    for box_id, det_boxes in top_det_dict.items():
        if box_id in att_top_ids:   # Skip matched ids
            det_start, det_end = det_boxes[0]['frame'], det_boxes[1]['frame']
            downward = det_boxes[0]['top'] < det_boxes[1]['top']   
                        
            crop_rectangle = crop_rt if downward else crop_lt
                
            save_folder = f'V{file_number}I{box_id:05d}S0D{0 if downward else 1}R0A1'
            rgb_save_path = os.path.join(rgb_features_output, save_folder)
            if os.path.exists(rgb_save_path):
                # print("File already exists, skipping...")
                continue
            crop_images(video_path, rgb_save_path, crop_rectangle, det_start-1, det_end-1, frame=32)

    for box_id, det_boxes in bot_det_dict.items():
        if box_id in att_bot_ids:   # Skip matched ids
            det_start, det_end = det_boxes[0]['frame'], det_boxes[1]['frame']
            downward = det_boxes[0]['top'] < det_boxes[1]['top']  
            
            crop_rectangle = crop_rb if downward else crop_lb   
            
            save_folder = f'V{file_number}I{box_id:05d}S1D{0 if downward else 1}R0A1'
            rgb_save_path = os.path.join(rgb_features_output, save_folder)
            if os.path.exists(rgb_save_path):
                # print("File already exists, skipping...")
                continue                
            crop_images(video_path, rgb_save_path, crop_rectangle, det_start-1, det_end-1, frame=32)    
   
