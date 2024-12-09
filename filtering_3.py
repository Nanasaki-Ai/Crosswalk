import os
from tqdm import tqdm
from tools import *

def process_files(folder_ori, folder_out):
    for filename in tqdm(os.listdir(folder_ori), desc="Processing files"):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_ori, filename)
            data_dict = read_filtering_trajectory_txt(file_path)  # Assuming this function exists

            # Record the first and last appearance of each ID
            first_last_appearance = {}

            for frame_id, boxes in data_dict.items():
                for box in boxes:
                    box_id = box['id']
                    if box_id not in first_last_appearance:
                        first_last_appearance[box_id] = {'first': (frame_id, box), 'last': None}
                    else:
                        # Update the information of the last appearance
                        first_last_appearance[box_id]['last'] = (frame_id, box)

            # Write the results to the output folder
            new_file_path = os.path.join(folder_out, filename)
            with open(new_file_path, 'w') as new_file:
                for box_id, appearance in first_last_appearance.items():
                    first_frame, _ = appearance['first']
                    last_frame, _ = appearance['last'] if appearance['last'] else appearance['first']
                    
                    # Check if the frame IDs of the first and last appearance are the same (discard samples with less than 32 frames)
                    # if first_frame != last_frame:
                    if (last_frame - first_frame) >= 32:
                        new_file.write(f"{box_id}\n")
                        new_file.write("2\n")
                        new_file.write(f"1 {first_frame} 0 {appearance['first'][1]['left']} {appearance['first'][1]['top']} {appearance['first'][1]['right']} {appearance['first'][1]['bottom']}\n")
                        new_file.write(f"2 {last_frame} 0 {appearance['last'][1]['left']} {appearance['last'][1]['top']} {appearance['last'][1]['right']} {appearance['last'][1]['bottom']}\n")

# Assume the initial and output folders are as follows
folder_ori = "filtering_woatt_output"
folder_out = "filtering_woatt_summary"
os.makedirs(folder_out, exist_ok=True)  
process_files(folder_ori, folder_out)

