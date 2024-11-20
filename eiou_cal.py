import os
from tqdm import tqdm
from tools import *

label_file = 'label_txt'   
det_file = 'filtering_woatt_summary' 
video_file = 'intersection-video'  
trajectory_file = 'generated_images'  

# folder_trajectory = 'tracking_output' 
folder_filtering_woatt = 'filtering_woatt_output'  
folder_filtering = 'filtering_output'  

save_name_eiou_pairs = []  # List to store save_name and best_eiou pairs
    
video_num = [f for f in os.listdir(trajectory_file)]
# print(video_num)

for file_name in tqdm(video_num, desc='Processing files'):    
    file_number = file_name.split('_')[-1]  
    
    label_path = os.path.join(label_file, f"{file_name}.txt")
    top_det_path = os.path.join(det_file, f"{file_name}_woatt_top.txt")
    bot_det_path = os.path.join(det_file, f"{file_name}_woatt_bot.txt")
    
    top_path = os.path.join(folder_filtering, f"{file_name}_top.txt")
    bot_path = os.path.join(folder_filtering, f"{file_name}_bot.txt")
    top_woatt_path = os.path.join(folder_filtering_woatt, f"{file_name}_woatt_top.txt")
    bot_woatt_path = os.path.join(folder_filtering_woatt, f"{file_name}_woatt_bot.txt")
    
    # video_path = os.path.join(video_file, file_name)
    # trajectory_path = os.path.join(trajectory_file, file_name)
      
    top_dict = read_filtering_trajectory_txt(top_path)
    bot_dict = read_filtering_trajectory_txt(bot_path) 
    top_woatt_dict = read_filtering_trajectory_txt(top_woatt_path)
    bot_woatt_dict = read_filtering_trajectory_txt(bot_woatt_path)

    att_top_ids = set()  
    att_bot_ids = set()  
    
    label_dict = read_label_txt(label_path)       
    top_det_dict = read_label_txt(top_det_path)   
    bot_det_dict = read_label_txt(bot_det_path)   
        
    for box_id in top_det_dict.keys():
        att_top_ids.add(box_id) 
    # print(att_top_ids)
    for box_id in bot_det_dict.keys():
        att_bot_ids.add(box_id)
    
    matched_top_ids = set()  
    matched_bot_ids = set()
    
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
        
        det_dict = top_det_dict if region_top else bot_det_dict
        matched_ids = matched_top_ids if region_top else matched_bot_ids
        not_matched_det_ids = att_top_ids if region_top else att_bot_ids
            
        for box_id, det_boxes in det_dict.items():
            if box_id in matched_ids:
                continue
            
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
            matched_ids.add(best_match_id)  
            not_matched_det_ids.discard(best_match_id)
            
        save_name = f'V{file_number}I{best_match_id:05d}S{0 if region_top else 1}D{0 if downward else 1}R0A0'

        save_name_eiou_pairs.append((save_name, best_eiou/2))        

# Save the save_name and best_eiou pairs to a txt file
output_file = 'eiou_values.txt'
with open(output_file, 'w') as f:
    for save_name, eiou in save_name_eiou_pairs:
        f.write(f"{save_name} {eiou:.2f}\n")  # Write each save_name and eiou pair

print(f'Total number of pairs: {len(save_name_eiou_pairs)}')