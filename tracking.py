# -*- coding: utf-8 -*-
import os
from collections import defaultdict
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Function to process a single video file
def process_video(video_path, output_folder):
    model = YOLO('E:/parking_test/v8/runs/detect/train14/weights/best.pt')
    cap = cv2.VideoCapture(video_path)
    
    # Extract video name without extension and create a corresponding text file path
    video_name = os.path.basename(video_path)
    video_name_without_extension = os.path.splitext(video_name)[0]
    text_file_path = os.path.join(output_folder, f"{video_name_without_extension}.txt")
    
    with open(text_file_path, "w") as new_file:
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame_number += 1
                results = model.track(frame, persist=True)
                
                new_file.write(f"{frame_number}\n")
                # If results[0].boxes is None, skip this frame
                if results[0] is None or results[0].boxes is None:
                    new_file.write("0\n")                
                    continue
                
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()
                
                new_boxes = [box for box, cls in zip(boxes, classes) if cls not in {2, 3}]
                new_file.write(f"{len(new_boxes)}\n")

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    if cls in {2, 3}:
                        continue
                    x, y, x1, y1 = box
                    class_id = 0 if cls in [0, 1, 5, 6] else 1
                    new_file.write(f"{track_id} {class_id} {x} {y} {x1} {y1}\n")
            else:
                break
        
    cap.release()

# Directory containing videos
video_directory = "E:/Bandicam/intersection-video"
# Directory to save the text files
output_directory = "E:/Bandicam/tracking_output"

# Create the output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# List all the .avi files in the video directory
video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.avi')]

# Use tqdm to show the progress
for video_path in tqdm(video_files, desc="Processing videos"):
    video_name = os.path.basename(video_path)
    video_name_without_extension = os.path.splitext(video_name)[0]
    output_text_file = os.path.join(output_directory, f"{video_name_without_extension}.txt")
    
    # Check if the text file already exists, if so, skip processing this video
    if not os.path.exists(output_text_file):
        process_video(video_path, output_directory)
    else:
        tqdm.write(f"Skipping {video_name}, output already exists.")
