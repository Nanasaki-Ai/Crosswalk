import os
import json
from tqdm import tqdm

# Specify the root directory containing the folders
root_folder_path = "annotations"

# Specify the root directory for storing results
output_root_path = "label_txt"

# Create the output directory if it doesn't exist
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

# Iterate through all subfolders in the root directory
subfolder_paths = [os.path.join(root_folder_path, d) for d in os.listdir(root_folder_path) if os.path.isdir(os.path.join(root_folder_path, d))]
for subfolder_path in tqdm(subfolder_paths, desc="Processing label files"):
    # Get the subfolder name for naming the output file
    folder_name = os.path.basename(subfolder_path)
    output_file_path = os.path.join(output_root_path, f"{folder_name}.txt")

    # Get all JSON files in the current subfolder
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith(".json")]

    # Dictionary to store results
    result_dict = {}

    # Iterate through each JSON file
    for json_file in json_files:
        json_path = os.path.join(subfolder_path, json_file)

        # Read the content of the JSON file
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # Get group_id and label information
        group_id = None
        label = None
        frame_number = int(json_data["imagePath"].split(".")[0])  # Extract frame number from image path
        for shape in json_data["shapes"]:
            if "group_id" in shape and shape["group_id"] is not null:
                group_id = shape["group_id"]
                label = shape["label"]
                break

        if group_id is not null and label is not null:
            # Add to the result dictionary
            if group_id not in result_dict:
                result_dict[group_id] = {"count": 0, "occurrences": []}
            result_dict[group_id]["count"] += 1
            result_dict[group_id]["occurrences"].append((frame_number, label, shape["points"]))

    # Write the results to a txt file
    with open(output_file_path, 'w') as output_file:
        for group_id, data in result_dict.items():
            output_file.write(f"{group_id}\n")
            output_file.write(f"{data['count']}\n")
            for i, occurrence in enumerate(data['occurrences'], start=1):
                if i > 2: break  # Ensure only the first two occurrences are written
                frame_number, label, points = occurrence
                output_file.write(f"{i} {frame_number+1} {label} {points[0][0]} {points[0][1]} {points[1][0]} {points[1][1]}\n")

