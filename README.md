# The Crosswalk Dataset

This is the dataset and code for manuscript

**Crosswalk: A Traffic Monitoring Dataset for Vehicle Non-Yielding Violations Detection and Interactive Scenarios Understanding**.

# Overview

- We introduce a novel traffic monitoring dataset, *Crosswalk*, which not only encompasses integrated pedestrian and vehicle interactions but also specifically detects instances where vehicles fail to yield to pedestrians.

- We propose a effective analytical framework to comprehend pedestrian-vehicle dynamics, facilitating the detection of specific violation incidents.

- A quick overview of this work: *Crosswalk*, *event awareness method*, *MRN*.

# Update Notification

- *Nov 18, 2024*

&nbsp;&nbsp;&nbsp;&nbsp; The current versions are **all available** for download from **Baidu Netdisk**.

&nbsp;&nbsp;&nbsp;&nbsp; Some important data is already available for download from **Google Drive**.

# Crosswalk

## Dataset Introduction

 This dataset originates from continuous 24-hour live-streams of a specific intersection in Bangkok, captured from a fixed overhead perspective on **[YouTube](https://www.youtube.com/watch?v=xbBKbDwlR0E)**.
 
 It consists of 10 hours of footage, divided into 120 untrimmed videos, recorded between November 2023 and May 2024.
 
 Video numbers range from *video_001* to *video_120*. All videos are five minutes long.
 
 These videos feature the same traffic scene under various weather conditions and times of day.
 
 Focusing on two pedestrian crosswalks, the dataset captures vehicles moving bidirectionally, entering and exiting the frame via the left and right boundaries.
 
 To reduce extraneous background activity, the recordings were confined to areas of interest, yielding a final video resolution of 1200×1100 pixels at 30 fps.

## Dataset Annotations

You can download the violation event annotations through [Baidu Disk](https://pan.baidu.com/s/1aoJLJUT-A7H4jO1Luzsp9w?pwd=6l8r) or [Google Drive](https://drive.google.com/file/d/1pKrevRdrWC7-hDcp8O-jYVJMv-YMPayW/view?usp=sharing).

We will *make the annotations fully available for download after the manuscript is accepted*. Howerver, you can still reproduce our work at this stage. You can download our preprocessed features and use them directly for downstream tasks.

We define an interaction event as the period during which a vehicle enters or exits the crosswalk area concurrent with a pedestrian's presence. As a result, we have meticulously annotated approximately 7.7k pedestrian-vehicle interaction events. The *Crosswalk* dataset comprises two categories of pedestrian-vehicle interaction events: violations and non-violations, totaling 1972 and 5752 events, respectively.

In this work, *event-level* annotations are used as they effectively reflect the instance situation.

The annotations are stored in JSON files, generated using LabelMe, where the file names correspond to the frame numbers at which the violations occur. For instance, if a vehicle enters the area of interest at frame 1301 and leaves at frame 1501, and fails to yield to pedestrians during this time, two JSON files will be created: 00001300.json and 00001500.json. 

Each JSON file contains two key data, namely *points* and *group_id*.

The bounding box coordinates are recorded under *points*, capturing the key vehicle's location as it enters and exits the crosswalk. The bounding box coordinates of the vehicle in the form of *(top left x, top left y, bottom right x, bottom right y)*.

*group_id* uniquely identifies the vehicle within the annotated area, distinguishing it from other vehicles. Note that if a vehicle violates the rules at two separate crosswalks, there will be four corresponding JSON files, each associated with two different group_ids.

Convert event annotations in json format to txt text. Run:

`python json_to_txt.py`

It is recommended to **put these files in a folder as follows**, and you can also **modify the path**.

        -annotations\
          -video_001\
            -00000291.json
            -00000377.json
            ...
          -video_002
          ...
          -video_120
        -json_to_txt.py

For a certain violation of a certain vehicle, the format of the txt file is:

        -group_id
        -How many times (all 2)
        -1 / Number of frames entering the zebra crossing area / (all 0) / Bounding box when entering the zebra crossing
        -2 / Number of frames leaving the zebra crossing area / (all 0) / Bounding box when leaving the zebra crossing

## Evaluation Criteria

To evaluate the model's generalization and robustness, we designed two distinct cross-validation benchmarks: *cross-video* and *cross-scene*.

- *Cross-video evaluation.* We sequentially numbered 120 video segments, with snippets from odd-numbered videos used for training and even-numbered for testing. This method evaluates the model's ability to generalize across diverse video sequences.

- *Cross-scene evaluation.* Under this benchmark, events from the upper zebra crossing areas were assigned to the training dataset, while events from the lower areas were allocated to testing. This approach assesses the model's capability to recognize violations across varied traffic scenarios.

## Dataset Preparation

We divide the video into frames, which is also an important pre-step for subsequent preprocessing.

`pip install video-cli`

Take video_001 as an example and divide it into frames:

`video-toimg video_001.avi`

# Event Awareness Method

## Pretrained Detector

To accurately detect objects of interest in traffic scenes, we use YOLOv8-m for pre-training on additional annotated datasets.

We recommend using the provided datasets for object detector training or utilizing the trained model parameters.

This additional [auxiliary dataset](https://app.roboflow.com/nnu-hi7if/nnu_intersection/7) is used to train the object detector.

Please install the environment according to the official website of [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/README.md).

All models were trained on an RTX 4090. You can use our **pretrained weights** or train the model yourself.

You can download the weights through [Baidu Disk](https://pan.baidu.com/s/1XhjA8IiR8zNuU3blHsymEQ?pwd=z8fl) or [Google Drive](https://drive.google.com/drive/folders/1fPra_scgzv_RP1yKMQ83HdyWiVFyeYkc?usp=sharing).

Training a detector from scratch：

`yolo task=detect mode=train model=yolov8m.pt data=intersection_images/data.yaml epochs=100 imgsz=640`

YOLOv8 has five variants: -n, -s, -m, -l, -x. The "model" parameter can be modified to specify the variant to use.

We recommend using -m as it achieves better results on auxiliary datasets.

## Prepeocessing Method

The overall steps include:

- **Tracking**

- **Filtering**

- **Localization**

To demonstrate the output at each step, we have prepared separate Python files corresponding to nearly every stage of the process. You can *follow them sequentially* or *download our preprocessed files* for quick reproduction.

# MRN

After completing the preprocessing, the main program can be called for training and testing.

You can also modify the parameters to call other models for training and testing.

You can use the model we have trained for testing.

# Acknowledgments and Contact

We would like to express our sincere gratitude to the following individuals and groups for their invaluable assistance in this work:

- The person in charge of the YouTube live broadcast platform for permitting data collection.

- The officers in Nanjing Transport for their meticulous annotation of the dataset.

- Potential contributors, including reviewers and researchers, for their interest and input in this work.

Any questions, feel free to contact me via email: `zeshenghu@njnu.edu.cn`

**In view of the double-blind review, we temporarily anonymized the contact email address and transport department.**
