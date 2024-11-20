# The Crosswalk Dataset

This is the dataset and code for manuscript

**Crosswalk: A Traffic Monitoring Dataset for Vehicle Non-Yielding Violations Detection and Interactive Scenarios Understanding**.

# Overview

- The overall process of our work is illustrated in <strong><a href="#figure1">Figure 1</a></strong>.

- We introduce a novel traffic monitoring dataset, *Crosswalk*, which not only encompasses integrated pedestrian and vehicle interactions but also specifically detects instances where vehicles fail to yield to pedestrians.

- We propose a effective analytical framework to comprehend pedestrian-vehicle dynamics, facilitating the detection of specific violation incidents.

<p align="center">
  <object data="demo/framework.pdf" type="application/pdf" width="1000" height="600">
    <p>It appears you don't have a PDF plugin for this browser. No problem... you can <a href="demo/framework.pdf">click here to download the PDF file.</a></p>
  </object>
</p>
<p align="center">
  <em><strong>Figure 1. Our framework.</strong></em>
</p>

# Update Notification

- *Nov 18, 2024*

&nbsp;&nbsp;&nbsp;&nbsp; The current versions are **all available** for download from **Baidu Netdisk**.

&nbsp;&nbsp;&nbsp;&nbsp; Some important data is already available for download from **Google Drive**.

# Crosswalk

## Dataset Introduction

 This dataset originates from continuous 24-hour live-streams of a specific intersection in Bangkok, captured from a fixed overhead perspective on **[YouTube](https://www.youtube.com/watch?v=xbBKbDwlR0E)**.
 
 It consists of 10 hours of footage, divided into **120** untrimmed videos, recorded between November 2023 and May 2024.
 
 Video numbers range from **video_001.avi** to **video_120.avi**. All videos are **five minutes** long.
 
 These videos feature the same traffic scene under various weather conditions and times of day.
 
 Focusing on **two** crosswalks, the dataset captures vehicles moving bidirectionally, entering and exiting the frame via the left and right boundaries.
 
 To reduce extraneous background activity, the recordings were confined to areas of interest, yielding a final video resolution of 1200×1100 pixels at 30 fps.

## Dataset Annotations

You can download the violation event annotations through [Baidu Disk](https://pan.baidu.com/s/1aoJLJUT-A7H4jO1Luzsp9w?pwd=6l8r) or [Google Drive](https://drive.google.com/file/d/1pKrevRdrWC7-hDcp8O-jYVJMv-YMPayW/view?usp=sharing).

We define **an interaction event** as **the period during which a vehicle enters or exits the crosswalk area concurrent with a pedestrian's presence**. As a result, we have meticulously annotated approximately **7.7k** pedestrian-vehicle interaction events. The *Crosswalk* dataset comprises two categories of pedestrian-vehicle interaction events: violations and non-violations, totaling 1972 and 5752 events, respectively.

In this work, **event-level** annotations are used as they effectively reflect the instance situation.

The annotations are stored in JSON files, generated using **LabelMe**, where the file names correspond to the frame numbers at which the violations occur. For instance, if a vehicle enters the area of interest at frame **1301** and leaves at frame **1501**, and fails to yield to pedestrians during this time, two JSON files will be created: **00001300.json** and **00001500.json**. 

Each JSON file contains two key data, namely **points** and **group_id**.

The bounding box coordinates are recorded under **points**, capturing the key vehicle's location as it enters and exits the crosswalk. The bounding box coordinates of the vehicle in the form of **(top left x, top left y, bottom right x, bottom right y)**.

**group_id** uniquely identifies the vehicle within the annotated area, distinguishing it from other vehicles. Note that if a vehicle violates the rules at two separate crosswalks, there will be four corresponding JSON files, each associated with two different group_ids.

*Convert event annotations in json format to txt text. Run:*

`python json_to_txt.py`

It is recommended to put these files in a folder as follows, and you can also **modify the path**.

        -annotations\
          -video_001\
            -00000291.json
            -00000377.json
            ...
          -video_002
          ...
          -video_120
        -json_to_txt.py

For a certain violation of a certain vehicle 𝒱, the format of the txt file is:

        -group_id
        -2
        -1 / The frame number in which 𝒱 enters the crosswalk / 0 / The bounding box when 𝒱 enters the crosswalk
        -2 / The frame number in which 𝒱 leaves the crosswalk / 0 / The bounding box when 𝒱 leaves the crosswalk

## Evaluation Criteria

To evaluate the model's generalization and robustness, we designed two distinct cross-validation benchmarks: **cross-video** and **cross-scene**.

- **Cross-video evaluation.** We sequentially numbered 120 video segments, with snippets from odd-numbered videos used for training and even-numbered for testing. This method evaluates the model's ability to generalize across diverse video sequences.

- **Cross-scene evaluation.** Under this benchmark, events from the upper crosswalk area were assigned to the training dataset, while events from the lower area were allocated to testing. This approach assesses the model's capability to recognize violations across varied traffic scenarios.

## Dataset Preparation

We divide the video into frames, which is also an important pre-step for subsequent preprocessing.

*Take video_001.avi as an example and divide it into frames:*

`pip install video-cli`

`video-toimg video_001.avi`

# Event Awareness Method

## Pretrained Detector

To accurately detect objects of interest in traffic scenes, we use **YOLOv8** for pre-training on additional annotated datasets.

We recommend using the provided [dataset](https://app.roboflow.com/nnu-hi7if/nnu_intersection/7) for object detector training or utilizing the trained model parameters.

Please install the environment according to the official website of [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/README.md).

All models were trained on an RTX 4090. You can use our **pretrained weights** or train the model yourself.

You can download the weights through [Baidu Disk](https://pan.baidu.com/s/1XhjA8IiR8zNuU3blHsymEQ?pwd=z8fl) or [Google Drive](https://drive.google.com/drive/folders/1fPra_scgzv_RP1yKMQ83HdyWiVFyeYkc?usp=sharing).

*Training a detector from scratch：*

`yolo task=detect mode=train model=yolov8m.pt data=intersection_images/data.yaml epochs=100 imgsz=640`

YOLOv8 has five variants: -n, -s, -m, -l, -x. The "model" parameter can be modified to specify the variant to use.

We recommend using **-m** as it achieves better results on the auxiliary dataset.

## Prepeocessing Method

The overall steps include:

- **Tracking**

- **Filtering**

- **Localization**

To demonstrate the output at each step, we have prepared separate Python files corresponding to nearly every stage of the process. You can *follow them sequentially* or *download our preprocessed files* for quick reproduction.

# MRN

## Train and Test

After completing the preprocessing, the main program can be called for training and testing.

It is recommended to **put these files in a folder as follows**, and you can also **modify the path**.

        -main.py
        -dataset_reader_mrn.py      # Load the preprocessed data.
        -model_mrn.py               # Call the model and give it arguments
        -model                      # Modelzoo
        -label                      # Label file

        -tra_att_volumes_region     # RR
        -rgb_volumes_region         # VR
        ...

*When training an MRN from scratch, run:*

`python main.py`



You can use the model we have trained for testing.

You can also modify the parameters to call other models for training and testing.

## EIoU and AP

For each event, we get two scores (violation and non-violation) during the test phase.

For MRN, we take the category corresponding to the maximum score.

For other models, we adopt the strategy of late fusion, and the scores of RR and VR are respectively input into the network for weighted fusion.

On this basis, we calculate **AP** based on the **EIoU** of each detected event and the ground truth.

In this work, **the threshold of EIoU** is set to **0.5**. 

That is, for each real event, **the EIoU of the detected event must be greater than the threshold** and **the predicted category must be correct** to be judged as **a successful detection**.

Each annotated event can **only be retrieved once**.

Therefore, for the predicted score, calculating AP requires **two** steps.

*The **first** step is to calculate the EIoU between the detected event and the ground truth:*

`python eiou_cal.py`

*The **second** step is to calculate AP based on the EIoU and whether the predicted category is accurate:*

`python ap_cal.py`

# Acknowledgments and Contact

We would like to express our sincere gratitude to the following individuals and groups for their invaluable assistance in this work:

- The person in charge of the YouTube live broadcast platform for permitting data collection.

- The officers in Nanjing Transport for their meticulous annotation of the dataset.

- Potential contributors, including reviewers and researchers, for their interest and input in this work.

Any questions, feel free to contact me via email: `zeshenghu@njnu.edu.cn`

**In view of the double-blind review, we temporarily anonymized the contact email address and transport department.**
