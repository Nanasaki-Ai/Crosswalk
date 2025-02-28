import argparse
import pickle
from sklearn.metrics import average_precision_score as ap
import numpy as np
from tqdm import tqdm
import os
import glob
from scipy.special import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', default='cs1',
                    choices={'cs1', 'cv1'},
                    help='the work folder for storing results')
parser.add_argument('--backbone',
                    default='mobv2',
                    choices={'res3d', 'p3d', 'densenet', 's3d',
                             'shuv2', 'mobv2', 'i3d', 'slowfast',
                             'vit', 'simplevit', 'vivit', 'i3d', 'c3d'},
                    help='visual encoder')
parser.add_argument('--threshold', default='0.5',
                    choices={'0.2', '0.3', '0.4', '0.5', '0.6', '0.7'},
                    help='if eiou > threshold, and predict right, then will be TP or TN')

arg = parser.parse_args()

benchmark = arg.benchmark
backbone = arg.backbone
threshold = arg.threshold

weight_rr = 0.8           
weight_vr = 1 - weight_rr

assert (weight_vr + weight_rr) == 1.0, "The sum of weights must be 1."

print('')
print('backbone:', backbone)
print('benchmark:', benchmark)
print('')

# Build the label path and automatically read the pkl file
label_path = os.path.join('label', benchmark, 'test_labels.pkl')
with open(label_path, 'rb') as f:
    label = np.array(pickle.load(f))
print('label_path', label_path)

# Load the scores file for VR and RR
score_path_vr = glob.glob(os.path.join('score_dir', benchmark, backbone, 'vr', '*.pkl'))[0]
print('score_path_vr', score_path_vr)
with open(score_path_vr, 'rb') as f:
    r1_vr = list(pickle.load(f).items())

score_path_rr = glob.glob(os.path.join('score_dir', benchmark, backbone, 'rr', '*.pkl'))[0]
print('score_path_rr', score_path_rr)
with open(score_path_rr, 'rb') as f:
    r1_rr = list(pickle.load(f).items())

def load_eiou(file_path):
    eiou = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                name, iou = parts
                eiou[name] = float(iou)
    return eiou

# Set the eiou threshold, which can be adjusted as needed
eiou_th = float(threshold)  

# Load eiou
eiou_path = 'eiou_values_updated.txt'
eiou = load_eiou(eiou_path)

y_true = []
y_score_0 = []    
y_score_1 = []  

for i in tqdm(range(len(label[0]))):
    sample_name = label[0][i]  
    l = int(label[1][i])       
    _, r_vr = r1_vr[i]        
    _, r_rr = r1_rr[i]    

    # Weighted fusion
    r_fused = weight_vr * np.array(r_vr) + weight_rr * np.array(r_rr)

    # Softmax normalization
    probabilities = softmax(r_fused)
    predicted_class = np.argmax(probabilities)  # Get the predicted category

    eiou_val = eiou.get(sample_name, 1.0)  

    if predicted_class == l and eiou_val > eiou_th:
        y_score_0.append(probabilities[0])
        y_score_1.append(probabilities[1])
    elif predicted_class != l:
        y_score_0.append(probabilities[0])
        y_score_1.append(probabilities[1])
    else:
        y_score_0.append(0)
        y_score_1.append(0)

    y_true.append(l)

print('y_true length', len(y_true))

keyframe_ap_V = ap(y_true,
                   y_score_0,
                   pos_label=0,
                   sample_weight=None)
                   
keyframe_ap_N = ap(y_true,
                   y_score_1,
                   pos_label=1,
                   sample_weight=None)
                   
print('AP-V: {:.1%}'.format(keyframe_ap_V))  
print('AP-N: {:.1%}'.format(keyframe_ap_N))  
mAP = (keyframe_ap_V + keyframe_ap_N) / 2
print('mAP: {:.1%}'.format(mAP))   
