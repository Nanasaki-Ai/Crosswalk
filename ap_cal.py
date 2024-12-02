import argparse
import pickle
from sklearn.metrics import average_precision_score as ap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import glob
from scipy.special import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', default='cs1',
                    choices={'cs1', 'cv1'},
                    help='the work folder for storing results')
parser.add_argument('--feature', default='3',
                    choices={'1', '2', '3', '4', '5', '6'},                  
                    help='input features, ...\
                    1: vr ...\
                    2: rr ...\
                    3: mr, i.e., vr+rr (especially for MAN) ...\
                    4: rr_mask (especially for different att_mask in slowfast) ...\
                    5: rr_obj (especially for object-based att_region in slowfast) ...\
                    6: rr_entire (especially for fixed att_region in slowfast) ...\
                    ')
parser.add_argument('--backbone',
                    default='man',
                    choices={'p3d', 'densenet', 's3d', 'i3d', 
                    'slowfast', 'slowonly', 'man',
                    'res3d', 'shuv2', 'mobv2',
                    'vit', 'simplevit', 'vivit'},
                    help='visual encoder')                    
parser.add_argument('--threshold', default='0.5',
                    choices={'0.2', '0.3', '0.4', '0.5', '0.6', '0.7'},                    
                    help='if eiou > threshold, and predict right, then will be correct')
                    
arg = parser.parse_args()

benchmark = arg.benchmark
backbone = arg.backbone
feature = arg.feature
threshold = arg.threshold

if feature == '1':
    features = 'vr'
elif feature == '2':
    features = 'rr'
elif feature == '3': 
    features = 'mr'
elif feature == '4': 
    features = 'rr_mask'
elif feature == '5': 
    features = 'rr_obj'
elif feature == '6': 
    features = 'rr_entire'
    
print('backbone:', backbone)
print('features:', features)
print('benchmark:', benchmark)
    
# Build the label path and automatically read the pkl file
label_path = os.path.join('label', benchmark, 'test_labels.pkl')
with open(label_path, 'rb') as f:
    label = np.array(pickle.load(f))
print('label_path', label_path)

score_path = glob.glob(os.path.join('score_dir', benchmark, backbone, features, '*.pkl'))[0]
print('score_path', score_path)
    
with open(score_path, 'rb') as f:
    r1 = list(pickle.load(f).items())

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
eiou_path = 'eiou_values.txt'
eiou = load_eiou(eiou_path)

y_true = []
y_score_0 = []    
y_score_1 = []  

for i in tqdm(range(len(label[0]))):
    sample_name = label[0][i]
    l = int(label[1][i])
    _, r = r1[i]  

    # softmax
    probabilities = softmax(r)
    
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
