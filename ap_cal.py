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
parser.add_argument('--benchmark', default='cv1',
                    choices={'cs1', 'cv1'},
                    help='the work folder for storing results')
parser.add_argument('--feature', default='3',
                    choices={'1', '2', '3'},                  
                    help='input features, ...\
                    1: vr ...\
                    2: rr ...\
                    3: mr, i.e., vr+rr (especially for mrn)')
parser.add_argument('--backbone',
                    default='mrn',
                    choices={'p3d', 'densenet', 's3d', 'i3d', 
                    'slowfast', 'slowonly', 'mrn',  
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

print('backbone:', backbone)
print('features:', features)
print('benchmark:', benchmark)
    
# 构建label的路径并自动读取pkl文件
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

# 设定iou阈值
eiou_th = float(threshold)  # 可以根据需要进行调整

# 加载iou
eiou_path = 'eiou_values.txt'
eiou = load_eiou(eiou_path)

y_true = []
y_score = []    
confidence = []

### 有softmax

for i in tqdm(range(len(label[0]))):
    sample_name = label[0][i]
    l = int(label[1][i])
    _, r = r1[i]  # `r` 是未经softmax处理的分数

    # 应用softmax
    probabilities = softmax(r)
    
    predicted_class = np.argmax(probabilities)  # 获取预测类别
    predicted_prob = np.max(probabilities)      # 获取最大概率作为置信度

    eiou_val = eiou.get(sample_name, 1.0)

    if predicted_class == 0 and eiou_val > eiou_th:
        y_score.append(predicted_prob)
    else:
        y_score.append(1 - predicted_prob)

    y_true.append(l)

print('y_true length', len(y_true))
print('y_score length', len(y_score))

keyframe_ap_P = ap(y_true,
                   y_score,
                   pos_label=0,
                   sample_weight=None)
                   
keyframe_ap_N = ap(y_true,
                   y_score,
                   pos_label=1,
                   sample_weight=None)
                   
print('AP-P: {:.1%}'.format(keyframe_ap_P))  
print('AP-N: {:.1%}'.format(keyframe_ap_N))  
mAP = (keyframe_ap_P + keyframe_ap_N) / 2
print('mAP: {:.1%}'.format(mAP))            
