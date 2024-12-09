import os
import torch
import pickle
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DataSetReader(Dataset):
    def __init__(self, args, mode='train'):
        self.exp_dir = args.exp_dir
        if mode == 'train':
            self.label_path = os.path.join(args.label_path, args.benchmark, 'train_labels.pkl')
        else:
            self.label_path = os.path.join(args.label_path, args.benchmark, 'test_labels.pkl')
            
        self.feature_path = args.feature_path
        self.modality = args.modality
        if self.modality == 'vr':
            self.slow_path = os.path.join(self.feature_path, 'rgb_volumes_region')
            self.fast_path = os.path.join(self.feature_path, 'tra_att_volumes_region')
        elif self.modality == 'rr':
            self.slow_path = os.path.join(self.feature_path, 'tra_att_volumes_region')
            self.fast_path = os.path.join(self.feature_path, 'rgb_volumes_region')

        # You may find 'tra_att_volumes_region' and 'rr' in our code and preprocessor files,
        # they mean the same thing, i.e., refined represention.

        # You may find 'rgb_volumes_region' and 'vr' in our code and preprocessor files,
        # they mean the same thing, i.e., video represention.
            
        self.number = args.frame_number
        if self.number == 32:
            self.interval = 1
        elif self.number == 16:
            self.interval = 2
        elif self.number == 8:
            self.interval = 4
        elif self.number == 4:
            self.interval = 8
            
        self.load_data()

    def load_data(self):
        # label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        strr = 'Load {} samples from {}'.format(len(self.label), self.label_path)
        print(strr)
        with open('{}/log.txt'.format(self.exp_dir), 'a') as f:
            print(strr, file=f)


    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        label = self.label[index]        
        sample_number = self.sample_name[index]

        folder_slow = os.path.join(self.slow_path, sample_number)
        folder_fast = os.path.join(self.fast_path, sample_number)
        
        fast_images = []
        slow_images = []
        interval = self.interval
        
        for i, filename in enumerate(sorted(os.listdir(folder_slow))):
            if i % interval == 0: 
                if self.modality == 'rr':
                    img = cv2.imread(os.path.join(folder_slow, filename), cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(os.path.join(folder_slow, filename))
                img = cv2.resize(img, (256, 256))
                fast_images.append(img)
                
        if self.modality == 'vr':
            slow_input = np.stack(fast_images, axis=3)
            slow_input = torch.from_numpy(slow_input).float()
            slow_input = slow_input.permute(2, 3, 0, 1).contiguous()            
        else:
            slow_input = np.stack(fast_images, axis=2)  
            slow_input = torch.from_numpy(slow_input).float()
            slow_input = slow_input.permute(2, 0, 1).contiguous()
            slow_input = slow_input.unsqueeze(0)

        for i, filename in enumerate(sorted(os.listdir(folder_fast))):
            if i % interval == 0: 
                if self.modality == 'rr':
                    img = cv2.imread(os.path.join(folder_fast, filename))
                else:
                    img = cv2.imread(os.path.join(folder_fast, filename), cv2.IMREAD_GRAYSCALE)                    
                img = cv2.resize(img, (256, 256))
                slow_images.append(img)
                
        if self.modality == 'rr':
            fast_input = np.stack(slow_images, axis=3)
            fast_input = torch.from_numpy(fast_input).float()
            fast_input = fast_input.permute(2, 3, 0, 1).contiguous()            
        else:
            fast_input = np.stack(slow_images, axis=2)  
            fast_input = torch.from_numpy(fast_input).float()
            fast_input = fast_input.permute(2, 0, 1).contiguous()
            fast_input = fast_input.unsqueeze(0)
        
        #print('slow', slow_input.shape)
        #print('fast', fast_input.shape)
        
        # return fast_input, slow_input, label
        return slow_input, fast_input, label