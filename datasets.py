import json
import torch
from torch.utils.data import Dataset
import numpy as np

from utils import pil_loader

class ImageCaptionDataset(Dataset):

    def __init__(self, dataset,
                 split_type,
                 use_img_feats,
                 transform,
                 img_src_path,
                 processed_data_path,
                 cnn_architecture):

        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.use_img_feats = use_img_feats
        self.transform = transform
        self.img_src_path = img_src_path
        self.processed_data_path = processed_data_path
        self.dataset = dataset
        self.cnn_architecture = cnn_architecture
        
        with open(processed_data_path + '/' + dataset + '/' + split_type + '/img_names.json') as f:
            self.img_names = json.load(f)
        
        with open(processed_data_path + '/' + dataset + '/' + split_type + '/captions.json') as f:
            self.caps = json.load(f)

        with open(processed_data_path + '/' + dataset + '/' + split_type + '/captions_len.json') as f:
            self.cap_lens = json.load(f)

        if split_type == 'val':
            with open(processed_data_path + '/' + dataset + '/' + split_type + '/caps_per_img.json') as f:
                self.caps_per_img = json.load(f)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        
        if self.use_img_feats:
            img_feats = np.load(self.processed_data_path + '/' +
                                self.dataset + '/' + self.split_type +
                                '/image_features/' + self.cnn_architecture + '/' +
                                img_name.split('/')[-1] + '.npy')
            
            img_feats = torch.from_numpy(img_feats)

            cap = torch.LongTensor(self.caps[index])

            cap_len = torch.LongTensor([self.cap_lens[index]])

            if self.split_type == 'train':  
                return img_feats, cap, cap_len
            else:
                # matching_idxs = [idx for idx, path in enumerate(self.img_names) if path == img_name]
                # all_captions = [self.caps[idx] for idx in matching_idxs]
                all_caps = self.caps_per_img[img_name]
                return img_feats, cap, cap_len, torch.LongTensor(all_caps)
        else:
            img = pil_loader(self.img_src_path + '/' + self.dataset + '/' + img_name)
            img = self.transform(img)
            
            img = torch.FloatTensor(img)
            cap = torch.LongTensor(self.caps[index])
            cap_len = torch.LongTensor([self.cap_lens[index]])

            if self.split_type == 'train':  
                return img, cap, cap_len 
            else:
                all_caps = self.caps_per_img[img_name]
                return img, cap, cap_len, torch.LongTensor(all_caps)
    
    def __len__(self):
        return len(self.caps)
    

class ImageDataset(Dataset):
    def __init__(self, split_type, dataset, transform, img_src_path, processed_data_path):
        super(ImageDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform
        self.processed_data_path = processed_data_path
        self.img_src_path = img_src_path
        self.dataset = dataset

        with open(processed_data_path + '/' + dataset + '/' + split_type + '/img_names.json') as f:
            self.img_names = json.load(f)

        self.img_names = sorted(set(self.img_names))
        
    def __getitem__(self, index):
        img_name = self.img_names[index]
        
        img = pil_loader(self.img_src_path + '/' + self.dataset + '/' + img_name)
        img = self.transform(img)
        img = torch.FloatTensor(img)
        
        return img, img_name
    
    def __len__(self):
        return len(self.img_names)
