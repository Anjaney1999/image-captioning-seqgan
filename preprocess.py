import json
from collections import Counter
import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import encode_captions
from datasets import ImageDataset
from models import Encoder

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data(dataset,
                 karpathy_split_path,
                 min_freq, 
                 dest_path,
                 max_caps_per_img
):
    with open(karpathy_split_path + '/dataset_{}.json'.format(dataset), 'r') as f:
        karpathy_split = json.load(f)
    
    caps_train = []
    img_names_train = []

    caps_per_img = {}
    
    caps_val = []
    img_names_val = []
    
    word_freq = Counter()

    max_len = 0
    for img in karpathy_split['images']:
        captions_count = 0
        img_name = img['filename']
        if dataset == 'coco':
            img_name = img['filepath'] + '/' + img['filename']
            
        for cap in img['sentences']:
            word_freq.update(cap['tokens'])
            if captions_count < max_caps_per_img:
                captions_count += 1
            else:
                break

            if img['split'] == 'train':
                caps_train.append(cap['tokens'])
                img_names_train.append(img_name)
            elif img['split'] == 'val':
                caps_val.append(cap['tokens'])
                img_names_val.append(img_name)

                if img_name not in caps_per_img:
                    caps_per_img[img_name] = [cap['tokens']]
                else:
                    caps_per_img[img_name].append(cap['tokens'])

            max_len = max(max_len, len(cap['tokens']))

    words = [word for word in word_freq.keys() if word_freq[word] > min_freq]
    word_index = {word: idx + 4 for idx, word in enumerate(words)}
    word_index['<pad>'] = 0
    word_index['<start>'] = 1
    word_index['<end>'] = 2
    word_index['<unk>'] = 3
    
    index_word = {v: k for k, v in word_index.items()}
    
    with open(os.path.join(dest_path, dataset, 'word_index.json'), 'w') as f:
        json.dump(word_index, f)

    with open(os.path.join(dest_path, dataset, 'index_word.json'), 'w') as f:
        json.dump(index_word, f)
    
    enc_caps_train, cap_lens_train = encode_captions(caps_train, word_index, max_len)
    
    enc_caps_val, cap_lens_val = encode_captions(caps_val, word_index, max_len)

    for img_name in caps_per_img:
        caps_per_img[img_name], _ = encode_captions(caps_per_img[img_name], word_index, max_len)

    with open(os.path.join(dest_path, dataset, 'train', 'captions.json'), 'w') as f:
        json.dump(enc_caps_train, f)
        
    with open(os.path.join(dest_path, dataset, 'train', 'captions_len.json'), 'w') as f:
        json.dump(cap_lens_train, f)
        
    with open(os.path.join(dest_path, dataset, 'val', 'captions.json'), 'w') as f:
        json.dump(enc_caps_val, f)

    with open(os.path.join(dest_path, dataset, 'val', 'captions_len.json'), 'w') as f:
        json.dump(cap_lens_val, f)
    
    with open(os.path.join(dest_path, dataset, 'train', 'img_names.json'), 'w') as f:
        json.dump(img_names_train, f)

    with open(os.path.join(dest_path, dataset, 'val', 'img_names.json'), 'w') as f:
        json.dump(img_names_val, f) 
    
    with open(os.path.join(dest_path, dataset, 'val', 'caps_per_img.json'), 'w') as f:
        json.dump(caps_per_img, f)


def main(args):

    process_data(args.dataset,
                 args.karpathy_split_path,
                 args.min_freq,
                 args.dest_path,
                 args.max_caps_per_img)

    if args.extract_image_features:

        encoder = Encoder(args.cnn_architecture)
        encoder.to(device)

        train_img_feats_path = args.storage + '/processed_data/' + args.dataset + \
                               '/train/image_features/' + args.cnn_architecture

        val_img_feats_path = args.storage + '/processed_data/' + args.dataset + \
                            '/val/image_features/' + args.cnn_architecture

        if len(os.listdir(train_img_feats_path)) == 0:

            train_loader = DataLoader(
                ImageDataset(split_type='train',
                             dataset=args.dataset,
                             transform=data_transforms,
                             img_src_path=args.storage + '/' + args.image_path,
                             processed_data_path=args.storage + '/processed_data'),
                batch_size=args.batch_size, num_workers=1)

            for imgs, img_names in tqdm(train_loader):

                imgs = imgs.to(device)

                img_feats = encoder(imgs)

                for feats, name in zip(img_feats, img_names):
                    np.save(args.storage + '/processed_data/' + args.dataset + '/train/image_features/' +
                            args.cnn_architecture + '/' + name.split('/')[-1] + '.npy', feats.cpu().numpy())

        if len(os.listdir(val_img_feats_path)) == 0:

            val_loader = DataLoader(
                ImageDataset(split_type='val',
                             dataset=args.dataset,
                             transform=data_transforms,
                             img_src_path=args.storage + '/images',
                             processed_data_path=args.storage + '/processed_data'),
                batch_size=args.batch_size, num_workers=1)

            for imgs, img_names in tqdm(val_loader):

                imgs = imgs.to(device)

                img_feats = encoder(imgs)

                for feats, name in zip(img_feats, img_names):
                    np.save(args.storage + '/processed_data/' + args.dataset + '/val/image_features/' +
                            args.cnn_architecture + '/' + name.split('/')[-1] + '.npy', feats.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-process data')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--karpathy-split-path', type=str, default='karpathy_splits')
    parser.add_argument('--min-freq', type=int, default=5)
    parser.add_argument('--dest-path', type=str, default='processed_data')
    parser.add_argument('--max-allowed-len', type=int, default=25)
    parser.add_argument('--image-path', type=str, default='images')
    parser.add_argument('--extract-image-features', type=bool, default=True)
    parser.add_argument('--max_caps-per-img', type=int, default=5)

    main(parser.parse_args())






