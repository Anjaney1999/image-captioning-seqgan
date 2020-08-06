import argparse
import csv
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import os.path as path
import time
import sys
import subprocess
from tqdm import tqdm

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
try:
    from nltk.translate.bleu_score import corpus_bleu
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "nltk"])

from nltk.translate.bleu_score import corpus_bleu
from datasets import ImageCaptionDataset
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(torch.cuda.is_available())

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def evaluate(encoder, generator, data_loader, word_index, index_word, args):
    references = list()
    hypotheses = list()
    for i, (imgs, caps, cap_lens, matching_caps) in tqdm(enumerate(data_loader)):
        k = args.beam_size
        imgs, caps = imgs.to(device), caps.to(device)

        if not args.use_image_features:
            imgs = encoder(imgs)

        imgs = imgs.expand(k, imgs.shape[1], imgs.shape[-1])
        k_prev_words = torch.LongTensor([[word_index['<start>']]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1
        hidden = generator.init_hidden_state(imgs)
        ref_dict = {}
        hyp_dict = {}
        idx = 0
        while True:
            embeddings = generator.embedding(k_prev_words).squeeze(1)
            context, _ = generator.attention_net(imgs, hidden)
            gate = generator.sigmoid(generator.f_beta(hidden))
            context = gate * context
            hidden = generator.gru(torch.cat([embeddings, context], dim=1), hidden)
            scores = F.log_softmax(generator.fc(hidden), dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
            prev_word_inds = top_k_words // len(word_index)
            next_word_inds = top_k_words % len(word_index)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_index['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            hidden = hidden[prev_word_inds[incomplete_inds]]
            imgs = imgs[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            if step > 50:
                break
            step += 1

        max_i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[max_i]
        # ref_dict[idx] = []
        for cap_set in matching_caps.tolist():
            refs = []
            for caption in cap_set:
                cap = [word_id for word_id in caption
                       if word_id != word_index['<start>'] and word_id != word_index['<pad>']]
                refs.append(cap)
                # ref_dict[idx].append(' '.join(cap).strip())
            references.append(refs)

        hypothesis = [w for w in seq if
                      w not in {word_index['<start>'], word_index['<end>'], word_index['<pad>']}]

        hypotheses.append(hypothesis)

        # hyp_dict[idx] = [' '.join(hypothesis).strip()]
        idx += 1

    bleu4 = corpus_bleu(references, hypotheses)
    print(bleu4)


def main(args):
    with open(args.storage + '/processed_data/' + args.dataset + '/word_index.json') as f:
        word_index = json.load(f)

    with open(args.storage + '/processed_data/' + args.dataset + '/index_word.json') as f:
        index_word = json.load(f)

    vocab_size = len(word_index)

    checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.checkpoint_filename

    generator = Generator(attention_dim=args.attention_dim,
                          gru_units=args.gru_units,
                          vocab_size=vocab_size,
                          embedding_dim=args.embedding_dim)
    generator.to(device)

    if path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        generator.to(device)

    encoder = None

    if args.use_image_features:
        data_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='val',
                                use_img_feats=True, transform=None,
                                img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=1, shuffle=True, num_workers=args.workers)
    else:
        encoder = Encoder(args.cnn_architecture)
        encoder.to(device)
        data_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='val',
                                use_img_feats=False, transform=data_transforms,
                                img_src_path=args.storage + '/images',
                                cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=1, shuffle=True, num_workers=args.workers)

    evaluate(encoder, generator, data_loader, word_index, index_word, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--beam-size', type=int, default=10)
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--image-path', type=str, default='images')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--checkpoint-filename', type=str, default='pg_gen_1900_2_0_1_10_multinomial_resnet152.pth')
    parser.add_argument('--use-image-features', type=bool, default=True)
    parser.add_argument('--attention-dim', type=int, default=512)
    parser.add_argument('--gru-units', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)

    main(parser.parse_args())




