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

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(torch.cuda.is_available())

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def main(args):
    with open(args.storage + '/processed_data/' + args.dataset + '/word_index.json') as f:
        word_index = json.load(f)

    vocab_size = len(word_index)

    checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.checkpoint_filename

    generator = Generator(attention_dim=args.attention_dim,
                          gru_units=args.gru_units,
                          vocab_size=vocab_size,
                          embedding_dim=args.embedding_dim)
    generator.to(device)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=word_index['<pad>'], reduction='sum').to(device)

    if path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        generator.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    encoder = None

    if args.use_image_features:
        train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='train',
                                use_img_feats=True, transform=None,
                                img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='val',
                                use_img_feats=True, transform=None,
                                img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    else:
        encoder = Encoder(args.cnn_architecture)
        encoder.to(device)

        train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='train',
                                use_img_feats=False, transform=data_transforms,
                                img_src_path=args.storage + '/images',
                                cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=args.batch_size, shuffle=True, num_workers=0)

        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='val',
                                use_img_feats=False, transform=data_transforms,
                                img_src_path=args.storage + '/images',
                                cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=args.batch_size, shuffle=True, num_workers=0)

    for e in range(args.epochs):
        gen_mle_train(epoch=e, encoder=encoder, generator=generator,
                      optimizer=optimizer, criterion=criterion,
                      train_loader=train_loader, args=args)

        validate(epoch=e, encoder=encoder, generator=generator,
                 criterion=criterion, val_loader=val_loader,
                 word_index=word_index, args=args)

        if args.save_model:
            torch.save({
                'gen_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, args.storage + '/ckpts/' + args.dataset + '/gen/{}_{}_{}.pth'.format('MLE_GEN',
                                                                                    args.cnn_architecture, e))
        scheduler.step()


def gen_mle_train(epoch, encoder, generator, optimizer, criterion, train_loader, args):
    losses = AverageMeter()
    top5 = AverageMeter()
    top1 = AverageMeter()

    if not args.use_image_features:
        encoder.eval()

    generator.train()

    for batch_id, (imgs, caps, cap_lens) in enumerate(train_loader):
        start_time = time.time()

        imgs, caps = imgs.to(device), caps.to(device)

        cap_lens = cap_lens.squeeze(-1)

        if not args.use_image_features:
            imgs = encoder(imgs)

        optimizer.zero_grad()
        preds, caps, output_lens, alphas, indices = generator(imgs, caps, cap_lens)
        loss = 0.0
        for i in range(caps.shape[0]):
            loss += criterion(preds[i, :], caps[i, 1:])
        loss = loss / (1.0 * caps.shape[0])
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        preds = pack_padded_sequence(preds, output_lens, batch_first=True)[0]
        targets = pack_padded_sequence(caps[:, 1:], output_lens, batch_first=True)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip)
        optimizer.step()
        top1_acc = categorical_accuracy(preds, targets, 1)
        top1.update(top1_acc, sum(output_lens))
        top5_acc = categorical_accuracy(preds, targets, 5)
        top5.update(top5_acc, sum(output_lens))
        losses.update(loss.item(), sum(output_lens))

        if batch_id % args.print_freq == 0:
            logging.info('Epoch: [{}]\t'
                         'Batch: [{}]\t'
                         'Time per batch: [{:.3f}]\t'
                         'Loss [{:.4f}]({:.3f})\t'
                         'Top 5 accuracy [{:.4f}]({:.3f})\t'
                         'Top 1 accuracy [{:.4f}]({:.3f})\t'.format(epoch, batch_id, time.time() - start_time,
                                                                    losses.avg, losses.val, top5.avg, top5.val,
                                                                    top1.avg, top1.val))

            if args.save_stats:
                with open(args.storage + '/stats/' + args.dataset + '/gen/TRAIN_MLE_GEN.csv', 'a+') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, batch_id, losses.avg, losses.val, top5.avg, top5.val, top1.avg, top1.val])


def validate(epoch, encoder, generator, criterion, val_loader, word_index, args):
    losses = AverageMeter()
    top5 = AverageMeter()
    top1 = AverageMeter()

    if not args.use_image_features:
        encoder.eval()

    generator.eval()

    references = []
    hypotheses = []
    hypotheses_tf = []

    with torch.no_grad():

        for batch_id, (imgs, caps, cap_lens, matching_caps) in enumerate(val_loader):

            imgs, caps, cap_lens = imgs.to(device), caps.to(device), cap_lens.to(device)

            cap_lens = cap_lens.squeeze(-1)

            if not args.use_image_features:
                imgs = encoder(imgs)
            preds, caps, output_lens, alphas, indices = generator(imgs, caps, cap_lens)
            loss = 0.0
            for i in range(caps.shape[0]):
                loss += criterion(preds[i, :], caps[i, 1:])
            loss = loss / (1.0 * caps.shape[0])
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            preds_clone = preds.clone()
            preds = pack_padded_sequence(preds, output_lens, batch_first=True)[0]
            targets = pack_padded_sequence(caps[:, 1:], output_lens, batch_first=True)[0]
            top1_acc = categorical_accuracy(preds, targets, 1)
            top1.update(top1_acc, sum(output_lens))
            top5_acc = categorical_accuracy(preds, targets, 5)
            top5.update(top5_acc, sum(output_lens))
            losses.update(loss.item(), sum(output_lens))

            matching_caps = matching_caps[indices]
            for cap_set in matching_caps.tolist():
                refs = []
                for caption in cap_set:
                    cap = [word_id for word_id in caption
                           if word_id != word_index['<start>'] and word_id != word_index['<pad>']]
                    refs.append(cap)
                references.append(refs)

            fake_caps, _ = generator.sample(cap_len=max(max(cap_lens), args.max_len),
                                            col_shape=caps.shape[1], img_feats=imgs[indices],
                                            input_word=caps[:, 0], sampling_method='max')
            word_idxs, _ = pad_generated_captions(fake_caps.cpu().numpy(), word_index)
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs if idx != word_index['<start>'] and idx != word_index['<pad>']])

            word_idxs = torch.max(preds_clone, dim=2)[1]
            word_idxs, _ = pad_generated_captions(word_idxs.cpu().numpy(), word_index)
            for idxs in word_idxs.tolist():
                hypotheses_tf.append(
                    [idx for idx in idxs if idx != word_index['<start>'] and idx != word_index['<pad>']])

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        bleu_1_tf = corpus_bleu(references, hypotheses_tf, weights=(1, 0, 0, 0))
        bleu_2_tf = corpus_bleu(references, hypotheses_tf, weights=(0.5, 0.5, 0, 0))
        bleu_3_tf = corpus_bleu(references, hypotheses_tf, weights=(0.33, 0.33, 0.33, 0))
        bleu_4_tf = corpus_bleu(references, hypotheses_tf)

        logging.info('VALIDATION\n')
        logging.info('Epoch: [{}]\t'
                     'Loss [{:.4f}]\t'
                     'Top 5 accuracy [{:.4f}]\t'
                     'Top 1 accuracy [{:.4f}]\n'
                     'bleu-1 [{:.3f}]\t'
                     'bleu-2 [{:.3f}]\t'
                     'bleu-3 [{:.3f}]\t'
                     'bleu-4 [{:.3f}]\n'
                     'TF bleu-1 [{:.3f}]\t'
                     'TF bleu-2 [{:.3f}]\t'
                     'TF bleu-3 [{:.3f}]\t'
                     'TF bleu-4 [{:.3f}]\t'.format(epoch, losses.avg, top5.avg, top1.avg, bleu_1, bleu_2,
                                                   bleu_3, bleu_4, bleu_1_tf, bleu_2_tf, bleu_3_tf, bleu_4_tf))

        if args.save_stats:
            with open(args.storage + '/stats/' + args.dataset + '/gen/VAL_MLE_GEN.csv', 'a+') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, losses.avg, losses.val, top5.avg, top5.val, top1.avg, top1.val,
                                 bleu_1, bleu_2, bleu_3, bleu_4, bleu_1_tf, bleu_2_tf, bleu_3_tf, bleu_4_tf])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Maximum Likelihood Estimation Training')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--alpha-c', type=float, default=1.)
    parser.add_argument('--step-size', type=float, default=5)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--image-path', type=str, default='images')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--checkpoint-filename', type=str, default='')
    parser.add_argument('--use-image-features', type=bool, default=True)
    parser.add_argument('--attention-dim', type=int, default=512)
    parser.add_argument('--gru-units', type=int, default=512)
    parser.add_argument('--max-len', type=int, default=20)
    parser.add_argument('--save-model', type=bool, default=False)
    parser.add_argument('--save-stats', type=bool, default=False)
    parser.add_argument('--workers', type=int, default=2)

    main(parser.parse_args())
