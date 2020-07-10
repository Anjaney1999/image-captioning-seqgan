import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import os.path as path
import logging
import time
from nltk.translate.bleu_score import corpus_bleu

from datasets import ImageCaptionDataset
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

logging.basicConfig(level=logging.INFO)


def main(args):
    with open(args.storage + '/processed_data/' + args.dataset + '/word_index.json') as f:
        word_index = json.load(f)

    vocab_size = len(word_index)

    checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/' + args.checkpoint_filename

    generator = Generator(attention_dim=args.attention_dim,
                          gru_units=args.gru_units,
                          vocab_size=vocab_size,
                          embedding_dim=args.embedding_dim)
    generator.to(device)

    encoder = None

    optimizer = optim.Adam(generator.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=word_index['<pad>']).to(device)

    if path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        generator.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if args.use_image_features:
        train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='train',
                                use_img_feats=True, transform=None,
                                img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='val',
                                use_img_feats=True, transform=None,
                                img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=args.batch_size, shuffle=True, num_workers=1)
    else:
        encoder = Encoder(args.cnn_architecture)
        encoder.to(device)

        train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='train',
                                use_img_feats=False, transform=data_transforms,
                                img_src_path=args.storage + '/images',
                                cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),
            batch_size=args.batch_size, shuffle=True, num_workers=0)

        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='val',
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

        torch.save({
            'gen_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, args.storage + '/ckpts/' + args.dataset + '/gen/{}_{}_{}.pth'.format('mle_gen', args.cnn_architecture, e))

        scheduler.step()


def gen_mle_train(epoch,
                  encoder,
                  generator,
                  optimizer,
                  criterion,
                  train_loader,
                  args):

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

        preds = pack_padded_sequence(preds, output_lens, batch_first=True)[0]
        targets = pack_padded_sequence(caps[:, 1:], output_lens, batch_first=True)[0]

        loss = criterion(preds, targets)

        loss.backward()

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
                         'Top 1 accuracy [{:.4f}]({:.3f})\t'.format(epoch,
                                                                    batch_id,
                                                                    time.time() - start_time,
                                                                    losses.avg, losses.val,
                                                                    top5.avg, top5.val,
                                                                    top1.avg, top1.val))


def validate(epoch,
             encoder,
             generator,
             criterion,
             val_loader,
             word_index,
             args):
    losses = AverageMeter()
    top5 = AverageMeter()
    top1 = AverageMeter()

    if not args.use_image_features:
        encoder.eval()

    generator.eval()

    references = []
    hypotheses = []

    with torch.no_grad():

        for batch_id, (imgs, caps, cap_lens, matching_caps) in enumerate(val_loader):

            imgs, caps, cap_lens = imgs.to(device), caps.to(device), cap_lens.to(device)

            cap_lens = cap_lens.squeeze(-1)

            if not args.use_image_features:
                imgs = encoder(imgs)

            preds, caps, output_lens, alphas, indices = generator(imgs, caps, cap_lens)

            preds_clone = preds.clone()

            preds = pack_padded_sequence(preds, output_lens, batch_first=True)[0]
            targets = pack_padded_sequence(caps[:, 1:], output_lens, batch_first=True)[0]

            loss = criterion(preds, targets)

            top1_acc = categorical_accuracy(preds, targets, 1)
            top1.update(top1_acc, sum(output_lens))

            top5_acc = categorical_accuracy(preds, targets, 5)
            top5.update(top5_acc, sum(output_lens))

            losses.update(loss.item(), sum(output_lens))

            matching_caps = matching_caps[indices]

            for cap_set in matching_caps.tolist():
                caps = []
                for caption in cap_set:
                    cap = [word_id for word_id in caption
                           if word_id != word_index['<start>'] and word_id != word_index['<pad>']]
                    caps.append(cap)
                references.append(caps)

            word_idxs = torch.max(preds_clone, dim=2)[1]
            word_idxs, _ = pad_generated_captions(word_idxs.cpu().numpy(), word_index)
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs
                                   if idx != word_index['<start>'] and idx != word_index['<pad>']])

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        logging.info('Epoch: [{}]\t'
                     'Batch: [{}]\t'
                     'Loss [{:.4f}]\t'
                     'Top 5 accuracy [{:.4f}]\t'
                     'Top 1 accuracy [{:.4f}]\t'
                     'bleu-1 [{:.3f}]\t'
                     'bleu-2 [{:.3f}]\t'
                     'bleu-3 [{:.3f}]\t'
                     'bleu-4 [{:.3f}]\t'.format(epoch,
                                                batch_id,
                                                losses.avg,
                                                top5.avg,
                                                top1.avg,
                                                bleu_1, bleu_2, bleu_3, bleu_4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Maximum Likelihood Estimation Training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
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

    main(parser.parse_args())
