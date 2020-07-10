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

    generator = Generator(embedding_dim=args.gen_embedding_dim,
                          attention_dim=args.attention_dim,
                          gru_units=args.gen_gru_units,
                          vocab_size=vocab_size)
    generator.to(device)

    discriminator = GRUDiscriminator(embedding_dim=args.dis_embedding_dim,
                                     gru_units=args.dis_gru_units,
                                     vocab_size=vocab_size,
                                     encoder_dim=2048)

    discriminator.to(device)

    encoder = None
    train_loader = None

    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
    dis_criterion = nn.BCELoss().to(device)

    gen_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.gen_checkpoint_filename
    dis_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/dis/' + args.dis_checkpoint_filename

    if path.isfile(gen_checkpoint_path):
        checkpoint = torch.load(gen_checkpoint_path)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        generator.to(device)

    if path.isfile(dis_checkpoint_path):
        checkpoint = torch.load(dis_checkpoint_path)
        discriminator.load_state_dict(checkpoint['dis_state_dict'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        discriminator.to(device)

    if args.use_image_features:
        train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='train',
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

    for e in range(args.epochs):
        train(epoch=e, generator=generator,
              discriminator=discriminator, dis_optimizer=dis_optimizer,
              dis_criterion=dis_criterion, train_loader=train_loader,
              word_index=word_index, args=args, encoder=None)

        torch.save({
            'dis_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': dis_optimizer.state_dict()
        }, args.storage + '/ckpts/' + args.dataset + '/dis/{}_{}_{}.pth'.format('pretrain_dis',
                                                                                args.cnn_architecture, e))

        logging.info('Completed epoch: ' + e)


def train(epoch, encoder, generator, discriminator, dis_optimizer, dis_criterion, train_loader, word_index, args):

    losses = AverageMeter()
    acc_pos = AverageMeter()
    acc_neg = AverageMeter()

    if not args.use_image_features:
        encoder.eval()

    discriminator.train()
    generator.eval()

    for batch_id, (imgs, caps, cap_lens) in enumerate(train_loader):
        start_time = time.time()

        imgs, caps = imgs.to(device), caps.to(device)

        cap_lens = cap_lens.squeeze(-1)

        if not args.use_image_features:
            imgs = encoder(imgs)

        dis_optimizer.zero_grad()

        with torch.no_grad():
            fake_caps, _ = generator.sample(cap_len=max(cap_lens) - 1, img_feats=imgs,
                                            input_word=caps[:, 0], hidden_state=None, sampling_method='max')

        fake_caps, fake_cap_lens = pad_generated_captions(fake_caps.cpu().numpy(), word_index)
        fake_caps, fake_cap_lens = torch.LongTensor(fake_caps).to(device), torch.LongTensor(fake_cap_lens).to(device)

        real_preds = discriminator(imgs, caps, cap_lens)
        fake_preds = discriminator(imgs, fake_caps, fake_cap_lens)

        ones = torch.ones(caps.shape[0]).to(device)
        zeros = torch.zeros(caps.shape[0]).to(device)

        real_loss = dis_criterion(real_preds, ones)
        fake_loss = dis_criterion(fake_preds, zeros)
        loss = real_loss + fake_loss

        loss.backward()
        dis_optimizer.step()

        losses.update(loss.item())

        acc_pos.update(binary_accuracy(real_preds, label='pos').item())
        acc_neg.update(binary_accuracy(fake_preds, label='neg').item())

        if batch_id % args.print_freq == 0:
            logging.info('Epoch: [{}]\t'
                         'Batch: [{}]\t'
                         'Time per batch: [{:.3f}]\t'
                         'Loss [{:.4f}]({:.3f})\t'
                         'Pos Accuracy [{:.4f}]({:.3f})\t'
                         'Neg Accuracy [{:.4f}]({:.3f})'.format(epoch, batch_id, time.time() - start_time, losses.avg,
                                                                losses.val, acc_pos.avg, acc_pos.val, acc_neg.avg,
                                                                acc_neg.val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-train discriminator')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--image-path', type=str, default='images')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--dis-embedding-dim', type=int, default=256)
    parser.add_argument('--dis-gru-units', type=int, default=256)
    parser.add_argument('--gen-embedding-dim', type=int, default=512)
    parser.add_argument('--gen-gru-units', type=int, default=512)
    parser.add_argument('--attention-dim', type=int, default=512)
    parser.add_argument('--gen-checkpoint-filename', type=str, default='')
    parser.add_argument('--dis-checkpoint-filename', type=str, default='')
    parser.add_argument('--use-image-features', type=bool, default=True)

    main(parser.parse_args())
