import argparse
import json
import logging
import os.path as path
import time

import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from datasets import ImageCaptionDataset
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

logging.basicConfig(level=logging.INFO)


def main(args):
    with open(args.storage + '/processed_data/' + args.dataset + '/word_index.json') as f:
        word_index = json.load(f)

    vocab_size = len(word_index)

    gen_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.gen_checkpoint_filename
    dis_checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/dis/' + args.dis_checkpoint_filename

    generator = Generator(attention_dim=args.attention_dim, gru_units=args.gen_gru_units, vocab_size=vocab_size,
                          embedding_dim=args.gen_embedding_dim)
    generator.to(device)

    discriminator = GRUDiscriminator(embedding_dim=args.dis_embedding_dim, gru_units=args.dis_gru_units,
                                     vocab_size=vocab_size, encoder_dim=2048)

    discriminator.to(device)

    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.dis_lr)
    dis_criterion = nn.BCELoss().to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=args.gen_lr)
    gen_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    encoder = None

    gen_iter = None
    dis_iter = None
    val_loader = None

    if path.isfile(gen_checkpoint_path):
        checkpoint = torch.load(gen_checkpoint_path)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        generator.to(device)
        if args.gen_checkpoint_filename.split('_')[0] == 'pg':
            gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])

    if path.isfile(dis_checkpoint_path):
        checkpoint = torch.load(dis_checkpoint_path)
        discriminator.load_state_dict(checkpoint['dis_state_dict'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])

    if args.use_image_features:
        train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='train', use_img_feats=True, transform=None,
                                img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'), batch_size=args.batch_size,
            shuffle=True, num_workers=1)

        gen_iter = iter(train_loader)
        dis_iter = iter(train_loader)

        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='val', use_img_feats=True, transform=None,
                                img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'), batch_size=args.batch_size,
            shuffle=True, num_workers=1)
    else:
        encoder = Encoder(args.cnn_architecture)
        encoder.to(device)

        train_loader = DataLoader(ImageCaptionDataset(dataset=args.dataset, split_type='train', use_img_feats=False,
                                                      transform=data_transforms, img_src_path=args.storage + '/images',
                                                      cnn_architecture=args.cnn_architecture,
                                                      processed_data_path=args.storage + '/processed_data'),
                                  batch_size=args.batch_size, shuffle=True, num_workers=0)

        gen_iter = iter(train_loader)
        dis_iter = iter(train_loader)

        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, split_type='val', use_img_feats=False, transform=data_transforms,
                                img_src_path=args.storage + '/images', cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'), batch_size=args.batch_size,
            shuffle=True, num_workers=0)

    gen_losses = AverageMeter()
    gen_top5 = AverageMeter()
    gen_top1 = AverageMeter()

    dis_losses = AverageMeter()
    dis_acc_pos = AverageMeter()
    dis_acc_neg = AverageMeter()

    gen_batch_id = 0
    dis_batch_id = 0
    epoch_completed = False

    gen_epoch = 0
    dis_epoch = 0

    for _ in range(args.epochs):
        for _ in range(args.g_steps):
            try:
                imgs, caps, cap_lens = next(gen_iter)
                cap_lens = cap_lens.squeeze(-1)

                for i in range(args.g_epochs):
                    time_taken, losses, top5, top1 = gen_train(imgs=imgs, caps=caps, cap_lens=cap_lens,
                                                               generator=generator, discriminator=discriminator,
                                                               gen_optimizer=gen_optimizer, gen_criterion=gen_criterion,
                                                               word_index=word_index, args=args, encoder=None,
                                                               losses=gen_losses, top1=gen_top1, top5=gen_top5)

                    if gen_batch_id % args.gen_print_freq == 0 and i == (args.g_epochs - 1):
                        logging.info('GENERATOR: Epoch: [{}]\t'
                                     'Batch: [{}]\t'
                                     'Time per batch: [{:.3f}]\t'
                                     'Loss [{:.4f}]({:.3f})\t'
                                     'Top 5 accuracy [{:.4f}]({:.3f})\t'
                                     'Top 1 accuracy [{:.4f}]({:.3f})\t'.format(gen_epoch, gen_batch_id, time_taken,
                                                                                losses.avg, losses.val, top5.avg,
                                                                                top5.val, top1.avg, top1.val))
                gen_batch_id += 1
            except StopIteration:
                epoch_completed = True
                gen_epoch += 1
                print('Generator has iterated over data fully')
                gen_iter = iter(train_loader)
        for _ in range(args.d_steps):
            try:
                imgs, caps, cap_lens = next(dis_iter)
                cap_lens = cap_lens.squeeze(-1)
                for i in range(args.d_epochs):
                    time_taken, losses, acc_pos, acc_neg = dis_train(imgs=imgs, caps=caps, cap_lens=cap_lens,
                                                                     generator=generator, discriminator=discriminator,
                                                                     dis_optimizer=dis_optimizer,
                                                                     dis_criterion=dis_criterion, word_index=word_index,
                                                                     args=args, encoder=None, losses=dis_losses,
                                                                     acc_pos=dis_acc_pos, acc_neg=dis_acc_neg)

                    if dis_batch_id % args.dis_print_freq == 0 and i == (args.d_epochs - 1):
                        logging.info('DISCRIMINATOR: Epoch: [{}]\t'
                                     'Batch: [{}]\t'
                                     'Time per batch: [{:.3f}]\t'
                                     'Loss [{:.4f}]({:.3f})\t'
                                     'Pos Accuracy [{:.4f}]({:.3f})\t'
                                     'Neg Accuracy [{:.4f}]({:.3f})'.format(dis_epoch, dis_batch_id, time_taken,
                                                                            losses.avg, losses.val, acc_pos.avg,
                                                                            acc_pos.val, acc_neg.avg, acc_neg.val))
                dis_batch_id += 1
            except StopIteration:
                dis_epoch += 1
                print('Discriminator has iterated over data fully')
                dis_iter = iter(train_loader)

        if epoch_completed:
            validate(epoch=gen_epoch, generator=generator, criterion=gen_criterion, val_loader=val_loader,
                     word_index=word_index, args=args, encoder=None)

            torch.save({'gen_state_dict': generator.state_dict(), 'optimizer_state_dict': gen_optimizer.state_dict()},
                       args.storage + '/ckpts/' + args.dataset + '/gen/{}-{}-{}.pth'.format('pg_gen',
                                                                                            args.cnn_architecture,
                                                                                            gen_epoch))

            torch.save(
                {'dis_state_dict': discriminator.state_dict(), 'optimizer_state_dict': dis_optimizer.state_dict()},
                args.storage + '/ckpts/' + args.dataset + '/dis/{}-{}-{}.pth'.format('pg_dis', args.cnn_architecture,
                                                                                     gen_epoch))
            epoch_completed = True

            gen_losses.reset()
            gen_top5.reset()
            gen_top1.reset()

            dis_losses.reset()
            dis_acc_pos.reset()
            dis_acc_neg.reset()

            gen_batch_id = 0
            dis_batch_id = 0


def dis_train(imgs, caps, cap_lens, encoder, generator, discriminator, dis_optimizer, dis_criterion,
              word_index, losses, acc_pos, acc_neg, args):

    if not args.use_image_features:
        encoder.eval()

    discriminator.train()
    generator.eval()

    start_time = time.time()

    imgs, caps, cap_lens = imgs.to(device), caps.to(device), cap_lens.to(device)

    if not args.use_image_features:
        imgs = encoder(imgs)

    dis_optimizer.zero_grad()

    with torch.no_grad():
        fake_caps, _ = generator.sample(cap_len=max(cap_lens) - 1, img_feats=imgs, input_word=caps[:, 0],
                                        hidden_state=None, sampling_method='max')

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

    return time.time() - start_time, losses, acc_pos, acc_neg


def rollout(samples, hidden_states, generator, discriminator, img_feats, word_index, sample_cap_lens, rollout_num):
    with torch.no_grad():
        rewards = []
        cap_len = max(sample_cap_lens)

        for i in range(rollout_num):
            for j in range(1, cap_len - 1):
                incomplete_fake_caps = generator.sample(cap_len=cap_len - (j + 1), img_feats=img_feats,
                                                        input_word=samples[:, j], hidden_state=hidden_states[:, j - 1])

                fake_caps = torch.cat([samples[:, :j + 1], incomplete_fake_caps], dim=-1)
                fake_caps, fake_cap_lens = pad_generated_captions(fake_caps.cpu().numpy(), word_index)
                fake_caps = torch.from_numpy(fake_caps)
                fake_cap_lens = torch.LongTensor(fake_cap_lens).to(device)
                fake_caps = fake_caps.to(device)
                reward = discriminator(img_feats, fake_caps, fake_cap_lens)

                if i == 0:
                    rewards.append(reward)
                else:
                    rewards[j - 1] += reward

            rewards = torch.stack(rewards).to(device) / (1.0 * rollout_num)
            reward = discriminator(img_feats, samples, sample_cap_lens)
            reward = reward.unsqueeze(0)
            rewards = torch.cat((rewards, reward)).squeeze(-1)

            return rewards.permute(1, 0)


def gen_train(imgs, caps, cap_lens, encoder, generator, discriminator, gen_optimizer, gen_criterion,
              word_index, losses, top1, top5, args):
    start_time = time.time()

    if not args.use_image_features:
        encoder.eval()
    generator.train()
    discriminator.eval()

    imgs, caps, cap_lens = imgs.to(device), caps.to(device), cap_lens.to(device)

    if not args.use_image_features:
        imgs = encoder(imgs)

    gen_optimizer.zero_grad()

    with torch.no_grad():
        fake_caps, hidden_states = generator.sample(cap_len=max(cap_lens) - 1, img_feats=imgs, input_word=caps[:, 0],
                                                    hidden_state=None, sampling_method='max')

    fake_caps, fake_cap_lens = pad_generated_captions(fake_caps.cpu().numpy(), word_index)
    fake_caps, fake_cap_lens = torch.LongTensor(fake_caps).to(device), torch.LongTensor(fake_cap_lens).to(device)
    rewards = rollout(samples=fake_caps, sample_cap_lens=fake_cap_lens, hidden_states=hidden_states,
                      generator=generator, discriminator=discriminator, img_feats=imgs, word_index=word_index,
                      rollout_num=args.rollout_num)

    rewards = rewards.detach().to(device)

    preds, fake_caps, output_lens, _, indices = generator(imgs, fake_caps, fake_cap_lens)

    preds = pack_padded_sequence(preds, output_lens, batch_first=True)[0]
    targets = pack_padded_sequence(fake_caps[:, 1:], output_lens, batch_first=True)[0]
    rewards = rewards[indices]
    rewards = pack_padded_sequence(rewards, output_lens, batch_first=True)[0]

    loss = gen_criterion(preds, targets)
    loss = loss * rewards
    loss = torch.mean(loss)
    loss = -1.0 * loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip)
    gen_optimizer.step()

    top1_acc = categorical_accuracy(preds, targets, 1)
    top1.update(top1_acc, sum(output_lens))

    top5_acc = categorical_accuracy(preds, targets, 5)
    top5.update(top5_acc, sum(output_lens))

    losses.update(loss.item(), sum(output_lens))

    return time.time() - start_time, losses, top5, top1


def validate(epoch, encoder, generator, criterion, val_loader, word_index, args):
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
                    cap = [word_id for word_id in caption if
                           word_id != word_index['<start>'] and word_id != word_index['<pad>']]
                    caps.append(cap)
                references.append(caps)

            word_idxs = torch.max(preds_clone, dim=2)[1]
            word_idxs, _ = pad_generated_captions(word_idxs.cpu().numpy(), word_index)
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs if idx != word_index['<start>'] and idx != word_index['<pad>']])

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
                     'bleu-4 [{:.3f}]\t'.format(epoch, batch_id, losses.avg, top5.avg, top1.avg, bleu_1, bleu_2, bleu_3,
                                                bleu_4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Training via Policy Gradients')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--g-steps', type=int, default=1)
    parser.add_argument('--d-steps', type=int, default=1)
    parser.add_argument('--g-epochs', type=int, default=1)
    parser.add_argument('--d-epochs', type=int, default=1)
    parser.add_argument('--gen-lr', type=float, default=1e-3)
    parser.add_argument('--dis-lr', type=float, default=1e-3)
    parser.add_argument('--gen-print-freq', type=int, default=10)
    parser.add_argument('--dis-print-freq', type=int, default=20)
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--image-path', type=str, default='images')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--rollout-num', type=int, default=20)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--dis-embedding-dim', type=int, default=256)
    parser.add_argument('--dis-gru-units', type=int, default=256)
    parser.add_argument('--gen-embedding-dim', type=int, default=512)
    parser.add_argument('--gen-gru-units', type=int, default=512)
    parser.add_argument('--attention-dim', type=int, default=512)
    parser.add_argument('--gen-checkpoint-filename', type=str, default='mle_gen-resnet152-1.pth')
    parser.add_argument('--dis-checkpoint-filename', type=str, default='')
    parser.add_argument('--use-image-features', type=bool, default=True)

    main(parser.parse_args())
