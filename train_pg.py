import argparse
import csv
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
from rollout import Rollout
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
    encoder = None
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.gen_lr)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.dis_lr)
    dis_criterion = nn.BCELoss().to(device)
    gen_pg_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=word_index['<pad>']).to(device)
    gen_mle_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=word_index['<pad>']).to(device)

    rollout = Rollout(generator, 0.0, args.rollout_num)

    if args.use_image_features:
        gen_train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='train', use_img_feats=True,
                                transform=None, img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)
        gen_iter = iter(gen_train_loader)
        dis_train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='discriminator', split_type='train', use_img_feats=True,
                                transform=None, img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)
        dis_iter = iter(dis_train_loader)
        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='val', use_img_feats=True,
                                transform=None, img_src_path=None, cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)
    else:
        encoder = Encoder(args.cnn_architecture)
        encoder.to(device)
        if not args.use_image_features:
            encoder.eval()
        gen_train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='train', use_img_feats=False,
                                transform=data_transforms, img_src_path=args.storage + '/images',
                                cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),  batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)
        gen_iter = iter(gen_train_loader)
        dis_train_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='discriminator', split_type='train', use_img_feats=False,
                                transform=data_transforms, img_src_path=args.storage + '/images',
                                cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'),  batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)
        dis_iter = iter(dis_train_loader)
        val_loader = DataLoader(
            ImageCaptionDataset(dataset=args.dataset, model='generator', split_type='val', use_img_feats=False,
                                transform=data_transforms, img_src_path=args.storage + '/images',
                                cnn_architecture=args.cnn_architecture,
                                processed_data_path=args.storage + '/processed_data'), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

    gen_batch_id = 0
    dis_batch_id = 0
    gen_epoch = 0
    dis_epoch = 0

    if path.isfile(gen_checkpoint_path):
        logging.info('loaded generator checkpoint')
        checkpoint = torch.load(gen_checkpoint_path)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        generator.to(device)
        if args.gen_checkpoint_filename.split('_')[0] == 'PG':
            gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            gen_batch_id = checkpoint['gen_batch_id']
            gen_epoch = checkpoint['gen_epoch']

    if path.isfile(dis_checkpoint_path):
        logging.info('loaded discriminator checkpoint')
        checkpoint = torch.load(dis_checkpoint_path)
        discriminator.load_state_dict(checkpoint['dis_state_dict'])
        if args.dis_checkpoint_filename.split('_')[0] == 'PG':
            dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
            dis_batch_id = checkpoint['dis_batch_id']
            dis_epoch = checkpoint['dis_epoch']

    gen_mle_losses = AverageMeter()
    gen_pg_losses = AverageMeter()
    dis_losses = AverageMeter()
    dis_acc = AverageMeter()

    completed_epoch = False

    for epoch in range(args.epochs):
        if gen_epoch == args.gen_epochs:
            break
        i = 0
        while i < args.g_steps:
            try:
                start_time = time.time()
                imgs, caps, cap_lens = next(gen_iter)
                cap_lens = cap_lens.squeeze(-1)
                gen_train(imgs=imgs, caps=caps, cap_lens=cap_lens,
                          generator=generator, rollout=rollout, discriminator=discriminator,
                          gen_optimizer=gen_optimizer, gen_pg_criterion=gen_pg_criterion,
                          gen_mle_criterion=gen_mle_criterion, word_index=word_index,
                          args=args, encoder=encoder,
                          pg_losses=gen_pg_losses, mle_losses=gen_mle_losses)
                time_taken = time.time() - start_time
                if epoch % args.print_freq == 0:
                    logging.info('GENERATOR: ADV EPOCH: [{}]\t'
                                 'GEN Epoch: [{}]\t'
                                 'Batch: [{}]\t'
                                 'Time per batch: [{:.3f}]\t'
                                 'PG Loss [{:.4f}]({:.3f})\t'
                                 'MLE Loss [{:.4f}]({:.3f})\t'.format(epoch, gen_epoch, gen_batch_id, time_taken,
                                                                      gen_pg_losses.avg, gen_pg_losses.val,
                                                                      gen_mle_losses.avg, gen_mle_losses.val))
                    if args.save_stats:
                        with open(args.storage + '/stats/' + args.dataset +
                                  '/gen/{}_'.format('TRAIN_PG_GEN') +
                                  'LR_{}_'.format(str(args.gen_lr)) +
                                  'ROLLOUT_{}_'.format(args.rollout_num) +
                                  'G-STEPS_{}_'.format(args.g_steps) +
                                  'D-STEPS_{}_'.format(args.d_steps) +
                                  'CNN-ARCH_{}.csv'.format(args.cnn_architecture), "a+") as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch, gen_epoch, gen_batch_id, gen_pg_losses.avg, gen_pg_losses.val,
                                             gen_mle_losses.avg, gen_mle_losses.val])
                gen_batch_id += 1
                i += 1
            except StopIteration:
                gen_batch_id = 0
                gen_pg_losses.reset()
                gen_mle_losses.reset()
                gen_epoch += 1
                gen_iter = iter(gen_train_loader)
                completed_epoch = True
        i = 0
        while i < args.d_steps:
            try:
                start_time = time.time()
                imgs, mismatched_imgs, caps, cap_lens = next(dis_iter)
                cap_lens = cap_lens.squeeze(-1)
                dis_train(imgs=imgs, mismatched_imgs=mismatched_imgs, caps=caps, cap_lens=cap_lens,
                          generator=generator, discriminator=discriminator,
                          dis_optimizer=dis_optimizer, encoder=encoder,
                          dis_criterion=dis_criterion, word_index=word_index,
                          args=args, losses=dis_losses, acc=dis_acc)
                time_taken = time.time() - start_time
                if epoch % args.print_freq == 0:
                    logging.info('DISCRIMINATOR: ADV Epoch: [{}]\t'
                                 'DIS Epoch: [{}]\t'
                                 'Batch: [{}]\t'
                                 'Time per batch: [{:.3f}]\t'
                                 'Loss [{:.4f}]({:.3f})\t'
                                 'Accuracy [{:.4f}]({:.3f})'.format(epoch, dis_epoch, dis_batch_id, time_taken,
                                                                    dis_losses.avg, dis_losses.val,
                                                                    dis_acc.val, dis_acc.avg))
                    if args.save_stats:
                        with open(args.storage + '/stats/' + args.dataset +
                                  '/dis/{}_'.format('TRAIN_PG_DIS') +
                                  'LR_{}_'.format(args.dis_lr) +
                                  'ROLLOUT_{}_'.format(args.rollout_num) +
                                  'G-STEPS_{}_'.format(args.g_steps) +
                                  'D-STEPS_{}_'.format(args.d_steps) +
                                  'CNN-ARCH_{}.csv'.format(args.cnn_architecture), 'a+') as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch, gen_epoch, gen_batch_id, dis_epoch, dis_batch_id, dis_losses.avg,
                                             dis_losses.val, dis_acc.val, dis_acc.avg])
                dis_batch_id += 1
                i += 1
            except StopIteration:
                dis_losses.reset()
                dis_acc.reset()
                dis_epoch += 1
                dis_batch_id = 0
                dis_iter = iter(dis_train_loader)

        if epoch % args.val_freq == 0 or completed_epoch:
            validate(epoch=epoch, gen_epoch=gen_epoch, gen_batch_id=gen_batch_id, generator=generator,
                     criterion=gen_mle_criterion, val_loader=val_loader, word_index=word_index,
                     args=args, encoder=encoder)
            if args.save_models:
                torch.save(
                    {'gen_state_dict': generator.state_dict(), 'optimizer_state_dict': gen_optimizer.state_dict(),
                     'gen_batch_id': gen_batch_id, 'gen_epoch': gen_epoch}, args.storage + '/ckpts/' + args.dataset +
                                                                            '/gen/{}_'.format('TRAIN_PG_GEN') +
                                                                            'EPOCH_{}_'.format(epoch) +
                                                                            'GEN_EPOCH_{}_'.format(gen_epoch) +
                                                                            'GEN_BATCH_{}_'.format(gen_batch_id) +
                                                                            'LR_{}_'.format(args.gen_lr) +
                                                                            'ROLLOUT_{}_'.format(args.rollout_num) +
                                                                            'G-STEPS_{}_'.format(args.g_steps) +
                                                                            'D-STEPS_{}_'.format(args.d_steps) +
                                                                            'CNN-ARCH_{}.pth'.format(args.cnn_architecture))
                torch.save(
                    {'dis_state_dict': discriminator.state_dict(), 'optimizer_state_dict': dis_optimizer.state_dict(),
                     'dis_batch_id': dis_batch_id, 'dis_epoch': dis_epoch}, args.storage + '/ckpts/' + args.dataset +
                                                                            '/dis/{}_'.format('TRAIN_PG_DIS') +
                                                                            'EPOCH_{}_'.format(epoch) +
                                                                            'DIS_EPOCH_{}_'.format(dis_epoch) +
                                                                            'DIS_BATCH_{}_'.format(dis_batch_id) +
                                                                            'LR_{}_'.format(args.dis_lr) +
                                                                            'ROLLOUT_{}_'.format(args.rollout_num) +
                                                                            'G-STEPS_{}_'.format(args.g_steps) +
                                                                            'D-STEPS_{}_'.format(args.d_steps) +
                                                                            'CNN-ARCH_{}.pth'.format(args.cnn_architecture))
            completed_epoch = False

        if args.rollout_num != 0:
            rollout.update_params()


def sample_from_start(imgs, caps, cap_lens, generator, word_index, args):
    with torch.no_grad():
        fake_caps, hidden_states = generator.sample(cap_len=max(torch.max(cap_lens).item(), args.max_len) - 1,
                                                    col_shape=caps.shape[1],
                                                    img_feats=imgs,
                                                    input_word=caps[:, 0],
                                                    hidden_state=None, sampling_method=args.sampling_method)
        fake_caps, fake_cap_lens = pad_generated_captions(fake_caps.cpu().numpy(), word_index)
        fake_caps, fake_cap_lens = torch.LongTensor(fake_caps).to(device), torch.LongTensor(fake_cap_lens)

    return fake_caps, fake_cap_lens, hidden_states


def dis_train(imgs, mismatched_imgs, caps, cap_lens, encoder, generator, discriminator, dis_optimizer, dis_criterion,
              word_index, losses, acc, args):
    discriminator.train()
    generator.eval()
    imgs, mismatched_imgs, caps = imgs.to(device), mismatched_imgs.to(device), caps.to(device)
    if not args.use_image_features:
        imgs = encoder(imgs)
        mismatched_imgs = encoder(mismatched_imgs)

    fake_caps, fake_cap_lens, _ = sample_from_start(imgs, caps, cap_lens, generator, word_index, args)
    ones = torch.ones(caps.shape[0]).to(device)
    zeros = torch.zeros(caps.shape[0]).to(device)
    dis_optimizer.zero_grad()
    true_preds = discriminator(imgs, caps, cap_lens)
    false_preds = discriminator(mismatched_imgs, caps, cap_lens)
    fake_preds = discriminator(imgs, fake_caps, fake_cap_lens)
    loss = dis_criterion(true_preds, ones) + 0.5 * dis_criterion(false_preds, zeros) + \
           0.5 * dis_criterion(fake_preds, zeros)
    loss.backward()
    dis_optimizer.step()
    losses.update(loss.item())
    acc.update(binary_accuracy(true_preds, ones).item())
    acc.update(binary_accuracy(false_preds, zeros).item())
    acc.update(binary_accuracy(fake_preds, zeros).item())


def gen_train(imgs, caps, cap_lens, encoder, generator, discriminator, rollout, gen_optimizer, gen_pg_criterion,
              gen_mle_criterion, word_index, pg_losses, mle_losses, args):
    discriminator.eval()
    generator.eval()
    imgs, caps = imgs.to(device), caps.to(device)
    if not args.use_image_features:
        imgs = encoder(imgs)

    fake_caps, fake_cap_lens, hidden_states = sample_from_start(imgs, caps, cap_lens, generator, word_index, args)
    rewards = rollout.get_reward(samples=fake_caps, sample_cap_lens=fake_cap_lens, hidden_states=hidden_states,
                                 discriminator=discriminator, img_feats=imgs, word_index=word_index,
                                 col_shape=caps.shape[1], args=args)

    generator.train()
    gen_optimizer.zero_grad()

    pg_preds, pg_caps, pg_output_lens, alphas, pg_indices = generator(imgs, fake_caps, fake_cap_lens)
    rewards = rewards[pg_indices]
    pg_loss = 0.0
    for i in range(caps.shape[0]):
        pg_loss += torch.sum(gen_pg_criterion(pg_preds[i, :pg_output_lens[i]],
                                              pg_caps[i, 1:pg_output_lens[i] + 1]) * rewards[i, :pg_output_lens[i]])
    pg_loss = pg_loss / (1.0 * caps.shape[0])
    loss = args.lambda1 * (pg_loss + args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean())

    if args.lambda2 != 0.0:
        mle_loss = 0.0
        mle_preds, mle_caps, mle_output_lens, mle_alphas, mle_indices = generator(imgs, caps, cap_lens)
        for i in range(mle_caps.shape[0]):
            mle_loss += gen_mle_criterion(mle_preds[i, :], mle_caps[i, 1:])
        mle_loss = mle_loss / (1.0 * mle_caps.shape[0])
        mle_loss += args.alpha_c * ((1. - mle_alphas.sum(dim=1)) ** 2).mean()
        loss += args.lambda2 * mle_loss
        mle_losses.update(mle_loss.item(), sum(mle_output_lens))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip)
    gen_optimizer.step()
    pg_losses.update(pg_loss.item(), sum(pg_output_lens))


def validate(epoch, gen_epoch, gen_batch_id, encoder, generator, criterion, val_loader, word_index, args):
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
            fake_caps, _ = generator.sample(cap_len=max(torch.max(cap_lens).item(), args.max_len),
                                            col_shape=caps.shape[1],
                                            img_feats=imgs[indices],
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

        logging.info('VALIDATION')
        logging.info('ADV Epoch: [{}]\t'
                     'GEN Epoch: [{}]\t'
                     'Batch: [{}]\t'
                     'Top 5 accuracy [{:.4f}]\t'
                     'Top 1 accuracy [{:.4f}]\n'
                     'bleu-1 [{:.3f}]\t'
                     'bleu-2 [{:.3f}]\t'
                     'bleu-3 [{:.3f}]\t'
                     'bleu-4 [{:.3f}]\n'
                     'TF bleu-1 [{:.3f}]\t'
                     'TF bleu-2 [{:.3f}]\t'
                     'TF bleu-3 [{:.3f}]\t'
                     'TF bleu-4 [{:.3f}]\t'.format(epoch, gen_epoch, gen_batch_id, top5.avg, top1.avg,
                                                   bleu_1, bleu_2, bleu_3, bleu_4, bleu_1_tf, bleu_2_tf,
                                                   bleu_3_tf, bleu_4_tf))
        if args.save_stats:
            with open(args.storage + '/stats/' + args.dataset +
                      '/gen/{}_'.format('VAL_PG_GEN') +
                      'LR_{}_'.format(args.gen_lr) +
                      'ROLLOUT_{}_'.format(args.rollout_num) +
                      'G-STEPS_{}_'.format(args.g_steps) +
                      'D-STEPS_{}_'.format(args.d_steps) +
                      'CNN-ARCH_{}.csv'.format(args.cnn_architecture), 'a+') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, gen_epoch, gen_batch_id, top5.avg, top5.val, top1.avg, top1.val, bleu_1,
                                 bleu_2, bleu_3, bleu_4, bleu_1_tf, bleu_2_tf, bleu_3_tf, bleu_4_tf])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Training via Policy Gradients')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--gen-epochs', type=int, default=5)
    parser.add_argument('--g-steps', type=int, default=1)
    parser.add_argument('--d-steps', type=int, default=1)
    parser.add_argument('--gen-lr', type=float, default=1e-4)
    parser.add_argument('--dis-lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--alpha-c', type=float, default=1.0)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.0)
    parser.add_argument('--val-freq', type=int, default=100)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-stats', type=bool, default=True)
    parser.add_argument('--save-models', type=bool, default=True)
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--image-path', type=str, default='images')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--rollout-num', type=int, default=0)
    parser.add_argument('--max-len', type=int, default=20)
    parser.add_argument('--dis-embedding-dim', type=int, default=512)
    parser.add_argument('--dis-gru-units', type=int, default=512)
    parser.add_argument('--gen-embedding-dim', type=int, default=512)
    parser.add_argument('--gen-gru-units', type=int, default=512)
    parser.add_argument('--attention-dim', type=int, default=512)
    parser.add_argument('--gen-checkpoint-filename', type=str, default='mle_gen_resnet152_5.pth')
    parser.add_argument('--dis-checkpoint-filename', type=str, default='pretrain_dis_5_multinomial_resnet152.pth')
    parser.add_argument('--use-image-features', type=bool, default=True)
    parser.add_argument('--sampling-method', type=str, default='multinomial')
    parser.add_argument('--workers', type=int, default=2)

    main(parser.parse_args())
