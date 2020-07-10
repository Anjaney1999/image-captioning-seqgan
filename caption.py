import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import argparse
from utils import pil_loader
from models import *
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def caption_image(encoder, generator, image_path, word_index, index_word, beam_size=5):

    vocab_size = len(word_index)

    img = pil_loader(image_path)

    img = data_transforms(img)

    img = torch.FloatTensor(img).to(device)

    img = img.unsqueeze(0)
    img_feats = encoder(img)

    img_feats = img_feats.expand(beam_size, img_feats.shape[1], img_feats.shape[-1])

    prev_words = torch.LongTensor([[word_index['<start>']]] * beam_size).to(device)

    sents = prev_words
    top_preds = torch.zeros(beam_size, 1).to(device)
    alphas = torch.ones(beam_size, 1, img_feats.size(1)).to(device)

    completed_sents = []
    completed_sents_alphas = []
    completed_sents_preds = []

    step = 1

    hidden_state = generator.init_hidden_state(img_feats)

    while True:
        embedding = generator.embedding(prev_words).squeeze(1)
        context, alpha = generator.attention_net(img_feats, hidden_state)
        gate = generator.sigmoid(generator.f_beta(hidden_state))
        context = gate * context

        input_word = torch.cat((embedding, context), dim=1)
        hidden_state = generator.gru(input_word, hidden_state)

        preds = generator.fc(hidden_state)
        preds = F.log_softmax(preds, dim=1)

        preds = top_preds.expand_as(preds) + preds

        if step == 1:
            top_preds, top_words = preds[0].topk(beam_size, 0, True, True)
        else:
            top_preds, top_words = preds.view(-1).topk(beam_size, 0, True, True)

        prev_word_ids = top_words // vocab_size
        next_word_ids = top_words % vocab_size

        sents = torch.cat([sents[prev_word_ids], next_word_ids.unsqueeze(1)], dim=1)

        alphas = torch.cat([alphas[prev_word_ids], alpha[prev_word_ids].unsqueeze(1)], dim=1)

        incomplete = [idx for idx, next_word in enumerate(next_word_ids) if next_word != word_index['<end>']]
        complete = list(set(range(len(next_word_ids))) - set(incomplete))

        if len(complete) > 0:
            completed_sents.extend(sents[complete].tolist())
            completed_sents_alphas.extend(alphas[complete].tolist())
            completed_sents_preds.extend(top_preds[complete])
        beam_size -= len(complete)

        if beam_size == 0:
            break
        sents = sents[incomplete]
        alphas = alphas[incomplete]
        hidden_state = hidden_state[prev_word_ids[incomplete]]
        img_feats = img_feats[prev_word_ids[incomplete]]
        top_preds = top_preds[incomplete].unsqueeze(1)
        prev_words = next_word_ids[incomplete].unsqueeze(1)

        if step > 50:
            break
        step += 1

    idx = completed_sents_preds.index(max(completed_sents_preds))
    sentence = completed_sents[idx]
    alpha = completed_sents_alphas[idx]

    for w in sentence:
        print(index_word[str(w)])


    #return words, alpha


def greedy_caption_image(encoder, generator, image_path, word_index, index_word):
    vocab_size = len(word_index)

    img = pil_loader(image_path)

    img = data_transforms(img)

    img = torch.FloatTensor(img).to(device)

    img = img.unsqueeze(0)
    img_feats = encoder(img)

    i = 0
    input_word = torch.LongTensor([[word_index['<start>']]]).to(device)

    hidden_state = generator.init_hidden_state(img_feats)

    while i < 20:
        embedding = generator.embedding(input_word).squeeze(1)
        context, alpha = generator.attention_net(img_feats, hidden_state)
        input_w = torch.cat((embedding, context), dim=1)
        hidden_state = generator.gru(input_w, hidden_state)
        preds = generator.fc(hidden_state)
        input_word = torch.topk(preds, 1)[1]
        print(index_word[str(input_word.item())], end=' ')
        i += 1

def main(args):

    with open(args.storage + '/processed_data/' + args.dataset + '/word_index.json') as f:
        word_index = json.load(f)

    with open(args.storage + '/processed_data/' + args.dataset + '/index_word.json') as f:
        index_word = json.load(f)

    generator = Generator(attention_dim=args.attention_dim, gru_units=args.gru_units,
                          embedding_dim=args.embedding_dim, vocab_size=len(word_index))

    encoder = Encoder(args.cnn_architecture)
    encoder.to(device)

    checkpoint = torch.load(args.model_path)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    generator.to(device)

    generator.eval()

    encoder.eval()

    with torch.no_grad():
        caption_image(encoder, generator, args.image_path, word_index, index_word, beam_size=args.beam_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Caption!')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--attention-dim', type=int, default=256)
    parser.add_argument('--gru-units', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--beam-size', type=int, default=20)

    main(parser.parse_args())





















