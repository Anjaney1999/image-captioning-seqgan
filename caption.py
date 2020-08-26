import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from models import *
import argparse
import json
from PIL import Image
import torchvision.transforms as transforms
from utils import pil_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def caption_image(encoder, generator, img, word_index, index_word, beam_size):

    vocab_size = len(word_index)

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
    alphas = completed_sents_alphas[idx]

    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(index_word[str(word_idx)])
        if word_idx == word_index['<end>']:
            break
    return sentence_tokens, alphas


def generate_visualization(encoder, generator, img_path, word_index, index_word, beam_size, smooth=True):
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(0)
    sentence_tokens, alphas = caption_image(encoder=encoder, generator=generator, img=img, word_index=word_index,
                                            index_word=index_word, beam_size=beam_size)
    img = Image.open(img_path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) / 2
    top = (h - 224) / 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
    img = img.astype('float32') / 255

    num_words = len(sentence_tokens)
    w = np.round(np.sqrt(num_words))
    h = np.ceil(np.float32(num_words) / w)
    alphas = torch.tensor(alphas)

    plot_height = ceil((num_words + 3) / 4.0)
    ax1 = plt.subplot(4, plot_height, 1)
    plt.imshow(img)
    plt.axis('off')
    for idx in range(num_words):
        ax2 = plt.subplot(4, plot_height, idx + 2)
        label = sentence_tokens[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, label, color='black', fontsize=13)
        plt.imshow(img)
        shape_size = 7
        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alphas[idx, :].reshape(shape_size, shape_size), upscale=28,
                                                         sigma=8)
        else:
            alpha_img = skimage.transform.resize(alphas[idx, :].reshape(shape_size, shape_size),
                                                 [img.shape[0], img.shape[1]])
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


def main(args):

    with open(args.storage + '/processed_data/' + args.dataset + '/word_index.json') as f:
        word_index = json.load(f)

    with open(args.storage + '/processed_data/' + args.dataset + '/index_word.json') as f:
        index_word = json.load(f)

    generator = Generator(attention_dim=args.attention_dim, gru_units=args.gru_units,
                          embedding_dim=args.embedding_dim, vocab_size=len(word_index))

    encoder = Encoder(args.cnn_architecture)
    encoder.to(device)
    checkpoint = torch.load(args.storage + '/ckpts/' + args.dataset + '/gen/' + args.model_ckpt)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    generator.to(device)

    generator.eval()
    encoder.eval()

    generate_visualization(encoder, generator, args.img_path, word_index, index_word, args.beam_size, smooth=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Caption!')
    parser.add_argument('--model-ckpt', type=str)
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--attention-dim', type=int, default=512)
    parser.add_argument('--gru-units', type=int, default=512)
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--cnn-architecture', type=str, default='resnet152')
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--beam-size', type=int, default=20)

    main(parser.parse_args())





















