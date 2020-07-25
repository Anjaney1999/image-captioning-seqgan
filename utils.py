import numpy as np
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def categorical_accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def binary_accuracy(output, targets):
    acc = (output.round() == targets).float().sum() / (1.0 * targets.shape[0])
    return acc * 100.0


def encode_captions(captions, word_index, max_len):
    caps = []
    cap_lens = []
    for tokens in captions:
        idx = [word_index[token] if token in word_index else word_index['<unk>'] for token in tokens]
        caps.append(
            [word_index['<start>']] + idx + [word_index['<end>']] +
            [word_index['<pad>']] * max(max_len - len(tokens), 0)
        )
        cap_lens.append(len(tokens) + 2)

    return caps, cap_lens


def pad_generated_captions(caps, word_index):
    true_lengths = []
    for i in range(caps.shape[0]):
        length_found = False
        for j in range(caps.shape[1]):
            if caps[i][j] == word_index['<end>'] or caps[i][j] == word_index['<pad>']:
                true_lengths.append(j + 1)
                np.put(caps[i], np.arange(j + 1, caps.shape[1]),
                       np.zeros(caps.shape[1] - (j + 1)))
                length_found = True
                break
        if not length_found:
            true_lengths.append(caps.shape[1])

    return caps, true_lengths


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

