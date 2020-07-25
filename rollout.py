import torch
import torch.nn as nn
import copy
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rollout(object):

    def __init__(self, generator):
        self.rollout_generator = generator

    def get_reward(self, samples, hidden_states, discriminator, img_feats, word_index, sample_cap_lens, col_shape,
                   args, index_word):
        with torch.no_grad():
            cap_len = torch.max(sample_cap_lens).item()
            if args.rollout_num > 0:
                rewards = torch.zeros(samples.shape[0], col_shape).to(device)
                for i in range(args.rollout_num):
                    for j in range(1, cap_len - 1):
                        incomplete_fake_caps = self.rollout_generator.sample(
                            cap_len=max(cap_len, args.max_len) - (j + 1),
                            col_shape=col_shape,
                            img_feats=img_feats,
                            input_word=samples[:, j],
                            hidden_state=hidden_states[:, j - 1],
                            sampling_method=args.sampling_method)
                        fake_caps = torch.cat([samples[:, :j + 1], incomplete_fake_caps], dim=-1)
                        fake_caps, fake_cap_lens = pad_generated_captions(fake_caps.cpu().numpy(), word_index)
                        fake_caps = torch.from_numpy(fake_caps)
                        fake_cap_lens = torch.LongTensor(fake_cap_lens)
                        fake_caps = fake_caps.to(device)
                        reward = discriminator(img_feats, fake_caps, fake_cap_lens)
                        rewards[:, j - 1] += reward
                rewards = rewards / (1.0 * args.rollout_num)
                reward = discriminator(img_feats, samples, sample_cap_lens)
                rewards[:, cap_len - 2] += reward
                return rewards
            else:
                reward = discriminator(img_feats, samples, sample_cap_lens)
                rewards = reward.unsqueeze(1).repeat(1, col_shape)
                return rewards.to(device)
