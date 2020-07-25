import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152, resnet101
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, architecture='resnet152'):
        super(Encoder, self).__init__()
        self.architecture = architecture
        if architecture == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        else:
            self.net = resnet101(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        
        self.fine_tune()
    
    def forward(self, img):
        feats = self.net(img)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.view(feats.size(0), -1, feats.size(-1))
        return feats
    
    def fine_tune(self, fine_tune=False):

        if not fine_tune:
            for param in self.net.parameters():
                param.requires_grad = False


class Attention(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(encoder_dim, attention_dim)
        self.W2 = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, img_feats, hidden):
        x = self.W1(img_feats)
        y = self.W2(hidden)
        x = self.V(self.tanh(x + y.unsqueeze(1))).squeeze(2)
        alphas = self.softmax(x)
        weighted_feats = (img_feats * alphas.unsqueeze(2)).sum(dim=1)
        
        return weighted_feats, alphas


class Generator(nn.Module):
    
    def __init__(self,
                 attention_dim, 
                 embedding_dim, 
                 gru_units, 
                 vocab_size, 
                 encoder_dim=2048, 
                 dropout=0.5
    ):
        super(Generator, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention_net = Attention(encoder_dim, gru_units, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.gru = nn.GRUCell(embedding_dim + encoder_dim, gru_units, bias=True)
        self.init_h = nn.Linear(encoder_dim, gru_units)
        self.f_beta = nn.Linear(gru_units, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(gru_units, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    
    def init_hidden_state(self, img_feats):
        mean_img_feats = img_feats.mean(dim=1)
        hidden = self.init_h(mean_img_feats)
        hidden = self.relu(hidden)
        return hidden
    
    def forward(self, img_feats, caps, cap_lens):
        
        batch_size = img_feats.size(0)

        vocab_size = self.vocab_size

        num_pixels = img_feats.size(1)
        
        cap_lens, indices = cap_lens.sort(dim=0, descending=True)
        img_feats = img_feats[indices]
        caps = caps[indices]
        
        embeddings = self.embedding(caps)
        
        hidden_state = self.init_hidden_state(img_feats)

        output_lens = (cap_lens - 1).tolist()
        
        preds = torch.zeros(batch_size, caps.shape[1] - 1, vocab_size).to(device)
        alphas = torch.zeros(batch_size, caps.shape[1] - 1, num_pixels).to(device)
        
        for t in range(max(output_lens)):
            context_vec, alpha = self.attention_net(img_feats, hidden_state)
            gate = self.sigmoid(self.f_beta(hidden_state))
            context_vec = gate * context_vec
            hidden_state = self.gru(torch.cat([embeddings[:, t],
                                               context_vec], dim=1), hidden_state)
            
            preds[:, t] = self.fc(self.dropout(hidden_state))

            alphas[:, t] = alpha
        
        return preds, caps, output_lens, alphas, indices

    def step(self, input_word, hidden_state, img_feats):
        embeddings = self.embedding(input_word)
        context_vec, alpha = self.attention_net(img_feats, hidden_state)
        gate = self.sigmoid(self.f_beta(hidden_state))
        context_vec = gate * context_vec
        hidden_state = self.gru(torch.cat([embeddings, context_vec], dim=1), hidden_state)
        preds = self.softmax(self.fc(hidden_state))

        return preds, hidden_state

    def sample(self, cap_len, col_shape, img_feats, input_word, sampling_method='multinomial', hidden_state=None):

        samples = torch.zeros(input_word.shape[0], col_shape).long().to(device)
        if hidden_state is None:
            hidden_states = torch.zeros(input_word.shape[0], col_shape, self.gru_units).to(device)
            hidden_state = self.init_hidden_state(img_feats)
            samples[:, 0] = input_word
            for i in range(cap_len):
                preds, hidden_state = self.step(input_word, hidden_state, img_feats)

                if sampling_method == 'multinomial':
                    input_word = torch.multinomial(preds, 1)
                    input_word = input_word.squeeze(-1)
                else:
                    input_word = torch.argmax(preds, 1)
                samples[:, i + 1] = input_word
                hidden_states[:, i] = hidden_state

            return samples, hidden_states

        else:
            for i in range(cap_len):
                preds, hidden_state = self.step(input_word, hidden_state, img_feats)
                if sampling_method == 'multinomial':
                    input_word = torch.multinomial(preds, 1)
                    input_word = input_word.squeeze(-1)
                else:
                    input_word = torch.argmax(preds, 1)
                samples[:, i] = input_word

            return samples


class GRUDiscriminator(nn.Module):

    def __init__(self, embedding_dim, encoder_dim, gru_units, vocab_size):

        super(GRUDiscriminator, self).__init__()

        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_units, batch_first=True)

        self.fc1 = nn.Linear(encoder_dim, embedding_dim)
        self.fc2 = nn.Linear(gru_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feats, caps, cap_lens):
        img_feats = img_feats.permute(0, 2, 1)
        img_feats = F.avg_pool1d(img_feats, img_feats.shape[-1]).squeeze(-1)
        img_feats = self.fc1(img_feats)
        embeddings = self.embedding(caps)
        inputs = torch.cat((img_feats.unsqueeze(1), embeddings), 1)
        inputs_packed = pack_padded_sequence(inputs, cap_lens + 1, batch_first=True, enforce_sorted=False)
        outputs, _ = self.gru(inputs_packed)
        try:
            outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        except:
            print(outputs)
            print(outputs.shape)
        row_indices = torch.arange(0, caps.size(0)).long()
        last_hidden = outputs[row_indices, cap_lens, :]
        pred = self.sigmoid(self.fc2(last_hidden))
        return pred.squeeze(-1)
