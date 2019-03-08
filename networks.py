import torch
from pytorch_pretrained_bert import BertModel

from torch import nn
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

#  taken from https://github.com/taoxugit/AttnGAN/


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=False)


class RNN_ENCODER(nn.Module):

    def __init__(self, ntoken, batch_size, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1):
        super(RNN_ENCODER, self).__init__()
        
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = True

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.nhidden = nhidden // self.num_directions
        
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        
        self.rnn = nn.LSTM(self.ninput, self.nhidden,
                           self.nlayers, batch_first=True,
                           bidirectional=self.bidirectional)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.nlayers* self.num_directions, batch_size, self.nhidden).cuda(),
                torch.zeros(self.nlayers* self.num_directions, batch_size, self.nhidden).cuda())

    def forward(self, captions, hidden):
    
        emb = self.drop(self.encoder(captions))
        output, hidden = self.rnn(emb, hidden)
        words_emb = output.transpose(1, 2)
        
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        
        return words_emb, sent_emb


class BERT_ENCODER(nn.Module):

    def __init__(self, nhidden):
        super(BERT_ENCODER, self).__init__()

        self.enc_size = nhidden  # size of each encoding vector
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.word_enc = nn.Sequential([nn.Linear(768*4, 768*2),
                                       nn.Linear(768*4, 768),
                                       nn.Linear(768, self.enc_size)])
        self.sent_enc = nn.Linear(768, self.enc_size)

    def forward(self, captions, input_mask):
        batch_size, seq_len = captions.size(0), captions.size(1)
        all_encoder_layers, pooled_output = self.bert(captions, token_type_ids=None, attention_mask=input_mask)

        words_features = torch.cat(all_encoder_layers[-5:-1], dim=-1).view(-1, 768*4)
        words_emb = self.word_enc(words_features).view(batch_size, seq_len, -1)
        words_emb = words_emb.transpose(1, 2)
        sent_emb = self.sent_enc(pooled_output)

        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        self.nef = nef  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299

        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code
