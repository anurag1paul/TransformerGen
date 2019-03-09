import os

import torch

from easydict import EasyDict

from data_loader import CubDataset, DefaultCaptionTokenizer, BertCaptionTokenizer
from data_preprocess import CubDataPreprocessor
from self_attn.self_attn_gan import SelfAttnGAN, SelfAttnBert
from utils import get_opts

torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CAPTION_SIZE = 30

opts = EasyDict(get_opts("config/attn_bird.yaml"))

output_dir = os.path.join("checkpoints/", opts.CHECKPOINTS_DIR)
data_dir = 'dataset/'
epoch_file = "epoch.txt"
log_file = "logs.log"

preprocessor = CubDataPreprocessor("cub", data_dir)

if opts.TEXT.ENCODER == "lstm":
    ixtoword = preprocessor.get_idx_to_word()
    tokenizer = DefaultCaptionTokenizer(preprocessor.get_word_to_idx(), MAX_CAPTION_SIZE)
else:
    tokenizer = BertCaptionTokenizer(MAX_CAPTION_SIZE)
    ixtoword = tokenizer.tokenizer.ids_to_tokens

train_set = CubDataset(preprocessor, opts, tokenizer, mode='train')
val_set = CubDataset(preprocessor, opts, tokenizer, mode='val')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)

if opts.TEXT.ENCODER == "lstm":
    self_attn_gan = SelfAttnGAN(device, output_dir, opts, ixtoword, train_loader, val_loader)
else:
    self_attn_gan = SelfAttnBert(device, output_dir, opts, ixtoword, train_loader, val_loader)

self_attn_gan.train()
