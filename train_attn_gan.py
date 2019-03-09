import torch

from easydict import EasyDict

from attnGan.attn_gan import AttnGAN
from data_loader import CubDataset, DefaultCaptionTokenizer
from data_preprocess import CubDataPreprocessor
from utils import get_opts

device = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "checkpoints/attnGAN"
data_dir = 'dataset/'
epoch_file = "epoch.txt"
log_file = "logs.log"

opts = EasyDict(get_opts("config/bird.yaml"))
MAX_CAPTION_SIZE = 30

preprocessor = CubDataPreprocessor("cub", data_dir)
ixtoword = preprocessor.get_idx_to_word()
tokenizer = DefaultCaptionTokenizer(preprocessor.get_word_to_idx(), MAX_CAPTION_SIZE)
train_set = CubDataset(preprocessor, opts, tokenizer, mode='train')
val_set = CubDataset(preprocessor, opts, tokenizer, mode='val')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)

attn_gan = AttnGAN(device, output_dir, opts, ixtoword, train_loader, val_loader)
attn_gan.train()
