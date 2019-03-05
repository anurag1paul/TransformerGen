import torch

from easydict import EasyDict

from data_loader import CubDataset
from data_preprocess import DataPreprocessor
from self_attn.self_attn_gan import SelfAttnGAN
from utils import get_opts

torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "checkpoints/selfAttnGAN"
data_dir = 'dataset/'
epoch_file = "epoch.txt"
log_file = "logs.log"

opts = EasyDict(get_opts("config/attn_bird.yaml"))

preprocessor = DataPreprocessor("cub", data_dir)
ixtoword = preprocessor.get_idx_to_word()
train_set = CubDataset(preprocessor, opts, mode='train')
val_set = CubDataset(preprocessor, opts, mode='val')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)

self_attn_gan = SelfAttnGAN(device, output_dir, opts, ixtoword, train_loader, val_loader)
self_attn_gan.train()