import os

import torch

from easydict import EasyDict

from data_loader import CubDataset, FlickrDataset, DefaultCaptionTokenizer, BertCaptionTokenizer
from data_preprocess import CubDataPreprocessor, get_preprocessor
from self_attn.self_attn_gan import SelfAttnGAN, SelfAttnBert
from utils import get_opts

torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CAPTION_SIZE = 30

opts = EasyDict(get_opts("config/bert_attn_flickr.yaml"))

output_dir = os.path.join("checkpoints/", opts.CHECKPOINTS_DIR)
data_dir = 'dataset/'
epoch_file = "epoch.txt"
log_file = "logs.log"

print("Dataset: ", opts.DATASET_NAME, opts.DATA_DIR) 
preprocessor = get_preprocessor(opts.DATASET_NAME, opts.DATA_DIR)

if opts.TEXT.ENCODER == "lstm":
    ixtoword = preprocessor.get_idx_to_word()
    tokenizer = DefaultCaptionTokenizer(preprocessor.get_word_to_idx(), MAX_CAPTION_SIZE)
else:
    tokenizer = BertCaptionTokenizer(MAX_CAPTION_SIZE)
    ixtoword = tokenizer.tokenizer.ids_to_tokens
    
if opts.DATASET_NAME == "cub":
    train_set = CubDataset(preprocessor, opts, tokenizer, mode='train')
    test_set = CubDataset(preprocessor, opts, tokenizer, mode='test')
else:
    train_set = FlickrDataset(preprocessor, opts, tokenizer, mode='train')
    test_set = FlickrDataset(preprocessor, opts, tokenizer, mode='test')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)
self_attn_gan = SelfAttnBert(device, output_dir, opts, ixtoword, train_loader, test_loader)

model_path = "pretrained_models/bert_flickr_190.pth.tar"
print(self_attn_gan.test(model_path, test_loader))

self_attn_gan.sampling(None, model_path)
