import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from data_loader import CubDataset
from data_preprocess import DataPreprocessor
from losses import words_loss, sent_loss
from networks import RNN_ENCODER, CNN_ENCODER
from utils import get_opts, build_super_images, make_dir

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

captions_path = '../cub_dataset/text_c10'
img_name_path = '../cub_dataset/images.txt'
data_path = '../cub_dataset/images/'
output_directory = "checkpoints/"
epoch_file = "epoch.txt"
log_file = "logs.log"

UPDATE_INTERVAL = 200
opts = get_opts("config/damsm_bird.yaml")


def create_loader(opts):
    preprocessor = DataPreprocessor("cub", img_name_path, data_path, captions_path)
    ixtoword = preprocessor.get_idx_to_word()
    train_set = CubDataset(device, preprocessor, mode='train')
    val_set = CubDataset(device, preprocessor, mode='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def train(dataloader, cnn_model, rnn_model, batch_size, labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()

    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0

    count = (epoch + 1) * len(dataloader)
    start_time = time.time()

    for step, data in enumerate(dataloader, 0):
        print('step', step)
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, class_ids = data

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17

        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        loss.backward()

        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(), opts.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0[0] / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1[0] / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0[0] / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1[0] / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            img_set, _ = build_super_images(imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                im.save(fullpath)
    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()

    s_total_loss = 0
    w_total_loss = 0

    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, class_ids = data

        words_features, sent_code = cnn_model(real_imgs[-1])

        words_emb, sent_emb = rnn_model(captions)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss[0] / step
    w_cur_loss = w_total_loss[0] / step

    return s_cur_loss, w_cur_loss


def build_models(dict_size):
    # build model ############################################################
    text_encoder = RNN_ENCODER(dict_size, nhidden=opts.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(opts.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0

    # if cfg.TRAIN.NET_E != '':
    #     state_dict = torch.load(cfg.TRAIN.NET_E)
    #     text_encoder.load_state_dict(state_dict)
    #     print('Load ', cfg.TRAIN.NET_E)
    #     #
    #     name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
    #     state_dict = torch.load(name)
    #     image_encoder.load_state_dict(state_dict)
    #     print('Load ', name)
    #
    #     istart = cfg.TRAIN.NET_E.rfind('_') + 8
    #     iend = cfg.TRAIN.NET_E.rfind('.')
    #     start_epoch = cfg.TRAIN.NET_E[istart:iend]
    #     start_epoch = int(start_epoch) + 1
    #     print('start_epoch', start_epoch)


    text_encoder = text_encoder.to(device)
    image_encoder = image_encoder.to(device)
    labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":

    manualSeed = 100
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s' % \
        (opts.DATASET_NAME, opts.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    make_dir(model_dir)
    make_dir(image_dir)

    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = opts.TREE.BASE_SIZE * (2 ** (opts.TREE.BRANCH_NUM-1))
    batch_size = opts.TRAIN.BATCH_SIZE

    train_loader, val_loader, ixtoword = create_loader(opts)

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()

    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = opts.TRAIN.ENCODER_LR
        optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))

        for epoch in range(start_epoch, opts.TRAIN.MAX_EPOCH):
            epoch_start_time = time.time()
            count = train(train_loader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          ixtoword, image_dir)
            print('-' * 89)
            if len(val_loader) > 0:
                s_loss, w_loss = evaluate(val_loader, image_encoder,
                                          text_encoder, batch_size)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > opts.TRAIN.ENCODER_LR/10.:
                optimizer.param_groups[0]['lr'] *= 0.98

            if (epoch % opts.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == opts.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
