import os
import sys
import time
import random
import datetime
import dateutil.tz
import numpy as np
from PIL import Image

from easydict import EasyDict
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data_loader import CubDataset, prepare_data
from data_preprocess import DataPreprocessor
from losses import words_loss, sent_loss
from networks import RNN_ENCODER, CNN_ENCODER
from utils import get_opts, build_super_images, make_dir, save_checkpoint, EpochTracker

# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
# sys.path.append(dir_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = 'dataset/'
output_directory = "checkpoints"
epoch_file = "epoch.txt"
log_file = "logs.log"

UPDATE_INTERVAL = 5
opts = EasyDict(get_opts("config/damsm_bird.yaml"))


def create_loader(opts):
    preprocessor = DataPreprocessor("cub", data_dir)
    ixtoword = preprocessor.get_idx_to_word()
    train_set = CubDataset(preprocessor, opts, mode='train')
    val_set = CubDataset(preprocessor, opts, mode='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.TRAIN.BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)
    return train_loader, val_loader, ixtoword


def train(dataloader, cnn_model, rnn_model, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()

    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0

    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    
    s_epoch_loss = 0
    w_epoch_loss = 0

    for step, data in enumerate(dataloader, 0):
        print('step', step)
        optimizer.zero_grad()

        imgs, captions, class_ids = prepare_data(data, device)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        
        words_features, sent_code = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17

        batch_size, nef, att_sze = words_features.size(0), words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)
        
        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, hidden)
        labels = Variable(torch.LongTensor(range(batch_size)))

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1
        w_epoch_loss += w_loss0.item() + w_loss1.item()

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        s_epoch_loss += s_loss0.item() + s_loss1.item()

        loss.backward(retain_graph=True)

        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), opts.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step != 0 and step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

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
            
        if step == batch_size-1:
            # attention Maps
            img_set, _ = build_super_images(imgs[-1].cpu(), captions, ixtoword,
                                            attn_maps, att_sze, None, batch_size, max_word_num=18)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                im.save(fullpath)
    s_epoch_loss /= len(dataloader)
    w_epoch_loss /= len(dataloader)
    return count, s_epoch_loss, w_epoch_loss


def evaluate(dataloader, cnn_model, rnn_model):
    cnn_model.eval()
    rnn_model.eval()

    s_total_loss = 0
    w_total_loss = 0

    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, class_ids = data
        real_imgs = real_imgs.to(device) 
        captions = captions.to(device)
        class_ids = class_ids.numpy()
        
        words_features, sent_code = cnn_model(real_imgs[-1])
        
        batch_size = words_features.size(0)
        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, hidden)
        labels = Variable(torch.LongTensor(range(batch_size)))

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss


def build_models(dict_size, batch_size, model_file_name):
    # build model ############################################################
    text_encoder = RNN_ENCODER(dict_size, batch_size=batch_size, nhidden=opts.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(opts.TEXT.EMBEDDING_DIM)

    if os.path.exists(model_file_name):
        checkpoint = torch.load(model_file_name)

        start_epoch = checkpoint['epoch'] + 1
        optimizer = checkpoint['optimizer']

        text_encoder = checkpoint['text_encoder']
        image_encoder = checkpoint['image_encoder'] 

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        # clear memory
        del checkpoint
        # del model_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        start_epoch = 0
        
        para = list(text_encoder.parameters())
        for v in image_encoder.parameters():
            if v.requires_grad:
                para.append(v)
        # different modules have different learning rate
        lr = opts.TRAIN.ENCODER_LR
        optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))


    text_encoder = text_encoder.to(device)
    image_encoder = image_encoder.to(device)

    return text_encoder, image_encoder, start_epoch, optimizer


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
    
    output_dir = '%s/%s' % \
        (output_directory, opts.DATASET_NAME)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    make_dir(model_dir)
    make_dir(image_dir)
    
    epoch_file = os.path.join(output_dir, "output.txt")
    epoch_tracker = EpochTracker(epoch_file)
    loss_log = open(os.path.join(model_dir, 'loss_log.txt'), 'a+')
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = opts.TREE.BASE_SIZE * (2 ** (opts.TREE.BRANCH_NUM-1))
    batch_size = opts.TRAIN.BATCH_SIZE

    train_loader, val_loader, ixtoword = create_loader(opts)

    # Train ##############################################################
    model_file_name = os.path.join(model_dir, 'checkpoint-' + str(epoch_tracker.epoch) + '.pth.tar')
    text_encoder, image_encoder, start_epoch, optimizer = build_models(len(ixtoword), batch_size, model_file_name)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, opts.TRAIN.MAX_EPOCH):
            epoch_start_time = time.time()
            count, s_train_loss, w_train_loss = train(train_loader, image_encoder, text_encoder, optimizer, epoch,
                          ixtoword, image_dir)
            print('-' * 89)
            if len(val_loader) > 0:
                s_loss, w_loss = evaluate(val_loader, image_encoder,
                                          text_encoder, batch_size)
                print('| end epoch {:3d} | val loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            
            loss_log.write('e:{} st:{} wt:{} sv:{} wv:{}\n'.format(epoch, s_train_loss, 
                                                                   w_train_loss, s_loss, w_loss))
            loss_log.flush()
            
            if lr > opts.TRAIN.ENCODER_LR/10.:
                optimizer.param_groups[0]['lr'] *= 0.98

            if (epoch % opts.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == opts.TRAIN.MAX_EPOCH):
                save_checkpoint({
                    'opts': opts,
                    'epoch': epoch,
                    'text_encoder': text_encoder,
                    'image_encoder': image_encoder,
                    'optimizer': optimizer
                }, epoch, model_dir)
                print('Saved models.')
            epoch_tracker.write(epoch)
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
