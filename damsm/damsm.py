import os
import time

import torch
from PIL import Image
from torch import optim
from torch.autograd import Variable

from data_loader import prepare_data
from losses import words_loss, sent_loss
from networks import RNN_ENCODER, CNN_ENCODER
from utils import build_super_images


class Damsm:

    def __init__(self, opts, update_interval, device):
        self.opts = opts
        self.update_interval = update_interval
        self. device = device

    def text_enc_forward(self, text_encoder, captions, input_mask):
        batch_size = captions.size(0)
        hidden = text_encoder.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = text_encoder(captions, hidden)
        return words_emb, sent_emb

    def train(self, dataloader, image_encoder, text_encoder, optimizer, epoch, ixtoword, image_dir, batch_size):

        image_encoder.train()
        text_encoder.train()

        s_total_loss0 = 0
        s_total_loss1 = 0
        w_total_loss0 = 0
        w_total_loss1 = 0

        count = (epoch + 1) * len(dataloader)
        start_time = time.time()

        s_epoch_loss = 0
        w_epoch_loss = 0
        num_batches = len(dataloader)

        for step, data in enumerate(dataloader, 0):
            print('step', step)
            optimizer.zero_grad()

            imgs, captions, class_ids, input_mask = prepare_data(data, self.device)

            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef

            words_features, sent_code = image_encoder(imgs[-1])
            # --> batch_size x nef x 17*17
            
            batch_size, nef, att_size, _ = words_features.shape
            # words_features = words_features.view(batch_size, nef, -1)

            words_emb, sent_emb = self.text_enc_forward(text_encoder, captions, input_mask)
            labels = Variable(torch.LongTensor(range(batch_size))).to(self.device)

            w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, class_ids, batch_size)
            w_total_loss0 += w_loss0.data.item()
            w_total_loss1 += w_loss1.data.item()
            loss = w_loss0 + w_loss1
            w_epoch_loss += w_loss0.item() + w_loss1.item()

            s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            loss += s_loss0 + s_loss1
            s_total_loss0 += s_loss0.data.item()
            s_total_loss1 += s_loss1.data.item()
            s_epoch_loss += s_loss0.item() + s_loss1.item()

            loss.backward(retain_graph=True)

            # `clip_grad_norm` helps prevent
            # the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), self.opts.TRAIN.RNN_GRAD_CLIP)
            optimizer.step()

            if step != 0 and step % self.update_interval == 0:
                count = epoch * len(dataloader) + step

                s_cur_loss0 = s_total_loss0 / self.update_interval
                s_cur_loss1 = s_total_loss1 / self.update_interval

                w_cur_loss0 = w_total_loss0 / self.update_interval
                w_cur_loss1 = w_total_loss1 / self.update_interval

                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                      's_loss {:5.2f} {:5.2f} | '
                      'w_loss {:5.2f} {:5.2f}'
                      .format(epoch, step, len(dataloader),
                              elapsed * 1000. / self.update_interval,
                              s_cur_loss0, s_cur_loss1,
                              w_cur_loss0, w_cur_loss1))
                s_total_loss0 = 0
                s_total_loss1 = 0
                w_total_loss0 = 0
                w_total_loss1 = 0
                start_time = time.time()

            if step == num_batches-1:
                # attention Maps
                img_set, _ = build_super_images(imgs[-1].cpu(), captions, ixtoword,
                                                attn_maps, att_size, None, batch_size, max_word_num=18)
                if img_set is not None:
                    im = Image.fromarray(img_set)
                    fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                    im.save(fullpath)
        s_epoch_loss /= len(dataloader)
        w_epoch_loss /= len(dataloader)
        return count, s_epoch_loss, w_epoch_loss

    def evaluate(self, dataloader, image_encoder, text_encoder):
        image_encoder.eval()
        text_encoder.eval()

        s_total_loss = 0
        w_total_loss = 0

        for step, data in enumerate(dataloader, 0):
            real_imgs, captions, class_ids, input_mask = prepare_data(data, self.device)

            words_features, sent_code = image_encoder(real_imgs[-1])

            batch_size = words_features.size(0)
            words_emb, sent_emb = self.text_enc_forward(text_encoder, captions, input_mask)
            labels = Variable(torch.LongTensor(range(batch_size))).to(self.device)

            w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels, class_ids, batch_size)
            w_total_loss += (w_loss0 + w_loss1).data

            s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            s_total_loss += (s_loss0 + s_loss1).data

        s_cur_loss = s_total_loss.item() / step
        w_cur_loss = w_total_loss.item() / step

        return s_cur_loss, w_cur_loss

    def build_models(self, dict_size, batch_size, model_file_name):
        # build model ############################################################
        text_encoder = RNN_ENCODER(dict_size, batch_size=batch_size, nhidden=self.opts.TEXT.EMBEDDING_DIM)
        image_encoder = CNN_ENCODER(self.opts.TEXT.EMBEDDING_DIM)

        text_encoder, image_encoder, start_epoch, optimizer = self.load_saved_model(text_encoder, image_encoder,
                                                                                    model_file_name)
        text_encoder = text_encoder.to(self.device)
        image_encoder = image_encoder.to(self.device)

        return text_encoder, image_encoder, start_epoch, optimizer

    def load_saved_model(self, text_encoder, image_encoder, model_file_name):

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
            lr = self.opts.TRAIN.ENCODER_LR
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))

        return text_encoder, image_encoder, start_epoch, optimizer
