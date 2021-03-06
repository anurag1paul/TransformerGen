import os
import time

import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR

from base_model import BaseModel
from data_loader import prepare_data
from inception_score import InceptionScore
from losses import words_loss, discriminator_loss, generator_loss, KL_loss
from networks import RNN_ENCODER, CNN_ENCODER
from attnGan.networks import D_NET64, D_NET128, D_NET256, G_NET
from utils import make_dir, weights_init, EpochTracker, copy_G_params, load_params, build_super_images


class AttnGAN(BaseModel):

    def __init__(self, device, output_dir, opts, ixtoword, train_loader, val_loader):
        super(AttnGAN, self).__init__(device, output_dir, opts)

        cudnn.benchmark = True

        self.batch_size = opts.TRAIN.BATCH_SIZE
        self.max_epoch = opts.TRAIN.MAX_EPOCH
        self.snapshot_interval = opts.TRAIN.SNAPSHOT_INTERVAL
        self.opts = opts

        self.n_words = len(ixtoword)
        self.ixtoword = ixtoword
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_batches = len(self.train_loader)

        self.pretrained_path = os.path.join("pretrained_models/", opts.DAMSM_MODEL)

        epoch_file = os.path.join(self.output_dir, "epoch.txt")
        self.epoch_tracker = EpochTracker(epoch_file)
        self.model_file_name = os.path.join(self.model_dir, "checkpoint_{}.pth.tar")
        self.val_logger = None
        self.losses_logger = None
        self.adam_betas = (self.opts.TRAIN.ADAM_BETA1, self.opts.TRAIN.ADAM_BETA2)
        self.use_lr_scheduler = False

    def build_models(self):
        # ###################encoders######################################## #
        checkpoint = torch.load(self.pretrained_path)

        text_encoder = checkpoint['text_encoder'].to(self.device)
        image_encoder = checkpoint['image_encoder'].to(self.device)
        # clear memory
        del checkpoint

        self.set_requires_grad([text_encoder, image_encoder])
        image_encoder.eval()
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []

        netG = G_NET(self.opts, self.device)
        if self.opts.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64(self.opts).to(self.device))
        if self.opts.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128(self.opts).to(self.device))
        if self.opts.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256(self.opts).to(self.device))

        netG.apply(weights_init)
        netG = netG.to(self.device)
        
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)

        print('# of netsD', len(netsD))

        epoch = 0
        file_name = self.model_file_name.format(self.epoch_tracker.epoch)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)

            netG.load_state_dict(checkpoint['netG'])
            epoch = checkpoint['epoch'] + 1
            for i in range(len(netsD)):
                key = "netsD_{}".format(i)
                netsD[i].load_state_dict(checkpoint[key])

            self.val_logger = open(os.path.join(self.output_dir, 'val_ic_log.txt'), 'a')
            self.losses_logger = open(os.path.join(self.output_dir, 'losses_log.txt'), 'a')
        else:
            self.val_logger = open(os.path.join(self.output_dir, 'val_ic_log.txt'), 'w')
            self.losses_logger = open(os.path.join(self.output_dir, 'losses_log.txt'), 'w')
        
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = torch.optim.Adam(netsD[i].parameters(),
                                   lr=self.opts.TRAIN.DISCRIMINATOR_LR,
                                   betas=self.adam_betas)
            optimizersD.append(opt)

        optimizerG = torch.optim.Adam(netG.parameters(),
                                lr=self.opts.TRAIN.GENERATOR_LR,
                                betas=self.adam_betas)

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).to(self.device)
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).to(self.device)
        match_labels = Variable(torch.LongTensor(range(batch_size))).to(self.device)

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        self.epoch_tracker.write(epoch)
        backup_para = copy_G_params(netG)
        checkpoint = {'epoch':epoch, 'netG':None, 'netsD_0':None, 'netsD_1':None, 'netsD_2':None,}
        load_params(netG, avg_param_G)
        checkpoint['netG'] = netG.state_dict()
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            checkpoint['netsD_{}'.format(i)] = netD.state_dict()
        torch.save(checkpoint, self.model_file_name.format(epoch))
        print('Save G/Ds models.')

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, epoch, step, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img, batch_size=self.batch_size, max_word_num=18)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d_%d.png'\
                    % (self.image_dir, name, epoch, step, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze,
                               None, batch_size=self.batch_size, max_word_num=18)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d_%d.png'\
                % (self.image_dir, name, epoch, step)
            im.save(fullpath)

    def text_encoder_forward(self, text_encoder, captions, captions_mask):
        batch_size = captions.size(0)
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = text_encoder(captions, hidden)
        return words_embs, sent_emb

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = self.opts.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        noise, fixed_noise = noise.to(self.device), fixed_noise.to(self.device)

        gen_iterations = 0

        lr_schedulers = []
        if self.use_lr_scheduler:
            for i in range(len(optimizersD)):
                lr_scheduler = LambdaLR(optimizersD[i], lr_lambda=lambda epoch:0.998**epoch)

                for m in range(start_epoch):
                   lr_scheduler.step()
                lr_schedulers.append(lr_scheduler)

        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.train_loader)
            step = 0

            for i in range(len(lr_schedulers)):
                lr_schedulers[i].step()

            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = next(data_iter)
                imgs, captions, class_ids, captions_mask = prepare_data(data, self.device)

                words_embs, sent_emb = self.text_encoder_forward(text_encoder, captions, captions_mask)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.data.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, class_ids, self.opts)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.data.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 10 == 0:
                    print("Epoch: "+ str(epoch) + " Step: " + str(step) + " " + D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 300 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, epoch, step, name='average')
                    load_params(netG, backup_para)

            is_mean, is_std, error_G_val = self.validate(netG, netsD, text_encoder, image_encoder)
            self.val_logger.write("{} {} {}\n".format(epoch, is_mean, is_std))
            self.val_logger.flush()

            self.losses_logger.write("{} {} {}\n".format(epoch, errG_total.data.item(), error_G_val))
            self.losses_logger.flush()

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data.item(), errG_total.data.item(),
                     end_t - start_t))

            print("IS: {} {}".format(is_mean, is_std))
            if epoch % self.opts.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def validate(self, netG, netsD, text_encoder, image_encoder):
        batch_size = self.batch_size
        nz = self.opts.GAN.Z_DIM
        real_labels, fake_labels, match_labels = self.prepare_labels()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        noise, fixed_noise = noise.to(self.device), fixed_noise.to(self.device)
       
        val_batches = len(self.val_loader)
        netG.eval()
        for i in range(len(netsD)):
            netsD[i].eval()

        inception_scorer = InceptionScore(val_batches, batch_size, val_batches)
        total_loss = []
        with torch.no_grad():
            for step, data in enumerate(self.val_loader):
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                imgs, captions, class_ids, input_mask = prepare_data(data, self.device)

                words_embs, sent_emb = self.text_encoder_forward(text_encoder, captions, input_mask)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)
                errG_total, G_logs = generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, class_ids, self.opts)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                total_loss.append(errG_total.data.item())
                inception_scorer.predict(fake_imgs[-1], step)

        netG.train()
        for i in range(len(netsD)):
            netsD[i].train()
            
        m, s = inception_scorer.get_ic_score()
        return m, s, sum(total_loss) / val_batches

    def test(self, model_path, test_loader):
        batch_size = self.batch_size
        nz = self.opts.GAN.Z_DIM
        real_labels, fake_labels, match_labels = self.prepare_labels()

        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        noise, fixed_noise = noise.to(self.device), fixed_noise.to(self.device)

        text_encoder, netG = self.build_models_for_test(model_path)
        val_batches = len(test_loader)

        inception_scorer = InceptionScore(val_batches, batch_size, val_batches)
        total_loss = []
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                imgs, captions, class_ids, input_mask = prepare_data(data, self.device)

                words_embs, sent_emb = self.text_encoder_forward(text_encoder, captions, input_mask)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)
                inception_scorer.predict(fake_imgs[-1], step)

        m, s = inception_scorer.get_ic_score()
        return m, s, sum(total_loss) / val_batches

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                make_dir(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir, model_path):

        text_encoder, netG = self.build_models_for_test(model_path)

        batch_size = self.batch_size
        nz = self.opts.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
        noise = noise.cuda()

        # the path to save generated images
        save_dir = os.path.join(self.output_dir, "samples")
        make_dir(save_dir)

        cnt = 0

        for _ in range(1):  # (opts.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(self.train_loader, 0):
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                if step > 50:
                    break

                imgs, captions, class_ids, input_mask = prepare_data(data, self.device)

                words_embs, sent_emb = self.text_encoder_forward(text_encoder, captions, input_mask)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                
                for j in range(batch_size):
                    cap = captions[j].data.cpu().numpy()
                    name = "%d_%d" % (step, j)
                    s_tmp = '%s/single/%s' % (save_dir, name)
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        make_dir(folder)

                    sentence = []
                    for m in range(len(cap)):
                        if cap[m] == 0:
                            break
                        word = self.ixtoword[cap[m]].encode('ascii', 'ignore').decode('ascii')
                        sentence.append(word)
                        sentence.append(' ')
                        
                    print(name, ''.join(sentence))

                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_s%d.png' % (s_tmp, k)
                    im.save(fullpath)

    def build_models_for_test(self, model_path):
        # ###################encoders######################################## #
        checkpoint = torch.load(self.pretrained_path)
        text_encoder = checkpoint['text_encoder'].to(self.device)
        # clear memory
        del checkpoint
        self.set_requires_grad([text_encoder])
        text_encoder.eval()
        # #######################generator and discriminators############## #
        netG = G_NET(self.opts, self.device)
        netG.apply(weights_init)
        netG = netG.to(self.device)

        file_name = self.model_file_name.format(self.epoch_tracker.epoch)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name)
            netG.load_state_dict(checkpoint['netG'])

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            netG.load_state_dict(checkpoint['netG'])
        else:
            print("Model Not found")
            exit()
        netG.eval()

        return text_encoder, netG
