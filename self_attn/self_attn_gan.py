import os

import torch

from attnGan.attn_gan import AttnGAN
from self_attn.networks import G_NET, D_NET64, D_NET128, D_NET256
from utils import weights_init


class SelfAttnGAN(AttnGAN):

    def __init__(self, device, output_dir, opts, ixtoword, train_loader, val_loader):
        super(SelfAttnGAN, self).__init__(device, output_dir, opts, ixtoword, train_loader, val_loader)


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
        else:
            self.val_logger = open(os.path.join(self.output_dir, 'val_ic_log.txt'), 'w')

        return [text_encoder, image_encoder, netG, netsD, epoch]
