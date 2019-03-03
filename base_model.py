from abc import ABC, abstractmethod

import torch

from utils import EpochTracker, weights_init_normal


class BaseModel(ABC):

    def __init__(self, device, nets, opts):
        self.device = device
        self.nets = nets
        self.opts = opts
        self.folder = opts.checkpoints_dir
        self.epoch_tracker = EpochTracker(self.folder + "/epoch.txt")

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def test(self):
        with torch.no_grad():
            self.forward()
            self.evaluate()

    def save_progress(self, epoch, iteration, save_epoch=False):
        path = self.folder

        for net, file in self.nets.items():
            if save_epoch:
                file = "{}/{}_{}".format(path, file, epoch)
            if torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), file)
                net.to(self.device)
            else:
                torch.save(net.cpu().state_dict(), file)

        self.epoch_tracker.write(epoch, iteration)

    @abstractmethod
    def save_image(self, path, name):
        pass

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def init_net(net, file=None):
        gpu_ids = list(range(torch.cuda.device_count()))

        if file is not None:
            net.load_state_dict(torch.load(file))
        else:
            net.apply(weights_init_normal)

        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)

        return net
