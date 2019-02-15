import os
import yaml
import pickle
import glob
import torch
import numpy as np
from torchvision.utils import save_image
from matplotlib import pyplot as plt


def weights_init_normal(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_opts():
    with open("params.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
        return data_loaded


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def store_loss_plots(train_losses, val_losses, opts):
    """Saves a plot of the training and validation loss curves.
    """
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title('{}'.format(opts.model), fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()


def sample_images(data, batches_done, generator, number):
    # to define
    # x, y = next(data.data_generator())
    # real_A = Variable(x.type(Tensor))
    # real_B = Variable(y.type(Tensor))
    # fake_B = generator(real_A)
    # img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    # save_image(img_sample, 'saved_images/%s.png' % (number), nrow=5, normalize=True)
    # return x, y
    pass


class EpochTracker():
    def __init__(self, in_file, log_file):
        self.epoch = 0
        self.iter = 0
        self.in_file = in_file
        self.file_exists = os.path.isfile(in_file)
        self. loss_log = open(log_file, 'w')
        if self.file_exists:
            with open(in_file, 'r') as f:
                d = f.read()
                a, b = d.split(";")
                self.epoch = int(a)
                self.iter = int(b)

    def write(self, epoch, iteration):
        self.epoch = epoch
        self.iter = iteration
        data = "{};{}".format(self.epoch, self.iter)
        with open(self.in_file, 'w') as f:
            f.write(data)

    def log_losses(self, epoch, train_loss, val_loss):
        self.loss_log.write('{} {} {}\n'.format(epoch, train_loss, val_loss))
        self.loss_log.flush()

class EarlyStopping:
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
