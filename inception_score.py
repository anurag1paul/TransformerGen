import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


class InceptionScore:

    def __init__(self, total_batches, batch_size, splits=1):
        self.total_imgs = total_batches * batch_size
        self.batch_size = batch_size
        self.splits = splits

        self.preds = np.zeros((self.total_imgs, 1000))

        # Set up dtype
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor

        # Load inception model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).type(self.dtype)
        self.inception_model.eval()

    def get_pred(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = self.inception_model(x)
        return nn.Softmax(dim=-1)(x).data.cpu().numpy()

    def predict(self, fake_imgs, batch_idx):
        """Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        """
        # Get predictions
        self.preds[batch_idx * self.batch_size: (batch_idx +1) * self.batch_size] = self.get_pred(fake_imgs)

    def get_ic_score(self):

        # Now compute the mean kl-div
        split_scores = []

        for k in range(self.splits):
            part = self.preds[k * (self.total_imgs // self.splits): (k+1) * (self.total_imgs // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)
