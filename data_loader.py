import os

import numpy
import numpy as np
import torch

import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from nltk import RegexpTokenizer
from torch.utils.data import Dataset

from data_preprocess import DataPreprocessor


class CubDataset(Dataset):

    def __init__(self, device, preprocessor: DataPreprocessor, mode="train"):
        super(CubDataset, self).__init__()
        self.device = device
        self.preprocessor = preprocessor
        self.mode = mode
        self.max_caption_size = 50
        self.word_to_idx = self.preprocessor.get_word_to_idx()

        if mode == "train":
            self.img_file_names = self.preprocessor.get_train_files()
        elif mode == "val":
            self.img_file_names = self.preprocessor.get_val_files()
        else:
            self.img_file_names = self.preprocessor.get_test_files()

        self.img_captions = []
        self.class_ids = []

        for name in self.img_file_names:
            name_parts = name.split('.')
            self.class_ids.append(int(name_parts[0]))
            txt_name = '.'.join(name_parts[0:-1]) + '.txt'
            txt_path = os.path.join(self.preprocessor.captions_path, txt_name)
            with open(txt_path,  encoding='utf-8') as captions_file:
                captions = self.tokenize(captions_file.read().splitlines())
                padded = self.padding(captions)
                self.img_captions.append(padded)
        
        self.data_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize([256, 256]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.img_file_names)

    def load_img(self, image_name):
        image = Image.open(self.preprocessor.data_path + image_name)
        image = self.data_transforms(image).float()
        image = torch.autograd.Variable(image, requires_grad=False)
        return image.to(self.device)
    
    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().detach().numpy()
        plt.figure(figsize = (5,5))
        plt.imshow(np.transpose(npimg, (1, 2, 0)), aspect='auto')
        
    def padding(self, unpadded):
        lens = np.array([len(item) for item in unpadded])
        mask = lens[:,None] > np.arange(self.max_caption_size)
        out = np.full(mask.shape,0)
        for i, e in enumerate(unpadded):
            if len(e) > 30:
                unpadded[i] = unpadded[i][:self.max_caption_size]
        out[mask] = np.concatenate(unpadded)
        return torch.Tensor(out)

    def tokenize(self, captions):
        all_cap_tokens = []
        for cap in captions:
            if len(cap) == 0:
                continue
            cap = cap.replace(u"\ufffd\ufffd", u" ")
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(cap.lower())

            if len(tokens) == 0:
                continue

            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    if t in self.word_to_idx:
                        tokens_new.append(self.word_to_idx[t])
            all_cap_tokens.append(tokens_new)
        return all_cap_tokens

    def __getitem__(self, idx):
        image = self.load_img(self.img_file_names[idx])
        # select a random sentence
        cap_idx = np.random.choice(np.arange(len(self.img_captions[idx])))
        caption = self.img_captions[idx][cap_idx].to(self.device)

        class_id = torch.Tensor(torch.from_numpy(numpy.array(self.class_ids[idx]))).to(self.device)

        return image, caption, class_id
