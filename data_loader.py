import os

import numpy
import numpy as np
import torch
import pandas as pd

from PIL import Image
from matplotlib import pyplot as plt
from nltk import RegexpTokenizer
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms

from data_preprocess import DataPreprocessor


def prepare_data(data, device):
    imgs, captions, class_ids = data

    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(Variable(imgs[i]).to(device))

    captions = captions.squeeze()
    class_ids = class_ids.numpy()

    captions = Variable(captions).to(device)

    return [real_imgs, captions, class_ids]


def get_imgs(img_path, imsize, opts, bbox=None,
             transform=None, normalize=None):

    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []

    for i in range(opts.TREE.BRANCH_NUM):
        # print(imsize[i])
        if i < (opts.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret


class CubDataset(Dataset):

    def __init__(self, preprocessor: DataPreprocessor, opts, mode="train"):
        super(CubDataset, self).__init__()
        self.preprocessor = preprocessor
        self.mode = mode
        self.max_caption_size = 30
        self.opts = opts
        self.word_to_idx = self.preprocessor.get_word_to_idx()

        if mode == "train":
            self.img_file_names = self.preprocessor.get_train_files()
        elif mode == "val":
            self.img_file_names = self.preprocessor.get_val_files()
        else:
            self.img_file_names = self.preprocessor.get_test_files()

        self.img_captions = []
        self.filename_bbox, self.filename_class = self.load_data()

        for name in self.img_file_names:
            name_parts = name.split('.')
            txt_name = '.'.join(name_parts[0:-1]) + '.txt'
            txt_path = os.path.join(self.preprocessor.captions_path, txt_name)
            with open(txt_path,  encoding='utf-8') as captions_file:
                captions = self.tokenize(captions_file.read().splitlines())
                padded = self.padding(captions)
                self.img_captions.append(padded)

        self.imsize = []
        base_size = opts.TREE.BASE_SIZE
        for i in range(opts.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

    def __len__(self):
        return len(self.img_file_names)
    
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
            if len(e) > self.max_caption_size:
                unpadded[i] = unpadded[i][:self.max_caption_size]
        out[mask] = np.concatenate(unpadded)
        return torch.LongTensor(out)

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

    def load_data(self):
        bbox_path = os.path.join(self.preprocessor.data_dir, 'bounding_boxes.txt')
        class_path = os.path.join(self.preprocessor.data_dir, 'image_class_labels.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        classes = pd.read_csv(class_path,delim_whitespace=True,
                                        header=None)

        df_filenames = pd.read_csv(self.preprocessor.images_path_file,
                                   delim_whitespace=True, header=None)

        filenames = df_filenames[1].tolist()
        filename_bbox = {}
        filename_classes = {}
        numImgs = len(filenames)

        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i]
            filename_bbox[key] = bbox
            filename_classes[key] = classes[1][i]
        return filename_bbox, filename_classes

    def __getitem__(self, idx):
        image_name = self.preprocessor.data_path + self.img_file_names[idx]
        image = get_imgs(image_name, self.filename_bbox[self.img_file_names[idx]], self.opts)
        # select a random sentence
        cap_idx = np.random.choice(np.arange(len(self.img_captions[idx])))
        caption = self.img_captions[idx][cap_idx]

        class_id = numpy.array(self.filename_class[self.img_file_names[idx]])

        return image, caption, class_id
